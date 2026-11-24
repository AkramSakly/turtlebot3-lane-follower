"""Camera segmentation and differential drive control for ROS 2 Humble."""
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
from PIL import Image
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image as ImageMsg

from .vision import SIZE, lane_geometry, command, lane_side_of, recovery, recovery_command


class Follower(Node):
    def __init__(self):
        super().__init__('tb3_lane_center')
        share = Path(get_package_share_directory('tb3_lane_center'))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        self.model = ort.InferenceSession(str(share/'models/yellow_line.onnx'), options,
                                          providers=['CPUExecutionProvider'])
        self.white_model = ort.InferenceSession(str(share/'models/white_line.onnx'), options,
                                                providers=['CPUExecutionProvider'])
        self.lane_side = self.declare_parameter('lane_side', 'auto').value
        if self.lane_side not in ('auto', 'yellow_left', 'yellow_right'):
            raise ValueError("lane_side must be auto, yellow_left or yellow_right")
        self.max_age = float(self.declare_parameter('max_camera_age', 0.4).value)
        if not 0.05 <= self.max_age <= 1.0:
            raise ValueError('max_camera_age must be between 0.05 and 1.0 s')
        self.max_speed = float(self.declare_parameter('max_speed', 0.035).value)
        if not 0.0 < self.max_speed <= 0.22:
            raise ValueError('max_speed must be between 0 and 0.22 m/s')
        self.recovery_time = float(self.declare_parameter('recovery_time', 1.5).value)
        if not 0.0 <= self.recovery_time <= 5.0:
            raise ValueError('recovery_time must be between 0 and 5 s')
        self.side = {'yellow_left': 1, 'yellow_right': -1}.get(self.lane_side, 0)
        self.last_good = -1e9
        self.hold_time = float(self.declare_parameter('hold_time', 0.3).value)
        self.smoothing = float(self.declare_parameter('smoothing', 0.5).value)
        if not 0.0 <= self.smoothing <= 0.9:
            raise ValueError('smoothing must be between 0 and 0.9')
        self.last_lane_cmd = None
        self.filtered = (0., 0.)
        self.mode = ''
        self.bridge = CvBridge()
        self.cmd = self.create_publisher(Twist, 'lane_cmd_vel', 1)
        self.mask = self.create_publisher(ImageMsg, 'lane_yellow_mask', 1)
        self.overlay = self.create_publisher(ImageMsg, 'lane_overlay', 1)
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(ImageMsg, 'image_raw', self.frame, qos)
        self.deadline = 0.
        self.create_timer(.05, self.watchdog)
        self.get_logger().info(f'Model ready (lane_side={self.lane_side}, max_camera_age={self.max_age}s, max_speed={self.max_speed}m/s, recovery_time={self.recovery_time}s, hold_time={self.hold_time}s, smoothing={self.smoothing}). Waiting for camera frames on image_raw.')

    def set_mode(self, mode):
        if mode != self.mode:
            self.mode = mode
            self.get_logger().info(f'MODE: {mode}')

    def stop(self):
        self.filtered = (0., 0.)
        self.deadline = 0.
        self.cmd.publish(Twist())

    def watchdog(self):
        if time.monotonic() >= self.deadline:
            self.stop()

    def frame(self, msg):
        try:
            rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            frame = np.asarray(Image.fromarray(rgb).resize(SIZE, getattr(Image, 'Resampling', Image).BILINEAR))
            mask = self.model.run(['yellow_mask'], {'rgb': frame})[0]
            white = self.white_model.run(['white_mask'], {'rgb': frame})[0]
            geometry = lane_geometry(mask.astype(bool), white.astype(bool), self.lane_side)
            now = time.monotonic()
            target = None
            if geometry is not None:
                self.last_good = now
                if self.lane_side == 'auto':
                    self.side = lane_side_of(geometry)
            elif self.side and now - self.last_good <= self.recovery_time:
                target = recovery(mask.astype(bool), white.astype(bool), self.side)
            debug = self.bridge.cv2_to_imgmsg(mask*255, encoding='mono8')
            debug.header = msg.header
            self.mask.publish(debug)
            if self.overlay.get_subscription_count():
                overlay = frame.copy()
                pixels = mask.astype(bool)
                overlay[pixels] = .4*frame[pixels] + .6*np.array([255, 0, 180])
                pixels = white.astype(bool)
                overlay[pixels] = .4*frame[pixels] + .6*np.array([0, 210, 255])
                if geometry is not None:
                    for row, left, right in geometry:
                        mx, my = int(round((left+right)/2)), int(round(row))
                        overlay[my-2:my+3, mx-2:mx+3] = (0,255,0)
                elif target is not None:
                    tx = int(np.clip(round(target), 3, overlay.shape[1]-4))
                    overlay[160:171, tx-3:tx+4] = (255,140,0)
                view = self.bridge.cv2_to_imgmsg(overlay, encoding='rgb8')
                view.header = msg.header
                self.overlay.publish(view)
            stamp = msg.header.stamp
            age = (self.get_clock().now().nanoseconds - (stamp.sec*10**9 + stamp.nanosec))/1e9
            if not -.1 <= age <= self.max_age:
                self.stop()
                self.set_mode('STOP (camera too old)')
                self.get_logger().warning(f'Camera age {age:.3f}s: preview only, velocity zero.',
                                          throttle_duration_sec=2.)
                return
            cmd = Twist()
            if geometry is not None:
                self.set_mode('LANE')
                raw = command(geometry, mask.shape[1], self.max_speed)
                self.last_lane_cmd = raw
            elif self.last_lane_cmd is not None and time.monotonic() - self.last_good <= self.hold_time:
                self.set_mode('LANE (hold)')
                raw = self.last_lane_cmd
            elif target is not None:
                self.set_mode('RECOVERY')
                raw = recovery_command(target, mask.shape[1], self.max_speed)
            else:
                if time.monotonic() - self.last_good > self.recovery_time:
                    self.set_mode('STOP (recovery time over)' if self.mode == 'RECOVERY' else 'STOP (no lane)')
                else:
                    self.set_mode('STOP (no line in lower image)')
                self.stop()
                return
            a = self.smoothing
            self.filtered = (a*self.filtered[0] + (1-a)*raw[0], a*self.filtered[1] + (1-a)*raw[1])
            cmd.linear.x, cmd.angular.z = self.filtered
            self.deadline = time.monotonic() + min(.4, self.max_age + .1 - age)
            self.cmd.publish(cmd)
        except Exception as exc:
            self.stop()
            self.get_logger().error(str(exc), throttle_duration_sec=2.)


def main():
    rclpy.init()
    node = Follower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.stop()
            rclpy.shutdown()
        node.destroy_node()
