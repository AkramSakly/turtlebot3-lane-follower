#!/usr/bin/env python3
"""Record what the TurtleBot sees (/lane_overlay) to an MP4, with speed and mode text.

Usage (PC, while the follower is running):
    python3 ~/record_overlay.py                 # saves ~/Videos/robot_view_<time>.mp4
    python3 ~/record_overlay.py my_demo.mp4     # custom file name
Stop with Ctrl+C: the file is closed and saved.
"""
import os
import sys
import time

import cv2
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

FPS = 15
SCALE = 3  # 256x192 overlay -> 768x576 video


class Recorder(Node):
    def __init__(self, path):
        super().__init__('lane_overlay_recorder')
        self.path = path
        self.bridge = CvBridge()
        self.writer = None
        self.frames = 0
        self.speed = 0.0
        self.turn = 0.0
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Image, 'lane_overlay', self.on_image, qos)
        self.create_subscription(Twist, 'cmd_vel', self.on_cmd, 10)
        self.get_logger().info(f'Recording /lane_overlay to {path}. Ctrl+C to stop.')

    def on_cmd(self, msg):
        self.speed, self.turn = msg.linear.x, msg.angular.z

    def on_image(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        frame = cv2.resize(frame, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)
        if self.writer is None:
            h, w = frame.shape[:2]
            self.writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (w, h))
        text = f'speed {self.speed:.3f} m/s   turn {self.turn:+.2f} rad/s'
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(frame, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        self.writer.write(frame)
        self.frames += 1

    def close(self):
        if self.writer is not None:
            self.writer.release()
        print(f'Saved {self.frames} frames (~{self.frames / FPS:.0f} s) to {self.path}')


def main():
    default = os.path.expanduser(f'~/Videos/robot_view_{time.strftime("%Y%m%d_%H%M%S")}.mp4')
    path = sys.argv[1] if len(sys.argv) > 1 else default
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    rclpy.init()
    node = Recorder(path)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
