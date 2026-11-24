#!/bin/bash
# Start the whole TurtleBot3 lane follower: adds 5 tabs to the terminal window you run it from.
#
# Usage:
#   ./start_robot.sh            -> preview only (robot does NOT move)
#   ./start_robot.sh drive      -> robot drives
#   ./start_robot.sh drive 0.05 -> robot drives with max_speed 0.05

# ---------------- EDIT THESE ----------------
PI_USER=ubuntu
PI_IP=192.168.0.100        # <-- put your Raspberry Pi IP here
DOMAIN=30
LDS=LDS-01                 # or LDS-02
LANE_SIDE=yellow_left
MAX_AGE=0.6
RECOVERY=3.0
# --------------------------------------------

MODE=${1:-preview}
SPEED=${2:-0.035}

PI_ENV="source /opt/ros/humble/setup.bash; [ -f ~/turtlebot3_ws/install/setup.bash ] && source ~/turtlebot3_ws/install/setup.bash; export TURTLEBOT3_MODEL=burger LDS_MODEL=$LDS ROS_DOMAIN_ID=$DOMAIN"
PC_ENV="source /opt/ros/humble/setup.bash; source ~/tb3_dl_ws/install/setup.bash; export ROS_DOMAIN_ID=$DOMAIN"

FOLLOW="trap true INT; ros2 run tb3_lane_center follow --ros-args -r image_raw:=/camera/image_raw -p lane_side:=$LANE_SIDE -p max_camera_age:=$MAX_AGE -p max_speed:=$SPEED -p recovery_time:=$RECOVERY"
if [ "$MODE" = "drive" ]; then
  FOLLOW="$FOLLOW -r lane_cmd_vel:=/cmd_vel"
  ECHO_TOPIC=/cmd_vel
  echo ">>> DRIVE MODE at max_speed=$SPEED. Robot will MOVE. Keep your hand near the switch."
else
  ECHO_TOPIC=/lane_cmd_vel
  echo ">>> PREVIEW MODE. Robot will NOT move."
fi
# After Ctrl+C in the follower tab, always send one stop command.
FOLLOW="$FOLLOW; echo 'Sending stop...'; ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist '{}'"

gnome-terminal --tab --title="Pi base" -- bash -c \
  "ssh -t $PI_USER@$PI_IP \"bash -ic '$PI_ENV; ros2 launch turtlebot3_bringup robot.launch.py'\"; exec bash"

gnome-terminal --tab --title="Pi camera" -- bash -c \
  "sleep 3; ssh -t $PI_USER@$PI_IP \"bash -ic '$PI_ENV; ros2 run camera_ros camera_node --ros-args -p format:=RGB888 -p width:=320 -p height:=240'\"; exec bash"

gnome-terminal --tab --title="Follower" -- bash -c \
  "$PC_ENV; echo 'Waiting 12 s for robot and camera...'; sleep 12; $FOLLOW; exec bash"

gnome-terminal --tab --title="rqt" -- bash -c \
  "$PC_ENV; sleep 14; ros2 run rqt_image_view rqt_image_view /lane_overlay; exec bash"

gnome-terminal --tab --title="Echo" -- bash -c \
  "$PC_ENV; sleep 16; ros2 topic echo $ECHO_TOPIC; exec bash"
