#!/bin/bash
# =============================================================================
# run_nvblox.sh - ROS Bridge for Direct ROS2 nvblox Integration
#
# This script sets up the ROS1->ROS2 bridge for nvblox integration with
# the new direct ROS2 mapping system. It replaces run_segway_bridge.sh
# by eliminating ZMQ components entirely.
#
# USAGE:
#   ./run_nvblox.sh
#
# Then in another terminal:
#   python ros2_nvblox_mapping.py --config ../config/ros2_nvblox_config.yaml --semantic
# =============================================================================

# --- Configuration ---
# IMPORTANT: Update these IP addresses to match your network setup
DESKTOP_IP=10.66.171.11 # IP address of this computer (running the bridge)
ROBOT_IP=10.66.171.10   # IP address of the robot (running ROS1 master)

# --- Environment Setup ---
export ROS_MASTER_URI="http://${ROBOT_IP}:11311"
export ROS_IP="${DESKTOP_IP}"
export ROS_HOSTNAME="${DESKTOP_IP}"

echo "======================================================="
echo "    ROS Bridge for Direct ROS2 nvblox Integration"
echo "======================================================="
echo ""
echo "--- Network Configuration ---"
echo "Desktop IP (ROS_IP):      ${DESKTOP_IP}"
echo "Robot IP (ROS_MASTER_URI): ${ROBOT_IP}"
echo "-----------------------------"
echo ""

# --- Cleanup Function ---
cleanup() {
  echo ""
  echo "Shutting down ROS bridge..."
  if [ -n "$BRIDGE_PID" ]; then
    echo " -> Killing ROS1-ROS2 bridge (PID $BRIDGE_PID)"
    kill "$BRIDGE_PID" 2>/dev/null
  fi
  echo "Bridge stopped successfully."
  echo ""
  echo "You can now stop your nvblox mapping application."
  exit 0
}
trap cleanup EXIT INT TERM

# --- Source ROS 2 ---
echo "Sourcing ROS 2 Humble..."
source /opt/ros/humble/setup.bash

# --- Start ROS 1 -> ROS 2 Bridge ---
# Using --bridge-all-topics is more robust as it guarantees that essential
# background topics like /tf and /tf_static are bridged correctly.
echo "Starting ROS1->ROS2 dynamic_bridge for all topics..."
ros2 run ros1_bridge dynamic_bridge --bridge-all-topics &
BRIDGE_PID=$!
sleep 3 # Give the bridge a moment to initialize
echo "  -> Bridge started with PID: $BRIDGE_PID"
echo ""
echo "======================================================="
echo "ROS BRIDGE IS NOW RUNNING (No ZMQ Server Required)"
echo "======================================================="
echo ""
echo "Expected topics being bridged:"
echo " • /camera/color/image_raw        (RGB images)"
echo " • /camera/depth/image_rect_raw   (Depth images)" 
echo " • /camera/color/camera_info      (Camera intrinsics)"
echo " • /joint_states                  (Robot joint states)"
echo " • /odom                          (Robot odometry)"
echo " • /tf, /tf_static                (Transforms)"
echo " • /nvblox_node/*                 (nvblox topics)"
echo ""
echo "Next: In another terminal, run your nvblox mapping:"
echo "  cd ~/code/stretch_ai/src/stretch/app"
echo "  python ros2_nvblox_mapping.py --config ../config/ros2_nvblox_config.yaml --semantic"
echo ""
echo "Press Ctrl+C in this terminal to stop the bridge."
echo "======================================================="
echo ""

# --- Wait for user to exit ---
# This will keep the script running until Ctrl+C is pressed,
# at which point the 'trap' will call the cleanup function.
wait $BRIDGE_PID