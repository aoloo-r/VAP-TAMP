#!/bin/bash
# =============================================================================
# run_segway_bridge.sh (Corrected for Teleoperation Mapping)
#
#   1) Exports DESKTOP_IP and ROBOT_IP.
#   2) Sets ROS_MASTER_URI, ROS_IP, ROS_HOSTNAME for ROS1 communication.
#   3) Sources ROS 2 Humble.
#   4) Starts a ROS 1 -> ROS 2 dynamic_bridge for ALL topics to ensure
#      critical transforms (/tf, /tf_static) are included.
#   5) Launches home_zmq_server.py from a hardcoded path.
#   6) On Ctrl+C or exit, kills both processes cleanly.
#
# USAGE:
#   ./run_segway_bridge.sh
# =============================================================================

# --- Configuration ---
# IMPORTANT: Update these IP addresses to match your network setup.
DESKTOP_IP=172.20.10.4  # IP address of this computer (running the bridge and ZMQ server)
ROBOT_IP=172.20.10.2   # IP address of the Segway robot (running ROS1 master)

# --- Path to ZMQ Server Script ---
# The full path to your home_zmq_server.py script is now defined here.
ZMQ_SCRIPT_PATH="/home/aoloo/code/stretch_ai/scripts/segway_adapter/home_zmq_server.py"

# --- Script Validation ---
# This check ensures the script will not run if the path is incorrect.
if [ ! -f "$ZMQ_SCRIPT_PATH" ]; then
  echo "ERROR: Cannot find home_zmq_server.py at the specified path:"
  echo "  '$ZMQ_SCRIPT_PATH'"
  echo "Please update the ZMQ_SCRIPT_PATH variable in this script if the location has changed."
  exit 1
fi

# --- Environment Setup ---
export ROS_MASTER_URI="http://${ROBOT_IP}:11311"
export ROS_IP="${DESKTOP_IP}"
export ROS_HOSTNAME="${DESKTOP_IP}"

echo "--- Network Configuration ---"
echo "Desktop IP (ROS_IP):      ${DESKTOP_IP}"
echo "Robot IP (ROS_MASTER_URI): ${ROBOT_IP}"
echo "-----------------------------"
echo ""

# --- Cleanup Function ---
cleanup() {
  echo ""
  echo "Shutting down all processes..."
  if [ -n "$BRIDGE_PID" ]; then
    echo " -> Killing ROS1-ROS2 bridge (PID $BRIDGE_PID)"
    kill "$BRIDGE_PID" 2>/dev/null
  fi
  if [ -n "$ZMQ_PID" ]; then
    echo " -> Killing home_zmq_server (PID $ZMQ_PID)"
    kill "$ZMQ_PID" 2>/dev/null
  fi
  exit 0
}
trap cleanup EXIT INT TERM

# --- Source ROS 2 ---
echo "Sourcing ROS 2 Humble..."
source /opt/ros/humble/setup.bash

# --- Check for existing bridge processes ---
echo "🔍 Checking for existing bridge processes..."
EXISTING_BRIDGE=$(pgrep -f "ros1_bridge.*dynamic_bridge" | head -1)
if [ -n "$EXISTING_BRIDGE" ]; then
    echo "⚠️  Found existing bridge process (PID: $EXISTING_BRIDGE)"
    echo "   Killing existing bridge to avoid conflicts..."
    kill -TERM $EXISTING_BRIDGE 2>/dev/null
    sleep 3
    # Force kill if still running
    if kill -0 $EXISTING_BRIDGE 2>/dev/null; then
        echo "   Force killing stubborn bridge process..."
        kill -KILL $EXISTING_BRIDGE 2>/dev/null
    fi
    sleep 2
fi

# --- Start ROS 1 -> ROS 2 Bridge ---
# Using --bridge-all-topics is more robust as it guarantees that essential
# background topics like /tf and /tf_static are bridged correctly.
echo "Starting ROS1->ROS2 dynamic_bridge for all topics..."
ros2 run ros1_bridge dynamic_bridge --bridge-all-topics &
BRIDGE_PID=$!
sleep 5 # Give the bridge more time to initialize properly
echo "  -> Bridge started with PID: $BRIDGE_PID"

# Verify bridge is still running
sleep 2
if ! kill -0 $BRIDGE_PID 2>/dev/null; then
    echo "❌ Bridge process died immediately - check for conflicts"
    echo "   Try running: killall -9 dynamic_bridge"
    echo "   Then restart this script"
    exit 1
fi

# Wait for topics to stabilize
echo "⏳ Waiting 2 minutes for topics to stabilize before starting ZMQ server..."
sleep 120
echo "✅ Topics stabilized - starting ZMQ server"
echo ""

# --- Launch ZMQ Server ---
echo "Starting home_zmq_server.py (publishing to ports 4401/4403/4404)..."
# Activate conda environment that has stretch_ai dependencies
export CONDA_PATH="/home/aoloo/miniforge3"
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate stretch_ai
# Also source ROS2 in the conda environment
source /opt/ros/humble/setup.bash
python "$ZMQ_SCRIPT_PATH" &
ZMQ_PID=$!
sleep 2
echo "  -> ZMQ Server started with PID: $ZMQ_PID"
echo ""
echo "======================================================"
echo "All components are now running."
echo "You can now start the teleop_mapping.py on your client."
echo "Press Ctrl+C in this terminal to stop everything."
echo "======================================================"
echo ""

# --- Wait for user to exit ---
# This will keep the script running until Ctrl+C is pressed,
# at which point the 'trap' will call the cleanup function.
wait $ZMQ_PID
