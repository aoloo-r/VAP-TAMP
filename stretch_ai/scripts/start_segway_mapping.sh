#!/bin/bash

# Start Segway mapping with proper configuration
# Usage: ./start_segway_mapping.sh [robot_ip]

ROBOT_IP=${1:-localhost}

echo "Starting Segway Mapping System"
echo "=============================="
echo "Robot IP: $ROBOT_IP"
echo ""

# Check if bridge is running
echo "Checking ZMQ bridge connection..."
python3 -c "
import zmq
ctx = zmq.Context()
sock = ctx.socket(zmq.SUB)
sock.connect('tcp://$ROBOT_IP:4401')
sock.setsockopt(zmq.SUBSCRIBE, b'')
sock.setsockopt(zmq.RCVTIMEO, 2000)
try:
    msg = sock.recv_pyobj()
    print('✓ ZMQ bridge is running')
except:
    print('✗ ZMQ bridge not responding!')
    print('  Please start the bridge first: ./scripts/run_segway_bridge.sh')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "Bridge check failed. Exiting."
    exit 1
fi

# Run quick diagnostics
echo ""
echo "Running diagnostics..."
echo "---------------------"
python3 scripts/zmq_rate_debug.py $ROBOT_IP 5

# Test observation reception
echo ""
echo "Testing observation reception..."
echo "-------------------------------"
python3 scripts/test_mapping_reception.py $ROBOT_IP 10

# Ask user to continue
echo ""
read -p "Continue with mapping? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Mapping cancelled."
    exit 0
fi

# Start mapping with proper settings
echo ""
echo "Starting teleoperation mapping..."
echo "================================"
echo "Instructions:"
echo "- Drive your robot to build the map"
echo "- Observations are collected automatically"
echo "- Checkpoints saved every 30 seconds"
echo "- Press Ctrl+C to stop and save final map"
echo ""

# Set environment variables for better performance
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Run mapping
python -m stretch.app.mapping \
    --robot_ip $ROBOT_IP \
    --teleop-mode \
    --no-semantic \
    --debug

echo ""
echo "Mapping session completed!"