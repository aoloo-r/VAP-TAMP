#!/usr/bin/env python3
"""Debug script to test robot observation receiving"""

import time
from stretch.core.client import RobotClient

def test_robot_observations(robot_ip="localhost"):
    """Test receiving observations from the robot"""
    print(f"Connecting to robot at {robot_ip}...")
    
    # Create robot client
    robot = RobotClient(robot_ip=robot_ip)
    
    print("Robot client created. Testing observation receiving...")
    print("-" * 60)
    
    # Debug: Check robot attributes
    print("\nDebug: Checking robot client attributes...")
    print(f"  Robot type: {type(robot)}")
    print(f"  Has 'running' attribute: {hasattr(robot, 'running')}")
    if hasattr(robot, 'running'):
        print(f"  Robot running: {robot.running}")
    print(f"  Available methods: {[m for m in dir(robot) if not m.startswith('_') and callable(getattr(robot, m))][:10]}...")
    
    # Test 1: Single observation
    print("\nTest 1: Getting single observation...")
    obs = robot.get_observation()
    if obs is None:
        print("ERROR: No observation received!")
    else:
        print(f"SUCCESS: Received observation")
        print(f"  Type: {type(obs)}")
        if hasattr(obs, 'timestamp'):
            print(f"  Timestamp: {obs.timestamp}")
        if hasattr(obs, 'lidar_timestamp'):
            print(f"  Lidar timestamp: {obs.lidar_timestamp}")
        if hasattr(obs, 'gps'):
            print(f"  GPS: {obs.gps}")
        if hasattr(obs, 'compass'):
            print(f"  Compass: {obs.compass}")
    
    # Test 2: Multiple observations
    print("\nTest 2: Getting observations for 5 seconds...")
    start_time = time.time()
    obs_count = 0
    timestamps = []
    
    while time.time() - start_time < 5.0:
        obs = robot.get_observation()
        if obs is not None:
            obs_count += 1
            if hasattr(obs, 'timestamp'):
                timestamps.append(obs.timestamp)
            elif hasattr(obs, 'lidar_timestamp'):
                timestamps.append(obs.lidar_timestamp)
        time.sleep(0.1)
    
    print(f"Received {obs_count} observations in 5 seconds")
    print(f"Rate: {obs_count/5.0:.1f} obs/s")
    
    # Check if timestamps are changing
    if len(timestamps) > 1:
        unique_timestamps = len(set(timestamps))
        print(f"Unique timestamps: {unique_timestamps}")
        if unique_timestamps == 1:
            print("WARNING: All observations have the same timestamp!")
    
    # Test 3: Check pose graph
    print("\nTest 3: Checking pose graph...")
    pose_graph = robot.get_pose_graph()
    if pose_graph is None:
        print("WARNING: Pose graph is None")
    else:
        print(f"Pose graph has {len(pose_graph)} vertices")
    
    print("\nDone!")

if __name__ == "__main__":
    test_robot_observations()