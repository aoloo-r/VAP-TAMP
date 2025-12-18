#!/usr/bin/env python3
"""
Script to find valid start poses for a given map pickle file.
This helps resolve the "you need to manually set the start pose to be valid" error.
"""

import numpy as np
import sys
import pickle
from pathlib import Path

# Add the stretch_ai src to Python path
import sys
sys.path.append('/home/aoloo/code/stretch_ai/src')

from stretch.agent import RobotAgent
from stretch.core import get_parameters
from stretch.perception import create_semantic_sensor
from stretch.utils.dummy_stretch_client import DummyStretchClient


def find_valid_start_poses(pickle_file, num_candidates=20):
    """
    Load a map and find valid start poses by sampling different positions.
    """
    print(f"Loading map from: {pickle_file}")
    
    # Load parameters (same as vlm_planning.py)
    parameters = get_parameters("default_planner.yaml")
    
    # Create semantic sensor
    semantic_sensor = create_semantic_sensor(parameters)
    
    # Create dummy robot client (same as vlm_planning.py)
    robot = DummyStretchClient()
    
    # Create robot agent
    robot_agent = RobotAgent(
        robot=robot,
        parameters=parameters,
        semantic_sensor=semantic_sensor,
    )
    
    # Load the map (same as vlm_planning.py)
    voxel_map = robot_agent.get_voxel_map()
    voxel_map.read_from_pickle(pickle_file, perception=semantic_sensor)
    
    print("Agent loaded successfully")
    
    # Get navigation space for collision checking
    space = robot_agent.get_navigation_space()
    
    print(f"Searching for {num_candidates} valid start poses...")
    
    # Try different positions in a grid pattern
    valid_poses = []
    tested_poses = []
    
    # Search in a reasonable area around the map center
    for x in np.linspace(-3, 3, 10):
        for y in np.linspace(-3, 3, 10):
            for theta in [0, np.pi/2, np.pi, 3*np.pi/2]:  # Try 4 orientations
                pose = np.array([x, y, theta])
                tested_poses.append(pose.copy())
                
                is_valid = space.is_valid(pose, verbose=False, debug=False)
                if is_valid:
                    valid_poses.append(pose.copy())
                    print(f"✅ Valid pose found: x={x:.2f}, y={y:.2f}, theta={theta:.2f}")
                    
                    if len(valid_poses) >= num_candidates:
                        break
            if len(valid_poses) >= num_candidates:
                break
        if len(valid_poses) >= num_candidates:
            break
    
    print(f"\nResults:")
    print(f"Tested {len(tested_poses)} poses")
    print(f"Found {len(valid_poses)} valid poses")
    
    if valid_poses:
        print(f"\nTop 5 valid start poses:")
        for i, pose in enumerate(valid_poses[:5]):
            print(f"{i+1}. x={pose[0]:.3f}, y={pose[1]:.3f}, theta={pose[2]:.3f}")
        
        print(f"\nTo use the first valid pose, set in vlm_planning.py:")
        pose = valid_poses[0]
        print(f"x0 = np.array([{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}])")
        
    else:
        print("❌ No valid poses found! The map may have issues or all areas are occupied.")
        
        # Let's check what's wrong with [0,0,0]
        print(f"\nDebugging default pose [0, 0, 0]:")
        default_pose = np.array([0, 0, 0])
        space.is_valid(default_pose, verbose=True, debug=True)
    
    return valid_poses


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python find_valid_start_pose.py <map.pkl>")
        sys.exit(1)
    
    pickle_file = sys.argv[1]
    if not Path(pickle_file).exists():
        print(f"Error: File {pickle_file} does not exist")
        sys.exit(1)
    
    try:
        valid_poses = find_valid_start_poses(pickle_file)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)