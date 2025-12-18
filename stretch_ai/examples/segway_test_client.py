#!/usr/bin/env python3

from stretch.agent import RobotClient
import time
import argparse

def main():
    parser = argparse.ArgumentParser(description='Test Segway with Stretch AI')
    parser.add_argument('--robot_ip', type=str, default='192.168.1.100',
                        help='IP address of the Segway robot')
    args = parser.parse_args()
    
    print(f"Connecting to Segway robot at {args.robot_ip}...")
    robot = RobotClient(robot_ip=args.robot_ip)
    
    # Test if we can get observations
    print("Attempting to get an observation...")
    try:
        obs = robot.get_observation()
        print("Successfully received observation!")
        
        # Print information about what we received
        if obs:
            print("\nObservation contents:")
            for key, value in obs.items():
                if hasattr(value, 'shape'):
                    print(f"  {key}: shape={value.shape}, type={type(value)}")
                else:
                    print(f"  {key}: type={type(value)}")
        
        print("\nTest successful!")
    except Exception as e:
        print(f"Error getting observation: {e}")
    
    print("Test complete")

if __name__ == "__main__":
    main()