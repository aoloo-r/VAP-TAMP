#!/usr/bin/env python3
import time
import click
import zmq
import pickle
import numpy as np

@click.command()
@click.option("--robot-ip", required=True, help="IP address of the robot")
def main(robot_ip):
    """Simple program to read and display joint states."""
    print(f"Connecting to robot at {robot_ip}...")
    
    # Create ZMQ context and sockets
    context = zmq.Context()
    
    # Socket for low-level state (contains joint state)
    llstate_socket = context.socket(zmq.SUB)
    llstate_socket.setsockopt(zmq.SUBSCRIBE, b"")
    llstate_socket.connect(f"tcp://{robot_ip}:4403")
    
    # Set a shorter timeout
    llstate_socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second timeout
    
    print("Waiting for joint state data...")
    
    try:
        while True:
            try:
                # Try to receive low-level state
                message = llstate_socket.recv()
                state_data = pickle.loads(message)
                
                # Extract joint positions
                if 'joint_positions' in state_data:
                    joint_positions = state_data['joint_positions']
                    
                    # Get arm, lift, gripper positions with safe defaults
                    arm = joint_positions.get(4, 0.0) if joint_positions else 0.0
                    lift = joint_positions.get(3, 0.0) if joint_positions else 0.0
                    gripper = joint_positions.get(5, 0.0) if joint_positions else 0.0
                    
                    print(f"Arm: {arm:.3f}, Lift: {lift:.3f}, Gripper: {gripper:.3f}")
                else:
                    print("No joint positions in data")
                    
            except zmq.error.Again:
                print("Waiting for data... (timeout)")
            except Exception as e:
                print(f"Error receiving data: {e}")
            
            # Sleep to avoid hammering
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        # Clean up
        llstate_socket.close()
        context.term()

if __name__ == "__main__":
    main()
