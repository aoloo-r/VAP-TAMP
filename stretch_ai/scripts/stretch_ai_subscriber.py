import zmq
import pickle
import cv2
import numpy as np
import argparse
import time
import torch

# --- Stretch AI Imports ---
from stretch.core.interfaces import Observations
from stretch.mapping.voxel import SparseVoxelMap
from stretch.perception.wrapper import OvmmPerception
from stretch.core.parameters import get_parameters

def main(args):
    """
    Connects to the ZMQ publisher, receives frames, resizes them to a consistent
    resolution, and builds a VoxelMap with instance memory.
    """
    # --- MODIFIED: Define the desired resolution ---
    FIXED_WIDTH = 640
    FIXED_HEIGHT = 480

    # --- ZMQ Setup ---
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect("tcp://localhost:5555")
    socket.setsockopt_string(zmq.SUBSCRIBE, "record3d_frames")
    print(f"ZMQ Subscriber connected to tcp://localhost:5555, processing at {FIXED_WIDTH}x{FIXED_HEIGHT}")

    # --- Initialize Stretch AI Components ---
    print("Initializing Stretch AI components...")
    try:
        parameters = get_parameters(args.config_path)
    except FileNotFoundError:
        print(f"\nERROR: Config file not found at: {args.config_path}")
        print("       Please provide the correct path using --config_path")
        return
        
    detector = OvmmPerception(parameters)
    voxel_map = SparseVoxelMap.from_parameters(parameters, None, use_instance_memory=True)
    print("Stretch AI components initialized.")

    try:
        while True:
            topic, data = socket.recv_multipart()
            data_packet = pickle.loads(data)

            frame_id = data_packet['frame_id']
            rgb_image = data_packet['rgb']
            depth_image_mm = data_packet['depth']
            camera_pose = data_packet['pose']
            intrinsics = data_packet['intrinsics']

            print(f"Received frame {frame_id}")

            # --- Get original dimensions for scaling intrinsics ---
            original_height, original_width, _ = rgb_image.shape

            # --- Resize images to the consistent, desired size ---
            rgb_image_resized = cv2.resize(rgb_image, (FIXED_WIDTH, FIXED_HEIGHT), interpolation=cv2.INTER_AREA)
            depth_image_mm_resized = cv2.resize(depth_image_mm, (FIXED_WIDTH, FIXED_HEIGHT), interpolation=cv2.INTER_NEAREST)

            # --- ADDED: Scale the camera intrinsics to match the new resolution ---
            scale_x = FIXED_WIDTH / original_width
            scale_y = FIXED_HEIGHT / original_height
            
            intrinsics_scaled = np.copy(intrinsics)
            intrinsics_scaled[0, 0] *= scale_x  # fx
            intrinsics_scaled[1, 1] *= scale_y  # fy
            intrinsics_scaled[0, 2] *= scale_x  # cx
            intrinsics_scaled[1, 2] *= scale_y  # cy
            
            # Convert depth from millimeters to meters
            depth_image_m = depth_image_mm_resized.astype(np.float32) / 1000.0

            # Create a comprehensive Observations object
            x = camera_pose[0, 3]
            y = camera_pose[1, 3]
            theta = np.arctan2(camera_pose[1, 0], camera_pose[0, 0])
            
            obs = Observations(
                rgb=rgb_image_resized,
                depth=depth_image_m,
                camera_pose=camera_pose,
                camera_K=intrinsics_scaled, # Use the new, scaled intrinsics
                gps=np.array([x, y]),
                compass=np.array([theta])
            )

            # Run Perception
            obs_with_detections = detector.predict(obs)
            num_objects = len(obs_with_detections.task_observations.get('instance_classes', []))
            print(f"--> Detected {num_objects} objects.")

            # Add to Voxel Map
            voxel_map.add_obs(obs_with_detections)

    except KeyboardInterrupt:
        print("\nSubscriber stopped by user.")
        print("Post-processing and merging instances...")
        voxel_map.postprocess_instances()

    finally:
        print(f"Saving map to {args.output_file}...")
        voxel_map.write_to_pickle(args.output_file)
        print("Map saved successfully.")
        socket.close()
        context.term()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZMQ Subscriber for Stretch AI")
    parser.add_argument(
        '--config_path', 
        type=str, 
        default='/home/aoloo/code/stretch_ai/src/stretch/config/default_planner.yaml', 
        help='Path to the Stretch AI configuration file.'
    )
    parser.add_argument(
        '--output_file', 
        type=str, 
        default='final_semantic_map.pkl', 
        help='Output file for the saved map.'
    )
    args = parser.parse_args()
    main(args)

