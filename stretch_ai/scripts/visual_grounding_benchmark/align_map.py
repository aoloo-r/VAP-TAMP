import torch
import pickle as pkl
import argparse
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import sys

# Your robust import solution
from rosbags.rosbag1 import Reader
from rosbags.serde import ros1_to_cdr
from rosbags.typesys import get_typestore, Stores

def get_xyz_coordinates(depth, pose, intrinsic):
    """Calculates the 3D world coordinates from a depth map, camera pose, and intrinsics."""
    _, height, width = depth.shape
    xs, ys = torch.meshgrid(
        torch.arange(width, device=depth.device),
        torch.arange(height, device=depth.device),
        indexing="xy",
    )
    x = (xs - intrinsic[0, 2]) / intrinsic[0, 0]
    y = (ys - intrinsic[1, 2]) / intrinsic[1, 1]
    z = depth[0]
    camera_coords = torch.stack((x * z, y * z, z, torch.ones_like(z)), axis=-1)
    world_coords = camera_coords @ pose.T
    return world_coords[..., :3]

def transform_to_matrix(transform):
    """Converts a ROS Transform message to a 4x4 numpy matrix."""
    trans = transform.translation
    rot = transform.rotation
    rotation_matrix = Rotation.from_quat([rot.x, rot.y, rot.z, rot.w]).as_matrix()
    matrix = np.eye(4)
    matrix[0:3, 0:3] = rotation_matrix
    matrix[0:3, 3] = [trans.x, trans.y, trans.z]
    return matrix

def main():
    parser = argparse.ArgumentParser(
        description="Aligns a 3D map from a .pkl file with a ROS 1 trajectory from a .bag file."
    )
    parser.add_argument("--pkl_file", type=str, required=True, help="Path to the unaligned .pkl file.")
    parser.add_argument("--bag_file", type=str, required=True, help="Path to the .bag file with /tf and /tf_static.")
    parser.add_argument("--output_file", type=str, required=True, help="Path for the final aligned .pkl file.")
    parser.add_argument("--robot_base_frame", type=str, default="base_link", help="The robot's base frame ID.")
    parser.add_argument("--camera_frame", type=str, default="iphone_camera_link", help="The camera's frame ID.")
    parser.add_argument("--map_frame", type=str, default="map", help="The ROS map frame ID.")
    args = parser.parse_args()

    print(f"Loading data from {args.pkl_file}...")
    with open(args.pkl_file, 'rb') as file:
        data = pkl.load(file)

    typestore = get_typestore(Stores.ROS2_FOXY)
    
    # --- NEW: Diagnostic Pre-Check ---
    print(f"\n--- Pre-check: Inspecting static transforms in {args.bag_file} ---")
    has_static_topic = False
    try:
        with Reader(args.bag_file) as reader:
            static_connections = [c for c in reader.connections if c.topic == '/tf_static']
            if not static_connections:
                print("WARNING: The topic '/tf_static' was NOT found in this bag file.")
            else:
                has_static_topic = True
                print("Found the following static transforms (parent -> child):")
                for conn, ts, raw in reader.messages(connections=static_connections):
                    msg = typestore.deserialize_cdr(ros1_to_cdr(raw, conn.msgtype), conn.msgtype)
                    for tf in msg.transforms:
                        print(f"  - '{tf.header.frame_id}' -> '{tf.child_frame_id}'")
    except Exception as e:
        print(f"Could not inspect bag file due to an error: {e}")
    print("--- End of Pre-check ---\n")


    print(f"Reading and processing transforms from {args.bag_file}...")
    base_to_camera_tf = None
    map_to_odom_transforms = {}
    odom_to_base_transforms = {}

    try:
        with Reader(args.bag_file) as reader:
            connections = [c for c in reader.connections if c.topic in ['/tf', '/tf_static']]
            for connection, timestamp, rawdata in reader.messages(connections=connections):
                try:
                    msg = typestore.deserialize_cdr(ros1_to_cdr(rawdata, connection.msgtype), connection.msgtype)
                except Exception as e:
                    print(f"Warning: Skipping message on topic '{connection.topic}' due to deserialization error: {e}", file=sys.stderr)
                    continue

                for transform_stamped in msg.transforms:
                    stamp = transform_stamped.header.stamp
                    timestamp_sec = stamp.sec + stamp.nanosec / 1e9
                    
                    # Look for base_link -> camera transform in /tf (from your static publisher)
                    if (transform_stamped.header.frame_id == args.robot_base_frame and
                        transform_stamped.child_frame_id == args.camera_frame and base_to_camera_tf is None):
                        base_to_camera_tf = transform_to_matrix(transform_stamped.transform)
                        print(f"Found camera transform: '{args.robot_base_frame}' -> '{args.camera_frame}'")
                    
                    # Collect map -> odom transforms
                    elif (transform_stamped.header.frame_id == args.map_frame and
                          transform_stamped.child_frame_id == 'odom'):
                        matrix = transform_to_matrix(transform_stamped.transform)
                        map_to_odom_transforms[timestamp_sec] = matrix
                    
                    # Collect odom -> base_link transforms  
                    elif (transform_stamped.header.frame_id == 'odom' and
                          transform_stamped.child_frame_id == args.robot_base_frame):
                        matrix = transform_to_matrix(transform_stamped.transform)
                        odom_to_base_transforms[timestamp_sec] = matrix
    except Exception as e:
        print(f"A critical error occurred while reading the bag file: {e}")
        return

    # Build trajectory by combining map->odom and odom->base_link transforms
    trajectory = []
    map_times = sorted(map_to_odom_transforms.keys())
    odom_times = sorted(odom_to_base_transforms.keys())
    
    if map_times and odom_times:
        print(f"Combining {len(map_times)} map->odom and {len(odom_times)} odom->base transforms...")
        map_times_arr = np.array(map_times)
        
        for odom_time, odom_to_base in odom_to_base_transforms.items():
            # Find closest map->odom transform
            closest_map_idx = np.abs(map_times_arr - odom_time).argmin()
            closest_map_time = map_times[closest_map_idx]
            map_to_odom = map_to_odom_transforms[closest_map_time]
            
            # Combine: map -> odom -> base_link
            map_to_base = map_to_odom @ odom_to_base
            trajectory.append((odom_time, map_to_base))

    if base_to_camera_tf is None:
        print(f"\nWARNING: Could not find static transform ('{args.robot_base_frame}' -> '{args.camera_frame}') in bag file.")
        if args.camera_frame == 'iphone_camera_link' and args.robot_base_frame == 'base_link':
            print("Using manual iPhone camera transform: translation=(0.12, 0, 1.2), rotation=(0, 0, 0, 1)")
            # Create the transform matrix manually based on your static transform
            base_to_camera_tf = np.eye(4)
            base_to_camera_tf[0:3, 3] = [0.12, 0, 1.2]  # translation
            # rotation is identity (0,0,0,1 quaternion = no rotation)
        else:
            print("Please check the Pre-check report above and verify your --robot_base_frame and --camera_frame arguments.")
            print("Exiting.")
            return

    if not trajectory:
        print(f"ERROR: No transforms found from '{args.map_frame}' to '{args.robot_base_frame}'. Exiting.")
        return
        
    trajectory.sort(key=lambda x: x[0])
    traj_times = np.array([t for t, m in trajectory])

    print("Aligning camera frames to robot trajectory...")
    
    # Check if timestamps are valid
    valid_timestamps = True
    try:
        if len(data['timestamps']) > 0:
            first_ts = data['timestamps'][0]
            if hasattr(first_ts, 'shape') and len(first_ts.shape) > 0:
                valid_timestamps = False
    except:
        valid_timestamps = False
    
    if not valid_timestamps:
        print("WARNING: Invalid timestamps detected. Using frame order and bag duration.")
        # Create synthetic timestamps based on frame order and bag duration
        num_frames = len(data['rgb'])
        bag_start_time = min(traj_times)
        bag_end_time = max(traj_times)
        bag_duration = bag_end_time - bag_start_time
        
        synthetic_timestamps = []
        for i in range(num_frames):
            # Distribute frames evenly across bag duration
            frame_time = bag_start_time + (i / (num_frames - 1)) * bag_duration
            synthetic_timestamps.append(frame_time)
        
        timestamps_to_use = synthetic_timestamps
        print(f"Created {len(synthetic_timestamps)} synthetic timestamps from {bag_start_time:.2f}s to {bag_end_time:.2f}s")
    else:
        timestamps_to_use = [ts.item() if hasattr(ts, 'item') else ts for ts in data['timestamps']]
    
    aligned_base_poses = []
    for timestamp in tqdm(timestamps_to_use, desc="Matching timestamps"):
        closest_idx = np.abs(traj_times - timestamp).argmin()
        transform_matrix = trajectory[closest_idx][1]  # 4x4 transformation matrix
        
        # Extract x, y, theta from the transformation matrix
        x = transform_matrix[0, 3]  # translation x
        y = transform_matrix[1, 3]  # translation y
        
        # Extract rotation angle (theta) from the rotation matrix
        # For a 2D rotation in the x-y plane, theta = atan2(R21, R11)
        theta = np.arctan2(transform_matrix[1, 0], transform_matrix[0, 0])
        
        # Create the [x, y, theta] tensor format expected by VLM planning
        base_pose_xyt = torch.tensor([x, y, theta], dtype=torch.float32)
        aligned_base_poses.append(base_pose_xyt)

    data['base_poses'] = aligned_base_poses

    print("Recalculating 3D point cloud in map frame...")
    aligned_world_xyz = []
    for i in tqdm(range(len(data['rgb'])), desc="Generating point clouds"):
        # Get the corresponding 4x4 matrix from trajectory for camera pose computation
        timestamp = timestamps_to_use[i] 
        closest_idx = np.abs(traj_times - timestamp).argmin()
        map_to_base_pose_4x4 = trajectory[closest_idx][1]  # Full 4x4 matrix for computation
        
        world_camera_pose = map_to_base_pose_4x4 @ base_to_camera_tf
        world_camera_pose_torch = torch.from_numpy(world_camera_pose.astype(np.float32))
        depth = data['depth'][i].float()
        intrinsics = data['camera_K'][i].float()
        xyz_points = get_xyz_coordinates(depth.unsqueeze(0), world_camera_pose_torch, intrinsics)
        aligned_world_xyz.append(xyz_points)

    data['world_xyz'] = aligned_world_xyz

    print(f"Saving aligned map to {args.output_file}...")
    with open(args.output_file, 'wb') as file:
        pkl.dump(data, file)
    
    print("Alignment complete!")

if __name__ == "__main__":
    main()

