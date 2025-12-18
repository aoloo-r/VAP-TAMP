#!/usr/bin/env python3

import pickle
import numpy as np
import cv2
import time
import click
from pathlib import Path
import datetime
from typing import Optional, Dict, Any, Tuple
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Add the scripts directory to path to import record3d_capture
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "scripts"))

class StreamToPklCapture:
    def __init__(self, output_dir: str = ".", target_height: int = 480, target_width: int = 640):
        """Initialize the direct stream capture
        
        Args:
            output_dir: Directory to save pkl files
            target_height: Target height to resize all frames to
            target_width: Target width to resize all frames to
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Target dimensions for consistent sizing
        self.target_height = target_height
        self.target_width = target_width
        
        # Data storage matching r3d_to_pkl.py format
        self.data = {
            'base_poses': [],
            'feats': [],
            'obs': [],
            'xyz': [],
            'world_xyz': [],
            'rgb': [],
            'depth': [],
            'camera_K': [],
            'camera_poses': []
        }
        
        self.frame_count = 0
        print(f"Will resize all frames to {target_width}x{target_height} for consistency")
        
        # Create synthetic camera poses for mapping (since Record3D WebRTC doesn't provide real poses)
        self.generate_synthetic_poses = True
        self.pose_step = 0.05  # 5cm forward movement per frame
        
    def compute_xyz_from_depth(self, depth: np.ndarray, camera_K: np.ndarray) -> np.ndarray:
        """Compute XYZ coordinates from depth and camera intrinsics"""
        height, width = depth.shape
        
        # Create pixel grid
        x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
        
        # Convert to camera coordinates
        fx, fy = camera_K[0, 0], camera_K[1, 1]
        cx, cy = camera_K[0, 2], camera_K[1, 2]
        
        x = (x_coords - cx) * depth / fx
        y = (y_coords - cy) * depth / fy
        z = depth
        
        # Stack to create XYZ array
        xyz = np.stack([x, y, z], axis=-1)
        return xyz
    
    def resize_and_adjust_intrinsics(self, rgb: np.ndarray, depth: np.ndarray, intrinsics: Optional[np.ndarray] = None):
        """Resize images to target dimensions and adjust camera intrinsics accordingly
        
        Args:
            rgb: RGB image array
            depth: Depth image array 
            intrinsics: 3x3 camera intrinsics matrix
            
        Returns:
            tuple: (resized_rgb, resized_depth, adjusted_intrinsics)
        """
        original_height, original_width = rgb.shape[:2]
        
        # Resize RGB image
        resized_rgb = cv2.resize(rgb, (self.target_width, self.target_height), interpolation=cv2.INTER_LINEAR)
        
        # Convert depth from millimeters to meters and resize
        # Record3D outputs depth in mm, but voxel mapping expects meters
        depth_meters = depth.astype(np.float32) / 1000.0  # Convert mm to meters
        
        # Resize depth image
        resized_depth = cv2.resize(depth_meters, (self.target_width, self.target_height), interpolation=cv2.INTER_NEAREST)
        
        # Adjust intrinsics matrix for the new resolution
        if intrinsics is not None:
            scale_x = self.target_width / original_width
            scale_y = self.target_height / original_height
            
            adjusted_intrinsics = intrinsics.copy()
            # Scale focal lengths
            adjusted_intrinsics[0, 0] *= scale_x  # fx
            adjusted_intrinsics[1, 1] *= scale_y  # fy
            # Scale principal points
            adjusted_intrinsics[0, 2] *= scale_x  # cx
            adjusted_intrinsics[1, 2] *= scale_y  # cy
        else:
            adjusted_intrinsics = intrinsics
            
        return resized_rgb, resized_depth, adjusted_intrinsics

    def add_frame(self, rgb: np.ndarray, depth: np.ndarray, intrinsics: Optional[np.ndarray] = None, pose: Optional[np.ndarray] = None):
        """Add a single frame to the dataset
        
        Args:
            rgb: RGB image array
            depth: Depth image array
            intrinsics: 3x3 camera intrinsics matrix
            pose: 4x4 camera pose matrix
        """
        # Resize images to consistent dimensions and adjust intrinsics
        resized_rgb, resized_depth, adjusted_intrinsics = self.resize_and_adjust_intrinsics(
            rgb, depth, intrinsics
        )
        
        # Generate synthetic camera pose if needed
        if self.generate_synthetic_poses and (pose is None or np.allclose(pose, np.eye(4))):
            # Create a simple forward movement trajectory
            synthetic_pose = np.eye(4, dtype=np.float32)
            synthetic_pose[2, 3] = self.frame_count * self.pose_step  # Move forward along Z-axis
            pose = synthetic_pose
            
        # Compute XYZ coordinates from depth and intrinsics
        xyz = None
        world_xyz = None
        if adjusted_intrinsics is not None:
            xyz = self.compute_xyz_from_depth(resized_depth, adjusted_intrinsics)
            
            # Convert to world coordinates if pose is available
            if pose is not None:
                # Reshape xyz for matrix multiplication
                original_shape = xyz.shape
                xyz_flat = xyz.reshape(-1, 3)
                
                # Add homogeneous coordinate
                ones = np.ones((xyz_flat.shape[0], 1), dtype=np.float32)
                xyz_homo = np.hstack([xyz_flat, ones])
                
                # Transform to world coordinates
                world_xyz_homo = xyz_homo @ pose.T
                world_xyz = world_xyz_homo[:, :3]
                
                # Reshape back to original shape
                world_xyz = world_xyz.reshape(original_shape)
        
        # Convert numpy arrays to tensors if torch is available (for compatibility)
        if TORCH_AVAILABLE and adjusted_intrinsics is not None:
            intrinsics_tensor = torch.from_numpy(adjusted_intrinsics.astype(np.float32))
        else:
            intrinsics_tensor = adjusted_intrinsics
        
        if TORCH_AVAILABLE and pose is not None:
            pose_tensor = torch.from_numpy(pose.astype(np.float32))
        else:
            pose_tensor = pose
        
        # Store core data (following r3d_to_pkl.py pattern)
        self.data['rgb'].append(resized_rgb)
        self.data['depth'].append(resized_depth)
        self.data['camera_K'].append(intrinsics_tensor)
        self.data['camera_poses'].append(pose_tensor)
        
        # Store computed XYZ data
        self.data['xyz'].append(xyz)
        self.data['world_xyz'].append(world_xyz)
        
        # Placeholder fields that can be None
        self.data['base_poses'].append(None)
        self.data['feats'].append(None) 
        self.data['obs'].append(None)
        
        self.frame_count += 1
        if self.frame_count == 1:
            print(f"Resized first frame from {rgb.shape[:2]} to {resized_rgb.shape[:2]}")
            if xyz is not None:
                print(f"Computed XYZ coordinates: {xyz.shape}")
        if self.frame_count % 10 == 0:
            print(f"Added frame {self.frame_count}")
    
    def capture_from_record3d_client(self, client, max_frames: int = 100) -> str:
        """Capture frames from an existing Record3D client instance
        
        Args:
            client: Record3DWebRTCClient instance that's already running
            max_frames: Maximum number of frames to capture
            
        Returns:
            Path to saved pkl file
        """
        print(f"Starting capture from Record3D client...")
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_filename = self.output_dir / f"record3d_capture_{timestamp}.pkl"
        
        frames_captured = 0
        start_time = time.time()
        timeout = 30  # 30 second timeout
        
        print(f"Capturing {max_frames} frames from Record3D stream...")
        
        try:
            while frames_captured < max_frames:
                # Get the latest frame from the client
                frame_data = client.get_latest_frame()
                
                if frame_data is not None:
                    # Add frame to our dataset
                    self.add_frame(
                        frame_data['rgb'],
                        frame_data['depth'], 
                        frame_data['intrinsics'],
                        frame_data['pose']
                    )
                    frames_captured += 1
                    
                    if frames_captured % 10 == 0:
                        print(f"Captured {frames_captured}/{max_frames} Record3D frames")
                else:
                    # No new frame available, wait a bit
                    time.sleep(0.1)
                
                # Timeout check
                if time.time() - start_time > timeout:
                    print(f"Timeout after {timeout}s, captured {frames_captured} frames")
                    break
            
            # Save the captured data
            print(f"\nSaving {frames_captured} frames to {output_filename}")
            with open(output_filename, 'wb') as f:
                pickle.dump(self.data, f)
                
            print(f"Successfully saved pkl file: {output_filename}")
            return str(output_filename)
            
        except Exception as e:
            print(f"Error during Record3D capture: {e}")
            import traceback
            traceback.print_exc()
            return ""


    def capture_from_running_record3d(self, max_frames: int = 100) -> str:
        """Capture frames from the running record3d_capture.py process
        
        Reads data from shared file written by record3d_capture.py
        
        Args:
            max_frames: Maximum number of frames to capture
            
        Returns:
            Path to saved pkl file
        """
        print("Looking for running record3d_capture.py process...")
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_filename = self.output_dir / f"record3d_stream_{timestamp}.pkl"
        
        shared_data_file = Path("/tmp/record3d_latest_frame.pkl")
        
        frames_captured = 0
        start_time = time.time()
        timeout = 30  # 30 second timeout
        last_frame_id = -1
        no_data_count = 0
        
        print(f"Reading frames from shared file: {shared_data_file}")
        print(f"Capturing {max_frames} frames...")
        
        try:
            while frames_captured < max_frames:
                try:
                    # Check if shared file exists
                    if not shared_data_file.exists():
                        no_data_count += 1
                        if no_data_count > 50:  # 5 seconds of no data
                            print("No shared data file found. Is record3d_capture.py running?")
                            print("Start it with: cd ~/code/stretch_ai/scripts && python record3d_capture.py")
                            break
                        time.sleep(0.1)
                        continue
                    
                    # Read frame data with retry on corruption
                    with open(shared_data_file, 'rb') as f:
                        frame_data = pickle.load(f)
                    
                    # Check if this is a new frame
                    current_frame_id = frame_data.get('frame_id', -1)
                    if current_frame_id > last_frame_id:
                        # New frame available
                        self.add_frame(
                            frame_data['rgb'],
                            frame_data['depth'],
                            frame_data['intrinsics'], 
                            frame_data['pose']
                        )
                        frames_captured += 1
                        last_frame_id = current_frame_id
                        no_data_count = 0
                        
                        if frames_captured % 10 == 0:
                            print(f"Captured {frames_captured}/{max_frames} frames")
                    else:
                        # No new frame, wait a bit
                        time.sleep(0.1)
                    
                    # Timeout check
                    if time.time() - start_time > timeout:
                        print(f"Timeout after {timeout}s, captured {frames_captured} frames")
                        break
                        
                except (FileNotFoundError, pickle.PickleError, KeyError, EOFError) as e:
                    no_data_count += 1
                    if no_data_count > 50:
                        print(f"Error reading shared data: {e}")
                        break
                    time.sleep(0.1)
                    continue
            
            # Save the captured data even if we didn't get all requested frames
            if frames_captured > 0:
                print(f"\nSaving {frames_captured} frames to {output_filename}")
                with open(output_filename, 'wb') as f:
                    pickle.dump(self.data, f)
                    
                print(f"Successfully saved pkl file: {output_filename}")
                return str(output_filename)
            else:
                print("❌ No frames captured. Make sure record3d_capture.py is running.")
                return ""
            
        except Exception as e:
            print(f"Error accessing running Record3D process: {e}")
            import traceback
            traceback.print_exc()
            return ""

@click.command()
@click.option("--output-dir", default=".", help="Directory to save pkl files")
@click.option("--max-frames", default=50, help="Maximum number of frames to capture")
@click.option("--width", default=640, help="Target width for resizing images")
@click.option("--height", default=480, help="Target height for resizing images")
def main(output_dir: str, max_frames: int, width: int, height: int):
    """Capture data from a RUNNING record3d_capture.py process and save as pkl file"""
    
    print("Record3D Stream to PKL Capture Tool")
    print("=" * 40)
    print("This tool captures data from an ALREADY RUNNING record3d_capture.py process")
    print()
    print("USAGE:")
    print("1. In terminal 1: cd ~/code/stretch_ai/scripts && python record3d_capture.py")
    print("2. In terminal 2: python stream_to_pkl.py")
    print()
    print(f"Output Directory: {output_dir}")
    print(f"Max Frames: {max_frames}")
    print(f"Target Resolution: {width}x{height}")
    print()
    
    # Create capture client
    capturer = StreamToPklCapture(output_dir=output_dir, target_height=height, target_width=width)
    
    try:
        pkl_file = capturer.capture_from_running_record3d(max_frames=max_frames)
        
        if pkl_file:
            print(f"\n✅ Capture completed successfully!")
            print(f"📁 PKL file saved: {pkl_file}")
            print(f"\nYou can now test this file with:")
            print(f"  python -m stretch.app.read_map -i {pkl_file}")
            print(f"  python -m stretch.app.view_teleop_map -i {pkl_file}")
        else:
            print("❌ No data captured")
            
    except Exception as e:
        print(f"❌ Error during capture: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()