#!/usr/bin/env python3
"""
Improved Record3D ZMQ to Voxel Mapping Consumer

This script follows the proven r3d_to_pkl.py approach for better map generation.
Key improvements:
- Proper coordinate transformations like r3d_to_pkl.py
- Correct data format matching Stretch AI expectations
- Better frame subsampling for quality over quantity
"""

import asyncio
import argparse
import pickle
import time
import zmq
import numpy as np
import torch
from typing import Dict, Any, Optional

from stretch.mapping.voxel import SparseVoxelMap
from stretch.core.interfaces import Observations
from stretch.perception.wrapper import OvmmPerception
from stretch.core.parameters import get_parameters
from stretch.utils.logger import Logger

logger = Logger(__name__)


def get_xyz_coordinates(depth: torch.Tensor, pose: torch.Tensor, intrinsic: torch.Tensor):
    """
    Calculate world and camera coordinates from depth image.
    This follows the exact same approach as r3d_to_pkl.py
    
    Args:
        depth: Depth tensor [1, H, W] 
        pose: Camera pose matrix [4, 4]
        intrinsic: Camera intrinsics [3, 3]
        
    Returns:
        world_coords: World coordinates [H, W, 3]  
        camera_coords: Camera coordinates [H, W, 3]
    """
    _, height, width = depth.shape
    
    # Gets the pixel grid
    xs, ys = torch.meshgrid(
        torch.arange(width, device=depth.device),
        torch.arange(height, device=depth.device), 
        indexing="xy",
    )
    
    x = (xs - intrinsic[0, 2]) / intrinsic[0, 0]
    y = (ys - intrinsic[1, 2]) / intrinsic[1, 1]
    
    # Depth array should be the same shape as x and y
    z = depth[0]
    
    # Prepare camera coordinates (homogeneous)
    camera_coords = torch.stack((x * z, y * z, z, torch.ones_like(z)), axis=-1)
    
    # Transform to world coordinates using the pose matrix
    world_coords = camera_coords @ pose.T
    
    # Return world coordinates (excluding the homogeneous coordinate)
    return world_coords[..., :3], camera_coords[..., :3]


class Record3DZMQVoxelMapper:
    """Improved Record3D ZMQ consumer following r3d_to_pkl.py patterns."""
    
    def __init__(
        self,
        zmq_port: int = 5555,
        voxel_size: float = 0.05,
        grid_resolution: float = 0.02,
        use_perception: bool = True,
        subsample_freq: int = 10,
    ):
        """Initialize improved voxel mapper.
        
        Args:
            zmq_port: Port to connect to Record3D ZMQ publisher
            voxel_size: Size of voxels in meters
            grid_resolution: Grid resolution for navigation
            use_perception: Whether to use perception for instance segmentation
            subsample_freq: Process every Nth frame (like r3d_to_pkl.py)
        """
        self.zmq_port = zmq_port
        self.subsample_freq = subsample_freq
        
        # ZMQ Setup
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"tcp://localhost:{zmq_port}")
        self.socket.setsockopt(zmq.SUBSCRIBE, b"record3d_frames")
        
        # Initialize voxel map
        self.voxel_map = SparseVoxelMap(
            resolution=voxel_size,
            feature_dim=512,
            grid_resolution=grid_resolution,
        )
        
        # Initialize perception
        self.use_perception = use_perception
        self.perception = None
        if use_perception:
            try:
                parameters = get_parameters("default_planner.yaml")
                self.perception = OvmmPerception(parameters)
                logger.info("Initialized perception module")
            except Exception as e:
                logger.warning(f"Failed to initialize perception: {e}")
                self.use_perception = False
        
        # Frame processing stats
        self.frame_count = 0
        self.processed_frames = 0
        self.running = False
        
    async def start_consuming(self, duration: Optional[int] = None):
        """Start consuming frames from ZMQ."""
        logger.info(f"Starting improved ZMQ consumer on port {self.zmq_port}")
        logger.info(f"Subsampling every {self.subsample_freq} frames for quality")
        self.running = True
        
        start_time = time.time()
        
        try:
            while self.running:
                try:
                    if self.socket.poll(timeout=1000):  # 1 second timeout
                        topic, data = self.socket.recv_multipart()
                        
                        # Deserialize frame data
                        frame_packet = pickle.loads(data)
                        self.frame_count += 1
                        
                        # Subsample frames (like r3d_to_pkl.py does)
                        if self.frame_count % self.subsample_freq == 0:
                            await self._process_frame(frame_packet)
                            self.processed_frames += 1
                            
                            if self.processed_frames % 5 == 0:
                                logger.info(f"Processed {self.processed_frames} frames ({self.frame_count} total received)")
                                
                    # Check duration limit
                    if duration is not None and time.time() - start_time > duration:
                        logger.info(f"Duration limit reached ({duration}s)")
                        break
                        
                except zmq.Again:
                    continue
                    
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.running = False
            
    async def _process_frame(self, frame_packet: Dict[str, Any]):
        """Process a single frame following r3d_to_pkl.py approach."""
        try:
            # Extract frame data
            rgb = frame_packet['rgb']  # numpy array
            depth = frame_packet['depth']  # numpy array  
            pose = frame_packet['pose']  # 4x4 pose matrix
            intrinsics = frame_packet['intrinsics']  # 3x3 intrinsics
            frame_id = frame_packet['frame_id']
            
            # Convert to tensors (following r3d_to_pkl.py format)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            rgb_tensor = torch.from_numpy(rgb).float().to(device) / 255.0  # Normalize to [0,1]
            depth_tensor = torch.from_numpy(depth).float().to(device) / 1000.0  # Convert mm to meters
            depth_tensor = depth_tensor.unsqueeze(0)  # Add batch dim [1, H, W]
            pose_tensor = torch.from_numpy(pose).float().to(device)
            intrinsics_tensor = torch.from_numpy(intrinsics).float().to(device)
            
            logger.info(f"[FRAME {frame_id}] Processing RGB {rgb.shape}, Depth {depth.shape}")
            logger.info(f"[DEPTH] Valid pixels: {np.sum(depth > 100)}, Max depth: {np.max(depth)/1000.0:.2f}m")  # depth is in mm
            
            # Calculate world and camera coordinates (like r3d_to_pkl.py)
            world_xyz, camera_xyz = get_xyz_coordinates(depth_tensor, pose_tensor, intrinsics_tensor)
            
            # Format RGB like r3d_to_pkl.py: (image * 255).to(torch.uint8).permute(1, 2, 0)  
            rgb_formatted = (rgb_tensor * 255).to(torch.uint8)
            if rgb_formatted.dim() == 3 and rgb_formatted.shape[0] == 3:
                rgb_formatted = rgb_formatted.permute(1, 2, 0)  # (3,H,W) -> (H,W,3)
                
            # Create observation
            obs = self._create_observation(
                rgb_formatted, depth_tensor, pose, intrinsics, world_xyz, camera_xyz
            )
            
            # Add to voxel map
            self.voxel_map.add_obs(obs)
            
            logger.info(f"[VOXEL] Added frame {frame_id} to voxel map")
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            import traceback
            traceback.print_exc()
            
    def _create_observation(
        self, 
        rgb: torch.Tensor,  # [H, W, 3] uint8
        depth: torch.Tensor,  # [1, H, W] float32 
        pose: np.ndarray,  # [4, 4]
        intrinsics: np.ndarray,  # [3, 3]
        world_xyz: torch.Tensor,  # [H, W, 3]
        camera_xyz: torch.Tensor,  # [H, W, 3]
    ) -> Observations:
        """Create Observations object with proper coordinate transforms."""
        
        # Convert to numpy
        rgb_array = rgb.cpu().numpy()
        depth_array = depth[0].cpu().numpy()  # Remove batch dimension
        xyz_array = camera_xyz.cpu().numpy()
        
        # Extract GPS and compass from pose
        gps = pose[:2, 3].astype(np.float32)
        yaw = np.arctan2(pose[1, 0], pose[0, 0])
        compass = np.array([yaw], dtype=np.float32)
        
        # Create observation
        obs = Observations(
            gps=gps,
            compass=compass,
            rgb=rgb_array,
            depth=depth_array,
            xyz=xyz_array,  # Use calculated camera coordinates
            camera_K=intrinsics.astype(np.float32),
            camera_pose=pose.astype(np.float32),
            instance=None,
            task_observations=None
        )
        
        # Add perception if available
        if self.use_perception and self.perception is not None:
            try:
                rgb_tensor = torch.from_numpy(rgb_array).float() / 255.0
                depth_tensor = torch.from_numpy(depth_array).float()
                base_pose = torch.tensor([gps[0], gps[1], compass[0]], dtype=torch.float32)
                
                semantic, instance, task_obs = self.perception.predict_segmentation(
                    rgb_tensor, depth_tensor, base_pose
                )
                
                if instance is not None:
                    # Handle both tensor and numpy array cases
                    if hasattr(instance, 'cpu'):
                        obs.instance = instance.cpu().numpy()
                    else:
                        obs.instance = instance
                if task_obs is not None:
                    obs.task_observations = task_obs
                    
                logger.debug(f"[PERCEPTION] Added semantic data")
                    
            except Exception as e:
                logger.warning(f"Perception failed: {e}")
                
        return obs
        
    def save_map(self, filename: str):
        """Save voxel map to pickle file."""
        logger.info(f"Saving map to {filename}")
        self.voxel_map.write_to_pickle(filename)
        
        # Print usage instructions
        print(f"\n🗺️  Map saved successfully!")
        print(f"📁 File: {filename}")
        print(f"📊 Processed frames: {self.processed_frames} (of {self.frame_count} received)")
        print(f"🎯 Subsample rate: 1/{self.subsample_freq}")
        print(f"\n🚀 You can now use this map for VLM planning:")
        print(f"   python3 -m stretch.app.vlm_planning -i {filename} --show-instances \\")
        print(f"           -c app/vlm_planning/gpt4v_planner.yaml --show-svm -f {self.processed_frames} -fs 3")
        print(f"\n📋 Or view the map with:")
        print(f"   python -m stretch.app.read_map -i {filename} --show-svm")
        print()
        
    def get_map_info(self) -> Dict[str, Any]:
        """Get map statistics."""
        try:
            obstacles, explored = self.voxel_map.get_2d_map()
            return {
                "processed_frames": self.processed_frames,
                "total_frames": self.frame_count,
                "subsample_rate": f"1/{self.subsample_freq}",
                "voxel_count": len(self.voxel_map.observations),
                "explored_area": torch.sum(explored).item() if explored is not None else 0,
                "obstacle_area": torch.sum(obstacles).item() if obstacles is not None else 0,
            }
        except Exception as e:
            return {"error": str(e)}
            
    def cleanup(self):
        """Cleanup ZMQ resources."""
        self.socket.close()
        self.context.term()


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Record3D ZMQ to Voxel Mapping")
    parser.add_argument("--zmq-port", type=int, default=5555, help="ZMQ port to connect to")
    parser.add_argument("--duration", type=int, default=60, help="Mapping duration in seconds")  
    parser.add_argument("--save-map", type=str, help="Save map to pickle file")
    parser.add_argument("--no-perception", action="store_true", help="Disable perception")
    parser.add_argument("--subsample", type=int, default=10, help="Process every Nth frame (default: 10)")
    args = parser.parse_args()
    
    # Create improved mapper
    mapper = Record3DZMQVoxelMapper(
        zmq_port=args.zmq_port,
        use_perception=not args.no_perception,
        subsample_freq=args.subsample
    )
    
    try:
        print(f"Connecting to ZMQ publisher on port {args.zmq_port}")
        print(f"Processing every {args.subsample} frames for better quality maps")
        print("Make sure your Record3D ZMQ client is running!")
        
        # Start consuming
        await mapper.start_consuming(duration=args.duration)
        
        # Print final stats
        info = mapper.get_map_info()
        print(f"Final map info: {info}")
        
        # Save map if requested
        if args.save_map:
            mapper.save_map(args.save_map)
            
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        mapper.cleanup()


if __name__ == "__main__":
    asyncio.run(main())