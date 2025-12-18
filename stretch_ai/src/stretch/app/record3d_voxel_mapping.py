#!/usr/bin/env python3
"""
Record3D to Voxel Mapping Integration

This script integrates Record3D WiFi streaming with Stretch AI's voxel mapping system
for real-time 3D environmental mapping and navigation.
"""

import asyncio
import time
from typing import Optional, Tuple, Dict, Any
import numpy as np
import torch
from dataclasses import dataclass

from stretch.app.record3d_client import Record3DClient
from stretch.mapping.voxel import SparseVoxelMap
from stretch.mapping.voxel.voxel import Frame
from stretch.mapping.voxel.voxel_map import SparseVoxelMapNavigationSpace
from stretch.core.interfaces import Observations
from stretch.motion import RobotModel
from stretch.utils.point_cloud_torch import unproject_masked_depth_to_xyz_coordinates
from stretch.utils.logger import Logger
from stretch.perception.wrapper import OvmmPerception
from stretch.core.parameters import get_parameters

logger = Logger(__name__)


@dataclass
class Record3DFrame:
    """Represents a frame from Record3D with pose information."""
    rgb: np.ndarray
    depth: np.ndarray
    camera_intrinsics: np.ndarray
    timestamp: float
    camera_pose: Optional[np.ndarray] = None  # 4x4 transformation matrix


class Record3DVoxelMapper:
    """Integrates Record3D streaming with Stretch AI voxel mapping."""
    
    def __init__(
        self,
        device_ip: str,
        voxel_size: float = 0.05,
        grid_resolution: float = 0.02,
        feature_dim: int = 512,
        robot_model: Optional[RobotModel] = None,
        port: int = 80,
        use_perception: bool = True,
    ):
        """Initialize Record3D voxel mapper.
        
        Args:
            device_ip: IP address of iOS device running Record3D
            voxel_size: Size of voxels in meters
            grid_resolution: Grid resolution for navigation
            feature_dim: Feature dimension for voxel map
            robot_model: Robot model for navigation planning
            port: Record3D port number
        """
        self.device_ip = device_ip
        self.port = port
        
        # Initialize Record3D client
        self.record3d_client = Record3DClient(device_ip, port)
        self.record3d_client.set_frame_callback(self._on_new_frame)
        
        # Initialize voxel map
        self.voxel_map = SparseVoxelMap(
            resolution=voxel_size,
            feature_dim=feature_dim,
            grid_resolution=grid_resolution,
        )
        
        # Navigation space for path planning
        self.robot_model = robot_model
        if robot_model is not None:
            self.navigation_space = SparseVoxelMapNavigationSpace(
                voxel_map=self.voxel_map,
                robot=robot_model,
                step_size=0.1,
                rotation_step_size=0.5,
            )
        else:
            self.navigation_space = None
            
        # Frame processing
        self.frame_queue = asyncio.Queue()
        self.processing_frames = False
        self.frame_count = 0
        
        # Default camera pose (can be updated with SLAM/tracking)
        self.current_camera_pose = np.eye(4)
        self.current_camera_pose[2, 3] = 1.5  # 1.5m height
        
        # Initialize perception for instance segmentation
        self.use_perception = use_perception
        self.perception = None
        if use_perception:
            try:
                # Load default parameters for perception
                parameters = get_parameters("default_planner.yaml")
                self.perception = OvmmPerception(parameters)
                logger.info("Initialized perception module for instance segmentation")
            except Exception as e:
                logger.warning(f"Failed to initialize perception module: {e}")
                self.use_perception = False
        
        
    async def start_streaming(self) -> bool:
        """Start Record3D streaming and voxel mapping."""
        logger.info(f"Starting Record3D streaming from {self.device_ip}:{self.port}")
        
        # Start frame processing first
        self.processing_frames = True
        asyncio.create_task(self._process_frame_queue())
        
        # Then connect to Record3D device
        if not await self.record3d_client.connect():
            logger.error("Failed to connect to Record3D device")
            self.processing_frames = False
            return False
        
        logger.info("Record3D voxel mapping started successfully")
        return True
        
    async def stop_streaming(self):
        """Stop streaming and processing."""
        logger.info("Stopping Record3D streaming")
        self.processing_frames = False
        await self.record3d_client.disconnect()
        
    def _on_new_frame(self, rgb: np.ndarray, depth: np.ndarray, intrinsics: np.ndarray):
        """Callback for new frames from Record3D."""
        logger.info(f"[FRAME] Received RGB {rgb.shape}, Depth {depth.shape}, valid_depth: {np.sum(depth > 0.1)}")
        
        if intrinsics is not None:
            # Update voxel map with camera intrinsics on first frame
            self._update_camera_intrinsics(intrinsics)
            
        # Create frame and add to queue
        frame = Record3DFrame(
            rgb=rgb.copy(),
            depth=depth.copy(),
            camera_intrinsics=intrinsics.copy() if intrinsics is not None else None,
            timestamp=time.time(),
            camera_pose=self.current_camera_pose.copy(),
        )
        
        # Add to queue (non-blocking)
        try:
            self.frame_queue.put_nowait(frame)
            logger.info(f"[QUEUE] Frame added, queue size: {self.frame_queue.qsize()}")
        except asyncio.QueueFull:
            logger.warning("Frame queue full, dropping frame")
            
    def _update_camera_intrinsics(self, intrinsics: np.ndarray):
        """Update voxel map with camera intrinsics."""
        logger.info(f"Updating camera intrinsics: {intrinsics}")
        # The voxel map expects the intrinsics in its camera_K parameter
        # This might need adjustment based on your voxel map implementation
        
    async def _process_frame_queue(self):
        """Process frames from the queue and update voxel map."""
        logger.info("Started frame processing")
        
        while self.processing_frames:
            try:
                # Get frame from queue with timeout
                frame = await asyncio.wait_for(self.frame_queue.get(), timeout=1.0)
                await self._process_frame(frame)
                self.frame_count += 1
                
                if self.frame_count % 10 == 0:
                    logger.info(f"Processed {self.frame_count} frames")
                    
            except asyncio.TimeoutError:
                # No frame available, continue
                continue
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
                
        logger.info("Frame processing stopped")
        
    async def _process_frame(self, frame: Record3DFrame):
        """Process a single frame and update voxel map."""
        try:
            # Convert numpy arrays to tensors
            rgb_tensor = torch.from_numpy(frame.rgb).permute(2, 0, 1).float() / 255.0
            depth_tensor = torch.from_numpy(frame.depth).float()
            
            # Get valid depth mask (depth > 0 and < 3.0 meters)
            valid_depth_mask = (depth_tensor > 0.01) & (depth_tensor < 3.0)
            
            if not torch.any(valid_depth_mask):
                logger.warning("No valid depth points in frame")
                return
                
            # Unproject depth to 3D coordinates
            if frame.camera_intrinsics is not None:
                camera_K = torch.from_numpy(frame.camera_intrinsics).float()
                
                # Unproject to camera coordinates
                xyz_camera = unproject_masked_depth_to_xyz_coordinates(
                    depth_tensor.unsqueeze(0),
                    camera_K.unsqueeze(0),
                    valid_depth_mask.unsqueeze(0),
                )
                
                # Transform to world coordinates
                camera_pose_tensor = torch.from_numpy(frame.camera_pose).float()
                xyz_world = self._transform_points(xyz_camera[0], camera_pose_tensor)
                
                # Extract RGB for valid points
                rgb_valid = rgb_tensor[:, valid_depth_mask]
                
                # Create observation-like structure for voxel map
                obs = self._create_observation(
                    xyz_world, rgb_valid, frame.camera_pose, frame.camera_intrinsics, depth_tensor
                )
                
                # Add to voxel map using correct method
                self.voxel_map.add_obs(obs)
                
                
            else:
                logger.warning("No camera intrinsics available for frame")
                
        except Exception as e:
            logger.error(f"Error in frame processing: {e}")
            
    def _transform_points(self, points_camera: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        """Transform points from camera to world coordinates."""
        # Add homogeneous coordinate
        ones = torch.ones(points_camera.shape[0], 1)
        points_homogeneous = torch.cat([points_camera, ones], dim=1)
        
        # Apply transformation
        points_world = (pose @ points_homogeneous.T).T
        
        return points_world[:, :3]
        
    def _create_observation(
        self, xyz_world: torch.Tensor, rgb: torch.Tensor, camera_pose: np.ndarray, intrinsics: np.ndarray, depth: torch.Tensor
    ) -> Observations:
        """Create an Observations object for the voxel map.
        
        This provides the minimal required data for voxel mapping:
        - rgb, depth, camera_pose, camera_K (essential)
        - gps, compass (derived from camera_pose for base_pose calculation)
        - xyz (optional - can be computed from depth+intrinsics if None)
        - task_observations (optional - for semantic information)
        """
        
        # Convert RGB from (3, H, W) to (H, W, 3) format
        rgb_image = rgb.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        depth_array = depth.cpu().numpy()
        
        # Extract GPS (x,y) and compass (theta) from camera pose for base_pose calculation
        # This is used by add_obs: base_pose = [gps[0], gps[1], compass[0]]
        gps = camera_pose[:2, 3].astype(np.float32)  # x, y translation
        yaw = np.arctan2(camera_pose[1, 0], camera_pose[0, 0])  # rotation around z-axis
        compass = np.array([yaw], dtype=np.float32)
        
        # Create observation with essential data
        obs = Observations(
            gps=gps,
            compass=compass,
            rgb=rgb_image,
            depth=depth_array,
            camera_K=intrinsics.astype(np.float32),
            camera_pose=camera_pose.astype(np.float32),
            xyz=None,  # Let voxel map compute from depth+intrinsics
            instance=None,
            task_observations=None
        )
        
        # Add semantic information if perception is available
        if self.use_perception and self.perception is not None:
            try:
                # Prepare inputs for perception
                rgb_tensor = torch.from_numpy(rgb_image).float() / 255.0  # [0,1] range
                depth_tensor = torch.from_numpy(depth_array).float()
                base_pose_tensor = torch.tensor([gps[0], gps[1], compass[0]], dtype=torch.float32)
                
                # Run perception
                semantic, instance, task_obs = self.perception.predict_segmentation(
                    rgb_tensor, depth_tensor, base_pose_tensor
                )
                
                # Update observation with semantic results
                if instance is not None:
                    obs.instance = instance.cpu().numpy()
                if task_obs is not None:
                    obs.task_observations = task_obs
                    
                logger.debug(f"[PERCEPTION] Added semantic data: instance={instance is not None}, task_obs={task_obs is not None}")
                
            except Exception as e:
                logger.warning(f"Perception prediction failed: {e}")
                
        return obs
        
    def update_camera_pose(self, pose: np.ndarray):
        """Update the current camera pose (4x4 transformation matrix)."""
        self.current_camera_pose = pose.copy()
        
    def get_navigation_goals(self, current_pos: np.ndarray, radius: float = 1.0) -> list:
        """Get navigation goals from frontier exploration."""
        if self.navigation_space is None:
            logger.warning("No navigation space available (robot model not provided)")
            return []
            
        try:
            goals = []
            for goal in self.navigation_space.sample_closest_frontier(current_pos, max_tries=10):
                if goal is not None:
                    goals.append(goal.cpu().numpy())
                else:
                    break
            return goals
        except Exception as e:
            logger.error(f"Error getting navigation goals: {e}")
            return []
            
    def start_visualization(self):
        """Start visualization (rerun removed - using navigation space visualization)."""
        logger.info("Visualization available through navigation space")
        
    def visualize_map(self, show_instances: bool = False):
        """Visualize the current voxel map."""
        if self.navigation_space is not None and hasattr(self.navigation_space, 'show'):
            self.navigation_space.show(instances=show_instances)
        else:
            logger.info("Navigation space visualization not available")
            
    def get_map_info(self) -> Dict[str, Any]:
        """Get information about the current map."""
        try:
            obstacles, explored = self.voxel_map.get_2d_map()
            return {
                "frame_count": self.frame_count,
                "voxel_count": len(self.voxel_map.observations) if hasattr(self.voxel_map, 'observations') else 0,
                "explored_area": torch.sum(explored).item() if explored is not None else 0,
                "obstacle_area": torch.sum(obstacles).item() if obstacles is not None else 0,
                "is_streaming": self.record3d_client.is_streaming,
            }
        except Exception as e:
            logger.error(f"Error getting map info: {e}")
            return {"error": str(e)}
    
    def save_map(self, filename: str, compress: bool = True):
        """Save the voxel map to a pickle file."""
        logger.info(f"Saving map to {filename}")
        self.voxel_map.write_to_pickle(filename, compress=compress)
        
        # Print usage instructions
        print(f"\n🗺️  Map saved successfully!")
        print(f"📁 File: {filename}")
        print(f"📊 Frames: {self.frame_count}")
        print(f"\n🚀 You can now use this map for VLM planning:")
        print(f"   python3 -m stretch.app.vlm_planning -i {filename} --show-instances \\")
        print(f"           -c app/vlm_planning/gpt4v_planner.yaml --show-svm -f {self.frame_count} -fs 3")
        print(f"\n📋 Or view the map with:")
        print(f"   python -m stretch.app.read_map -i {filename} --show-svm")
        print()


async def main():
    """Demo usage of Record3D voxel mapping."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Record3D to Voxel Mapping")
    parser.add_argument("device_ip", help="IP address of iOS device running Record3D")
    parser.add_argument("--port", type=int, default=80, help="Port number (default: 80)")
    parser.add_argument("--voxel-size", type=float, default=0.05, help="Voxel size in meters")
    parser.add_argument("--visualize", action="store_true", help="Show map visualization")
    parser.add_argument("--duration", type=int, default=60, help="Mapping duration in seconds")
    parser.add_argument("--save-map", type=str, help="Save map to pickle file")
    parser.add_argument("--no-perception", action="store_true", help="Disable perception module")
    args = parser.parse_args()
    
    # Create mapper
    mapper = Record3DVoxelMapper(
        device_ip=args.device_ip,
        port=args.port,
        voxel_size=args.voxel_size,
        use_perception=not args.no_perception,
    )
    
    try:
        print(f"Starting Record3D voxel mapping on {args.device_ip}:{args.port}")
        
        # Start visualization if requested
        if args.visualize:
            mapper.start_visualization()
        
        if await mapper.start_streaming():
            print(f"Mapping for {args.duration} seconds... Move your device to scan the environment")
            
            # Map for specified duration
            start_time = time.time()
            while time.time() - start_time < args.duration:
                await asyncio.sleep(1)
                
                # Print map info every 5 seconds
                if int(time.time() - start_time) % 5 == 0:
                    info = mapper.get_map_info()
                    print(f"Map info: {info}")
                    
            print("Mapping completed!")
            
            # Show final map info
            final_info = mapper.get_map_info()
            print(f"Final map info: {final_info}")
            
            # Visualize if requested
            if args.visualize:
                print("Showing map visualization...")
                mapper.visualize_map()
                
            # Save map if requested
            if args.save_map:
                print(f"Saving map to {args.save_map}")
                mapper.save_map(args.save_map)
                
        else:
            print("Failed to start streaming")
            
    except KeyboardInterrupt:
        print("\nStopping mapping...")
    finally:
        await mapper.stop_streaming()


if __name__ == "__main__":
    asyncio.run(main())