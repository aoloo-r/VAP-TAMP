#!/usr/bin/env python3
# direct_nvblox_import.py - Import perfect nvblox map directly into Stretch AI

import time
import datetime
import click
import numpy as np
from typing import Optional
import pickle

from stretch.agent.robot_agent import RobotAgent
from stretch.agent.ros2_robot_client import ROS2RobotClient  
from stretch.mapping.voxel import SparseVoxelMap
from stretch.utils.logger import Logger
from stretch.perception.wrapper import create_semantic_sensor
from stretch.core.parameters import Parameters, get_parameters
from stretch.perception.encoders import get_encoder

# ROS2 imports for direct nvblox data access
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import PointCloud2
    from nvblox_msgs.msg import Mesh  # nvblox mesh message
    import sensor_msgs_py.point_cloud2 as pc2
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False

logger = Logger(__name__)


class DirectNvbloxImporter:
    """Import nvblox map directly as the base geometry for Stretch AI."""
    
    def __init__(self, config_file: str, enable_rerun: bool = True):
        self.config_file = config_file
        self.enable_rerun = enable_rerun
        self.robot = None
        self.agent = None
        self.base_map = None  # This will store the imported nvblox map
        self.nvblox_geometry = None
        
    def setup(self) -> bool:
        """Setup the direct import system."""
        print("\n" + "="*70)
        print("    DIRECT NVBLOX MAP IMPORT FOR STRETCH AI")
        print("    Import perfect nvblox geometry as base map")
        print("="*70)
        
        # Initialize ROS2
        if not HAS_ROS2:
            print("Error: ROS2 not available")
            return False
            
        if not rclpy.ok():
            rclpy.init()
        
        # Load parameters
        print(f"Loading configuration: {self.config_file}")
        try:
            params = get_parameters(self.config_file)
        except Exception as e:
            print(f"Error loading config: {e}")
            return False
        
        # Create ROS2 client
        print("Creating ROS2 client for nvblox data access...")
        try:
            self.robot = ROS2RobotClient(
                parameters=params,
                enable_rerun_server=self.enable_rerun
            )
        except Exception as e:
            print(f"Failed to create ROS2 client: {e}")
            return False
        
        # Wait for system to be ready
        print("Waiting for nvblox and camera data...")
        if not self.wait_for_data_ready():
            return False
        
        print("✓ System ready for nvblox import")
        return True
    
    def wait_for_data_ready(self, timeout: float = 30.0) -> bool:
        """Wait for nvblox and camera data to be available."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check if we have nvblox data
            if self.robot.has_nvblox_data():
                print("✓ nvblox data available")
                return True
                
            print(".", end="", flush=True)  
            time.sleep(0.5)
        
        print(f"\nTimeout: nvblox data not available after {timeout}s")
        return False
    
    def import_nvblox_base_map(self) -> bool:
        """Import the current nvblox map as the base geometry."""
        print("\nImporting nvblox geometry as base map...")
        
        # Get current nvblox pointcloud
        nvblox_pointcloud = self.robot.get_nvblox_pointcloud()
        if nvblox_pointcloud is None:
            print("Error: No nvblox pointcloud available")
            return False
        
        # Convert nvblox pointcloud to numpy
        self.nvblox_geometry = self.convert_nvblox_to_numpy(nvblox_pointcloud)
        if self.nvblox_geometry is None:
            print("Error: Failed to convert nvblox pointcloud")
            return False
            
        print(f"✓ Imported nvblox geometry: {len(self.nvblox_geometry)} points")
        
        # Create base voxel map from nvblox geometry
        return self.create_base_voxel_map()
    
    def convert_nvblox_to_numpy(self, pointcloud: PointCloud2) -> Optional[np.ndarray]:
        """Convert ROS2 PointCloud2 to numpy array."""
        try:
            # Extract points from pointcloud
            points = []
            for point in pc2.read_points(pointcloud, skip_nans=True):
                points.append([point[0], point[1], point[2]])
            
            if len(points) == 0:
                return None
                
            return np.array(points, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Failed to convert nvblox pointcloud: {e}")
            return None
    
    def create_base_voxel_map(self) -> bool:
        """Create a voxel map from nvblox geometry."""
        print("Creating base voxel map from nvblox geometry...")
        
        try:
            # Load parameters
            params = get_parameters(self.config_file)
            
            # Create encoder
            encoder = get_encoder(params["encoder"], params.get("encoder_args", {}))
            
            # Create voxel map
            self.base_map = SparseVoxelMap.from_parameters(
                params,
                encoder,
                voxel_size=params["voxel_size"],
                use_instance_memory=False  # Start with geometry only
            )
            
            # Add nvblox geometry to voxel map
            # Create a fake observation to insert the geometry
            fake_obs = self.create_observation_from_pointcloud(self.nvblox_geometry)
            
            if fake_obs:
                self.base_map.add_obs(fake_obs)
                print(f"✓ Base voxel map created with {len(self.nvblox_geometry)} nvblox points")
                return True
            else:
                print("Error: Could not create observation from nvblox geometry")
                return False
                
        except Exception as e:
            logger.error(f"Failed to create base voxel map: {e}")
            return False
    
    def create_observation_from_pointcloud(self, points: np.ndarray):
        """Create a fake observation containing the nvblox pointcloud."""
        from stretch.core.interfaces import Observations
        import torch
        
        try:
            # Create a fake RGB image (we only care about geometry)
            fake_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Create fake depth from pointcloud Z values
            fake_depth = np.zeros((480, 640), dtype=np.float32)
            
            # Create fake camera pose (identity - nvblox coordinates are already in world frame)
            fake_pose = np.eye(4)
            
            # Convert points to tensor
            points_tensor = torch.from_numpy(points).float()
            
            # Create observation object
            obs = Observations(
                rgb=torch.from_numpy(fake_rgb).permute(2, 0, 1),  # CHW format
                depth=torch.from_numpy(fake_depth),
                xyz=points_tensor,
                camera_pose=torch.from_numpy(fake_pose).float(),
                # Add other required fields as needed
            )
            
            return obs
            
        except Exception as e:
            logger.error(f"Failed to create observation from pointcloud: {e}")
            return None
    
    def run_semantic_overlay_mapping(self):
        """Run mapping that adds semantic understanding on top of nvblox base."""
        print("\n" + "="*70)
        print("SEMANTIC OVERLAY ON NVBLOX BASE MAP")
        print("Your perfect nvblox geometry is preserved.")
        print("Drive around to add semantic understanding (object detection).")
        print("Press Ctrl+C when done to save the enhanced map.")
        print("="*70 + "\n")
        
        # Create semantic sensor
        params = get_parameters(self.config_file)
        semantic_sensor = create_semantic_sensor(params)
        
        # Create robot agent that uses our base map
        self.agent = EnhancedNvbloxAgent(
            robot=self.robot,
            parameters=params,
            voxel_map=self.base_map,  # Start with our imported nvblox base
            semantic_sensor=semantic_sensor,
            use_instance_memory=True,
            enable_realtime_updates=True,
            teleop_only=True
        )
        
        last_save_time = time.time()
        start_time = time.time()
        save_interval = 60  # Save every minute
        
        try:
            while True:
                current_time = time.time()
                
                # Status updates
                if current_time % 15 < 0.5:  # Every 15 seconds
                    elapsed = current_time - start_time
                    obs_count = getattr(self.agent, 'obs_count', 0)
                    instances = self.agent.get_instances() if hasattr(self.agent, 'get_instances') else []
                    
                    print(f"[Status] Time: {elapsed:.1f}s | Observations: {obs_count} | Objects: {len(instances)}")
                    
                # Periodic saves
                if current_time - last_save_time > save_interval:
                    self.save_enhanced_map(is_checkpoint=True)
                    last_save_time = current_time
                
                time.sleep(1.0)
                
        except KeyboardInterrupt:
            print("\n" + "="*70)
            print("SAVING ENHANCED MAP (nvblox geometry + semantic understanding)")
            print("="*70)
            
        finally:
            self.save_enhanced_map(is_checkpoint=False)
            self.show_final_map()
            
            if self.robot:
                self.robot.stop()
                
            print("Enhanced nvblox mapping complete!")
    
    def save_enhanced_map(self, is_checkpoint: bool):
        """Save the enhanced map (nvblox + semantic)."""
        prefix = "nvblox_enhanced_checkpoint" if is_checkpoint else "nvblox_enhanced_final"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.pkl"
        
        if is_checkpoint:
            print(f"[Checkpoint] Saving enhanced nvblox map...")
        else:
            print(f"Saving final enhanced nvblox map...")
        
        # Save the enhanced map
        if self.agent:
            enhanced_map = self.agent.get_voxel_map()
            enhanced_map.write_to_pickle(filename)
            
            # Also save metadata
            metadata = {
                'nvblox_base_points': len(self.nvblox_geometry) if self.nvblox_geometry is not None else 0,
                'enhanced_with_semantics': True,
                'creation_method': 'direct_nvblox_import',
                'timestamp': timestamp
            }
            
            metadata_filename = f"{prefix}_{timestamp}_metadata.json"
            import json
            with open(metadata_filename, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Enhanced map saved: {filename}")
            print(f"Metadata saved: {metadata_filename}")
    
    def show_final_map(self):
        """Display the final enhanced map."""
        print("\n" + "="*70)
        print("DISPLAYING ENHANCED NVBLOX MAP")
        print("="*70)
        
        try:
            if self.agent and hasattr(self.agent, 'show_voxel_map'):
                self.agent.show_voxel_map()
            elif self.base_map:
                self.base_map.show(instances=True)
            print("Enhanced map visualization opened.")
        except Exception as e:
            print(f"Could not display map: {e}")


class EnhancedNvbloxAgent(RobotAgent):
    """Robot agent that preserves nvblox base geometry and adds semantics."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info("Enhanced nvblox agent: preserving base geometry, adding semantics")
    
    def update_map_with_observation(self, obs):
        """Add semantic information while preserving nvblox base geometry."""
        # Only add semantic information, don't modify the base geometry
        self.get_voxel_map().add_obs(obs, process_geometry=False, semantic_only=True)


@click.command()
@click.option("--config", default="fixed_nvblox_config.yaml", help="Configuration file")
@click.option("--rerun/--no-rerun", default=True, help="Enable Rerun visualization") 
def main(config, rerun):
    """Import perfect nvblox map directly into Stretch AI and add semantic understanding."""
    
    print("DIRECT NVBLOX MAP IMPORT")
    print("This will:")
    print("  1. Import your perfect nvblox geometry as the base map")
    print("  2. Preserve the high-quality nvblox reconstruction")
    print("  3. Add semantic understanding (object detection) on top")
    print("  4. Give you the best of both worlds!")
    
    importer = DirectNvbloxImporter(config, enable_rerun=rerun)
    
    if not importer.setup():
        print("Failed to setup direct nvblox import")
        return
    
    # Step 1: Import the current nvblox map
    print("\n" + "="*50)
    print("STEP 1: Import nvblox base map")
    print("="*50)
    
    if not importer.import_nvblox_base_map():
        print("Failed to import nvblox base map")
        return
    
    # Step 2: Run semantic overlay mapping
    print("\n" + "="*50) 
    print("STEP 2: Add semantic understanding")
    print("="*50)
    
    importer.run_semantic_overlay_mapping()


if __name__ == "__main__":
    main()