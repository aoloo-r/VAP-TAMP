#!/usr/bin/env python3
# ros2_nvblox_hybrid_mapping.py - High-quality mapping using nvblox + Stretch AI hybrid approach

import time
import datetime
import click
import numpy as np

from stretch.agent.robot_agent import RobotAgent
from stretch.agent.ros2_robot_client import ROS2RobotClient
from stretch.mapping.voxel.nvblox_voxel import create_nvblox_voxel_map
from stretch.utils.logger import Logger
from stretch.perception.wrapper import create_semantic_sensor
from stretch.core.parameters import Parameters, get_parameters
from stretch.perception.encoders import get_encoder

logger = Logger(__name__)


class ROS2NvbloxHybridMapper:
    """
    Hybrid mapper that combines:
    - nvblox's superior 3D reconstruction
    - Stretch AI's semantic understanding and navigation planning
    """
    
    def __init__(self, save_interval: int, use_semantic: bool, config_file: str, enable_rerun: bool = False):
        self.save_interval = save_interval
        self.use_semantic = use_semantic
        self.config_file = config_file
        self.enable_rerun = enable_rerun
        self.robot = None
        self.agent = None
        self.voxel_map = None

    def setup(self) -> bool:
        """Initialize hybrid nvblox + Stretch AI mapping system."""
        print("\n" + "="*70)
        print("    INITIALIZING HYBRID NVBLOX + STRETCH AI MAPPING")
        print("    High-Quality 3D Reconstruction + Semantic Understanding")
        print("="*70)
        
        # 1. Load parameters
        print(f"Loading configuration from: {self.config_file}")
        try:
            params = get_parameters(self.config_file)
        except Exception as e:
            print(f"Warning: Error loading config file: {e}")
            print("Using default parameters...")
            params = Parameters()
            self._set_default_parameters(params)
        
        # 2. Configure ROS2 topic mapping
        self._configure_ros2_topics(params)
        
        # 3. Create ROS2 robot client
        print("Creating ROS2 robot client...")
        try:
            self.robot = ROS2RobotClient(parameters=params, enable_rerun_server=self.enable_rerun)
            if self.enable_rerun:
                print("✓ ROS2 robot client created with Rerun visualization")
            else:
                print("✓ ROS2 robot client created")
        except Exception as e:
            logger.error(f"Failed to create ROS2 robot client: {e}")
            return False
        
        # 4. Wait for robot data
        print("Waiting for robot data from ROS2 topics...")
        timeout = 30.0
        start_time = time.time()
        
        while not self.robot.is_running() or not self.robot.node.has_complete_observation():
            if time.time() - start_time > timeout:
                logger.error("Timeout waiting for robot data from ROS2 topics")
                return False
            time.sleep(0.5)
            print(".", end="", flush=True)
        
        print("\n✓ Robot data received from ROS2 topics")
        
        # 5. Create encoder
        print("Creating encoder...")
        encoder = get_encoder(params["encoder"], params.get("encoder_args", {}))
        print("✓ Encoder created")
        
        # 6. Create semantic sensor if requested
        semantic_sensor = None
        if self.use_semantic:
            print("Creating semantic sensor...")
            semantic_sensor = create_semantic_sensor(params)
            print("✓ Semantic sensor created")
        
        # 7. Create HYBRID nvblox + Stretch AI voxel map
        print("Creating hybrid nvblox + Stretch AI voxel map...")
        # Pass the robot's ROS2 node AND executor to share the same ROS2 context
        robot_node = getattr(self.robot, 'node', None)
        robot_executor = getattr(self.robot, 'executor', None)
        print(f"[DEBUG] Robot node for voxel map: {robot_node is not None}")
        print(f"[DEBUG] Robot executor for voxel map: {robot_executor is not None}")
        print(f"[DEBUG] Robot node type: {type(robot_node) if robot_node else 'None'}")
        print(f"[DEBUG] Robot executor type: {type(robot_executor) if robot_executor else 'None'}")
        
        self.voxel_map = create_nvblox_voxel_map(
            params,
            encoder,
            voxel_size=params["voxel_size"],
            use_instance_memory=self.use_semantic,
            ros_node=robot_node,
            executor=robot_executor,
        )
        print("✓ Hybrid nvblox voxel map created")
        print(f"[DEBUG] Voxel map ROS node after creation: {getattr(self.voxel_map, '_ros_node', None) is not None}")
        
        # 8. Create robot agent (let voxel map handle nvblox exclusively)
        print("Creating hybrid mapping agent...")
        # Disable agent's nvblox subscription to avoid conflicts with voxel map's subscription
        self.agent = HybridNvbloxAgent(
            robot=self.robot,
            parameters=params,
            voxel_map=self.voxel_map,
            semantic_sensor=semantic_sensor,
            use_instance_memory=self.use_semantic,
            enable_realtime_updates=True,
            teleop_only=True,
            use_nvblox=False,  # Let voxel map handle nvblox exclusively
        )
        
        print("✓ Hybrid nvblox + Stretch AI agent created")
        print("✓ System initialization complete")
        print("\nThis system will use:")
        print("  • nvblox for high-quality 3D reconstruction")  
        print("  • Stretch AI for semantic understanding")
        print("  • Combined system for navigation planning")
        return True
    
    def _set_default_parameters(self, params: Parameters):
        """Set parameters optimized for nvblox integration."""
        params.data = {
            "motion": {
                "moving_threshold": 0.001,
                "angle_threshold": 0.01,
                "min_steps_not_moving": 5,
            },
            "agent": {
                "use_realtime_updates": True,
                "sweep_head_on_update": False,
                "in_place_rotation_steps": 0,
            },
            "voxel_size": 0.02,
            "local_radius": 2.0,
            "obs_min_height": 0.05,
            "obs_max_height": 2.5,
            "obs_min_density": 5,
            "min_points_per_voxel": 3,
            "min_depth": 0.1,
            "max_depth": 4.0,
            "add_local_every_step": False,
            "pad_obstacles": 0.1,
            "encoder": "siglip",
            "encoder_args": {},
        }
    
    def _configure_ros2_topics(self, params: Parameters):
        """Configure ROS2 topics for hybrid operation."""
        ros2_config = {
            "rgb_topic": "/camera/color/image_raw",
            "depth_topic": "/camera/depth/image_rect_raw", 
            "camera_info_topic": "/camera/color/camera_info",
            "joint_state_topic": "/joint_states",
            "odom_topic": "/odom",
            "nvblox_pointcloud_topic": "/nvblox_node/static_esdf_pointcloud",
            "nvblox_mesh_topic": "/nvblox_node/mesh",
            "cmd_vel_topic": "/cmd_vel",
        }
        
        params["ros2"] = ros2_config
        print(f"✓ Configured hybrid ROS2 topics:")
        for key, topic in ros2_config.items():
            print(f"    {key}: {topic}")
    
    def run_mapping_loop(self):
        """Main hybrid mapping loop."""
        print("\n" + "="*70)
        print("HYBRID NVBLOX + STRETCH AI MAPPING STARTED!")
        print("This system combines:")
        print("  • nvblox's high-quality 3D reconstruction")
        print("  • Stretch AI's semantic understanding")  
        print("  • Advanced navigation planning capabilities")
        print("Drive your robot to build a high-quality semantic map.")
        print("Press Ctrl+C to stop and save the final map.")
        print("="*70 + "\n")
        
        last_save_time = time.time()
        last_status_time = time.time()
        start_time = time.time()
        
        try:
            while True:
                current_time = time.time()
                
                # Check system health
                if not self.robot.is_running():
                    logger.error("Lost connection to ROS2 topics")
                    break
                
                # Status update every 30 seconds
                if current_time - last_status_time > 30.0:
                    elapsed = current_time - start_time
                    obs_count = getattr(self.agent, 'obs_count', 0)
                    
                    # Check nvblox status on voxel map (the correct place now)
                    voxel_map_nvblox_enabled = getattr(self.voxel_map, 'use_nvblox_geometry', False)
                    voxel_map_has_nvblox_data = hasattr(self.voxel_map, 'nvblox_pointcloud') and self.voxel_map.nvblox_pointcloud is not None
                    
                    nvblox_status = "✓ ACTIVE" if voxel_map_nvblox_enabled and voxel_map_has_nvblox_data else (
                        "⚠ ENABLED-NO-DATA" if voxel_map_nvblox_enabled else "✗ DISABLED"
                    )
                    
                    # Agent nvblox should be disabled now
                    agent_nvblox_enabled = getattr(self.agent, 'use_nvblox', False)
                    
                    print(f"[Hybrid Status] Time: {elapsed:.1f}s | Observations: {obs_count}")
                    print(f"[nvblox Status] VoxelMap: {nvblox_status} | Agent: {'✗ DISABLED (no conflict)' if not agent_nvblox_enabled else '✓ ENABLED'}")
                    last_status_time = current_time
                
                # Save checkpoint periodically  
                if current_time - last_save_time > self.save_interval:
                    self.save_map(is_checkpoint=True)
                    last_save_time = current_time
                
                time.sleep(1.0)
                
        except KeyboardInterrupt:
            print("\n" + "="*70)
            print("HYBRID MAPPING STOPPED - Saving final map...")
            print("="*70)
            
        finally:
            # Stop agent
            print("Stopping hybrid mapping agent...")
            if self.agent and hasattr(self.agent, 'stop_realtime_updates'):
                self.agent.stop_realtime_updates()
            
            # Save final map
            self.save_map(is_checkpoint=False)
            
            print("\n" + "="*70)
            print("DISPLAYING FINAL HYBRID 3D MAP")
            print("="*70)
            self.show_final_map()
            
            print("\nHybrid mapping complete. You can close the visualization when done.")
            self.shutdown()

    def show_final_map(self):
        """Display the final hybrid nvblox + semantic map."""
        try:
            voxel_map = self.agent.get_voxel_map()
            
            logger.info("Displaying hybrid nvblox + Stretch AI map. Close the Open3D window to exit.")
            voxel_map.show(
                instances=self.use_semantic,
                add_planner_visuals=False  # Clean visualization, no red/green overlays
            )

        except Exception as e:
            logger.error(f"Could not display hybrid map: {e}")
            logger.info("You can still view the saved map file using:")
            logger.info(f"    python -m stretch.app.read_map -i <map_file.pkl> --show-svm")

    def save_map(self, is_checkpoint: bool):
        """Save the hybrid map to file."""
        if is_checkpoint:
            prefix = "nvblox_hybrid_checkpoint"
            print(f"[Checkpoint] Saving hybrid map...")
        else:
            prefix = "nvblox_hybrid_final"
            print(f"Saving final hybrid map...")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.pkl"
        self.agent.get_voxel_map().write_to_pickle(filename)
        print(f"Hybrid map saved to {filename}")

    def shutdown(self):
        """Clean shutdown with nvblox cleanup."""
        print("Shutting down hybrid nvblox + Stretch AI mapper...")
        
        if self.agent and hasattr(self.agent, 'stop_realtime_updates'):
            self.agent.stop_realtime_updates()
            
        if self.voxel_map and hasattr(self.voxel_map, 'cleanup'):
            self.voxel_map.cleanup()
            
        if self.robot:
            self.robot.stop()
            
        print("Hybrid shutdown complete.")


class HybridNvbloxAgent(RobotAgent):
    """Agent that combines nvblox 3D reconstruction with Stretch AI semantics."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info("Hybrid nvblox + Stretch AI agent initialized")
        
        # Debug information
        logger.info(f"nvblox integration: {'✓ ENABLED' if self.use_nvblox else '✗ DISABLED'}")
        if self.use_nvblox:
            logger.info(f"nvblox topic: {self.nvblox_topic}")
        logger.info(f"Realtime updates: {'✓ ENABLED' if self._realtime_updates else '✗ DISABLED'}")
        logger.info(f"Teleop only mode: {'✓ ENABLED' if self.teleop_only else '✗ DISABLED'}")


@click.command()
@click.option("--save-interval", default=60, help="Map checkpoint save interval (seconds)")
@click.option("--semantic", is_flag=True, help="Enable semantic mapping with object detection")
@click.option("--config", default="default_planner.yaml", help="Configuration file")
@click.option("--rerun/--no-rerun", default=False, help="Enable/disable Rerun visualization (default: disabled for hybrid mode)")
def main(save_interval, semantic, config, rerun):
    """Hybrid nvblox + Stretch AI mapping system."""
    
    print("Starting Hybrid nvblox + Stretch AI Mapping System...")
    print("This system combines the best of both worlds:")
    print("  • nvblox: High-quality 3D reconstruction")
    print("  • Stretch AI: Advanced semantic understanding")
    print("  • Integrated: Powerful navigation planning")
    
    if rerun:
        print("  • Rerun 3D visualization enabled!")
    else:
        print("  • Rerun disabled (recommended for hybrid mode)")
    
    mapper = ROS2NvbloxHybridMapper(save_interval, semantic, config, enable_rerun=rerun)
    if mapper.setup():
        mapper.run_mapping_loop()


if __name__ == "__main__":
    main()