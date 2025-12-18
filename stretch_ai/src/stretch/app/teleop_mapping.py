#!/usr/bin/env python3
# teleop_mapping.py - Teleoperation mapping script with nvblox integration.

import time
import datetime
import click
import numpy as np

from stretch.agent.robot_agent import RobotAgent
from stretch.agent.zmq_client import HomeRobotZmqClient
from stretch.mapping.voxel import SparseVoxelMap
from stretch.utils.logger import Logger
from stretch.perception.wrapper import create_semantic_sensor
from stretch.core.parameters import Parameters, get_parameters
from stretch.perception.encoders import get_encoder

logger = Logger(__name__)

class TeleopMapper:
    """Class to manage teleoperation mapping."""
    def __init__(self, robot_ip: str, save_interval: int, use_semantic: bool):
        self.robot_ip = robot_ip
        self.save_interval = save_interval
        self.use_semantic = use_semantic
        self.robot = None
        self.agent = None
        self.voxel_map = None

    def setup(self) -> bool:
        """Initialize robot, map, and agent."""
        print("\n" + "="*60)
        print("      INITIALIZING TELEOPERATION MAPPING")
        print("="*60)
        
        # 1. Create Robot Client
        print(f"Connecting to robot at: {self.robot_ip}...")
        self.robot = HomeRobotZmqClient(
            robot_ip=self.robot_ip,
            use_remote_computer=(self.robot_ip != "localhost"),
            enable_rerun_server=False  # Disable Rerun to eliminate warnings
        )
        if not self.robot.is_running():
            logger.error("Could not connect to the robot. Exiting.")
            return False
        print("✓ Robot client connected.")

        # 2. Create parameters
        print("Setting up parameters...")
        params = get_parameters("default_planner.yaml")
        params["use_scene_graph"] = True
        
        # Enhanced mapping parameters for cleaner, non-overlapping maps
        params["voxel_size"] = 0.05  # Larger voxels to merge nearby points
        params["obs_min_height"] = 0.15  # Filter ground noise more aggressively
        params["obs_max_height"] = 2.0   # Lower ceiling to reduce clutter
        params["obs_min_density"] = 10   # Higher density requirement
        params["min_points_per_voxel"] = 8  # More points required per voxel
        
        # Depth filtering for better quality
        params["min_depth"] = 0.5   # Ignore close points that cause clutter
        params["max_depth"] = 3.5   # Shorter range for cleaner mapping
        
        # More aggressive pose filtering to prevent overlapping
        params["min_movement_threshold"] = 0.1   # Require significant movement
        params["min_rotation_threshold"] = 0.15  # Require significant rotation
        
        # Temporal filtering to avoid rapid updates
        params["update_frequency_hz"] = 2  # Lower update frequency
        params["skip_frames"] = 5  # Skip frames to reduce overlapping
        
        # Memory management 
        params["max_observations"] = 500  # Reduce memory usage

        # 3. Create encoder
        print("Creating encoder...")
        encoder = get_encoder(params["encoder"], params.get("encoder_args", {}))
        print("✓ Encoder created.")

        # 4. Create Semantic Sensor if requested
        semantic_sensor = None
        if self.use_semantic:
            print("Creating semantic sensor (this may download models)...")
            semantic_sensor = create_semantic_sensor(params)
            print("✓ Semantic sensor created.")

        # 5. Create Voxel Map using the same method as RobotAgent
        print("Creating voxel map...")
        self.voxel_map = SparseVoxelMap.from_parameters(
            params,
            encoder,
            voxel_size=params["voxel_size"],
            use_instance_memory=self.use_semantic,
        )
        print("✓ Voxel map created.")

        # 6. Create Robot Agent with Visual SLAM integration  
        print("Creating robot agent with Visual SLAM integration...")
        self.agent = RobotAgent(
            robot=self.robot,
            parameters=params,
            voxel_map=self.voxel_map,
            semantic_sensor=semantic_sensor,
            use_instance_memory=self.use_semantic,
            enable_realtime_updates=True,
            teleop_only=True,
            use_nvblox=False,  # Use camera data with Visual SLAM odometry
        )
        print("✓ Robot agent created with Visual SLAM odometry integration.")
        print("✓ Background threads started.")
        return True

    def run_mapping_loop(self):
        """Main loop for continuous mapping with periodic status."""
        print("\n" + "="*60)
        print("MAPPING HAS STARTED!")
        print("Drive your robot around to build the map.")
        print("Map checkpoints will be saved periodically.")
        print("Press Ctrl+C to stop mapping and save the final map.")
        print("="*60 + "\n")
        
        last_save_time = time.time()
        last_status_time = time.time()
        start_time = time.time()
        
        try:
            while True:
                current_time = time.time()
                
                # Status update every 30 seconds
                if current_time - last_status_time > 30.0:
                    elapsed = current_time - start_time
                    obs_count = self.agent.obs_count if hasattr(self.agent, 'obs_count') else 0
                    print(f"[Status] Mapping time: {elapsed:.1f}s | Observations: {obs_count}")
                    last_status_time = current_time
                
                # Save checkpoint periodically
                if current_time - last_save_time > self.save_interval:
                    self.save_map(is_checkpoint=True)
                    last_save_time = current_time
                
                time.sleep(1.0)
                
        except KeyboardInterrupt:
            print("\n" + "="*60)
            print("MAPPING STOPPED - Saving final map...")
            print("="*60)
            
        finally:
            # Stop background threads
            print("Stopping background threads...")
            if self.agent:
                self.agent.stop_realtime_updates()
            
            # Save final map
            self.save_map(is_checkpoint=False)
            
            print("\n" + "="*60)
            print("DISPLAYING FINAL 3D MAP")
            print("="*60)
            self.show_final_map()
            
            print("\nMapping complete. You can close the visualization window when done.")
            self.shutdown()

    def show_final_map(self):
        """Display the final map."""
        try:
            if hasattr(self.agent, 'show_voxel_map'):
                self.agent.show_voxel_map()
            else:
                self.agent.get_voxel_map().show(instances=self.use_semantic)
        except Exception as e:
            print(f"Could not display map: {e}")

    def save_map(self, is_checkpoint: bool):
        """Saves the current voxel map to a pickle file."""
        if is_checkpoint:
            prefix = "checkpoint"
            print(f"[Checkpoint] Saving map...")
        else:
            prefix = "final_map"
            print(f"Saving final map...")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.pkl"
        self.agent.get_voxel_map().write_to_pickle(filename)
        print(f"Map saved to {filename}")

    def shutdown(self):
        """Cleanly shut down the agent and robot client."""
        print("Shutting down...")
        if self.agent:
            self.agent.stop_realtime_updates()
        if self.robot:
            self.robot.stop()
        print("Mapping complete.")


@click.command()
@click.option("--robot-ip", default="localhost", help="IP address of the robot.")
@click.option("--save-interval", default=60, help="Interval in seconds to save map checkpoints.")
@click.option("--semantic", is_flag=True, help="Enable semantic mapping with object detection.")
def main(robot_ip, save_interval, semantic):
    """Teleoperation mapping with nvblox integration."""
    mapper = TeleopMapper(robot_ip, save_interval, semantic)
    if mapper.setup():
        mapper.run_mapping_loop()


if __name__ == "__main__":
    main()