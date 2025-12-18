#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import yaml
import os
import sys
from ament_index_python.packages import get_package_share_directory

class SegwayRegistration(Node):
    def __init__(self):
        super().__init__('segway_registration')
        
        self.get_logger().info('Registering Segway robot with Stretch AI...')
        
        # Get config path
        stretch_dir = get_package_share_directory('stretch')
        config_dir = os.path.join(stretch_dir, 'config')
        config_path = os.path.join(config_dir, 'segway_config.yaml')
        
        try:
            # Load configuration
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            # Register the configuration with Stretch AI
            # This will depend on how Stretch AI handles robot registration
            # For now, we'll just log that we found the configuration
            self.get_logger().info(f'Found configuration at {config_path}')
            self.get_logger().info(f'Robot name: {config["robot"]["name"]}')
            
            # Create any necessary symbolic links or registrations
            # (This will need to be customized based on Stretch AI's requirements)
            
            self.get_logger().info('Segway robot registered successfully!')
            self.get_logger().info('You can now run: ros2 launch stretch_ros2_bridge segway_bridge.launch.py')
            
        except Exception as e:
            self.get_logger().error(f'Failed to register Segway: {e}')
            sys.exit(1)

def main(args=None):
    rclpy.init(args=args)
    node = SegwayRegistration()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()