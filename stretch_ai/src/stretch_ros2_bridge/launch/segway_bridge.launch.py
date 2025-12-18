import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Get the launch directory
    stretch_dir = get_package_share_directory('stretch')
    config_dir = os.path.join(stretch_dir, 'config')
    
    # Create launch configuration variables
    config_file = os.path.join(config_dir, 'segway_config.yaml')
    
    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'config_file',
            default_value=config_file,
            description='Path to the Segway configuration file'
        ),
        
        # Launch the bridge server with custom config
        Node(
            package='stretch_ros2_bridge',
            executable='bridge_server',
            name='segway_bridge_server',
            parameters=[LaunchConfiguration('config_file')],
            output='screen',
            emulate_tty=True,
            arguments=['--no-d405']  # Since Segway doesn't have the gripper camera
        ),
        
        # Launch the observation adapter
        Node(
            package='stretch',
            executable='segway_observation_adapter.py',
            name='segway_observation_adapter',
            parameters=[LaunchConfiguration('config_file')],
            output='screen'
        )
    ])