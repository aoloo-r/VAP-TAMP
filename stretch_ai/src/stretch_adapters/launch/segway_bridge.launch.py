from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess

def generate_launch_description():
    return LaunchDescription([
        # Start ROS1-ROS2 bridge (assumes ROS1 is running on the Segway)
        ExecuteProcess(
            cmd=['ros2', 'run', 'ros1_bridge', 'dynamic_bridge', '--bridge-all-topics'],
            output='screen'
        ),
        
        # Start the observation adapter
        Node(
            package='stretch_adapters',
            executable='segway_observation_adapter.py',
            name='segway_observation_adapter',
            output='screen',
            parameters=[{'verbose': True}]
        )
    ])