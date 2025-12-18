#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, JointState
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage
import time
import threading

class SegwayObservationAdapter(Node):
    def __init__(self):
        super().__init__('segway_observation_adapter')
        
        self.declare_parameter('verbose', True)
        verbose = self.get_parameter('verbose').value
        
        # Create publishers for the topics that Stretch AI expects
        self.image_pub = self.create_publisher(
            Image, '/stretch/camera/color/image_raw', 10)
        self.depth_pub = self.create_publisher(
            Image, '/stretch/camera/depth/image_rect_raw', 10)
        self.pointcloud_pub = self.create_publisher(
            PointCloud2, '/stretch/camera/depth/color/points', 10)
        self.camera_info_pub = self.create_publisher(
            CameraInfo, '/stretch/camera/color/camera_info', 10)
        self.depth_info_pub = self.create_publisher(
            CameraInfo, '/stretch/camera/depth/camera_info', 10)
        self.joint_states_pub = self.create_publisher(
            JointState, '/stretch/joint_states', 10)
        self.odom_pub = self.create_publisher(
            Odometry, '/stretch/odom', 10)
        
        # Create subscribers to the Segway topics
        self.image_sub = self.create_subscription(
            Image, '/camera/color/image_raw', 
            lambda msg: self.republish(msg, self.image_pub, 'color image'), 10)
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth/image_rect_raw', 
            lambda msg: self.republish(msg, self.depth_pub, 'depth image'), 10)
        self.pointcloud_sub = self.create_subscription(
            PointCloud2, '/camera/depth/color/points', 
            lambda msg: self.republish(msg, self.pointcloud_pub, 'pointcloud'), 10)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/color/camera_info', 
            lambda msg: self.republish(msg, self.camera_info_pub, 'camera info'), 10)
        self.depth_info_sub = self.create_subscription(
            CameraInfo, '/camera/depth/camera_info', 
            lambda msg: self.republish(msg, self.depth_info_pub, 'depth camera info'), 10)
        self.joint_states_sub = self.create_subscription(
            JointState, '/joint_states', 
            lambda msg: self.republish(msg, self.joint_states_pub, 'joint states'), 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', 
            lambda msg: self.republish(msg, self.odom_pub, 'odometry'), 10)
        
        # Create a timer to report status
        if verbose:
            self.timer = self.create_timer(5.0, self.report_status)
            self.message_counts = {
                'color image': 0,
                'depth image': 0, 
                'pointcloud': 0,
                'camera info': 0,
                'depth camera info': 0,
                'joint states': 0,
                'odometry': 0
            }
        
        self.get_logger().info('Segway Observation Adapter is running')
    
    def republish(self, msg, publisher, msg_type):
        # Add a header frame_id if it doesn't exist
        if not msg.header.frame_id:
            msg.header.frame_id = 'segway_' + msg_type.replace(' ', '_')
        
        # Update timestamp to current time
        msg.header.stamp = self.get_clock().now().to_msg()
        
        # Publish the message
        publisher.publish(msg)
        
        # Update message count
        if hasattr(self, 'message_counts'):
            self.message_counts[msg_type] += 1
    
    def report_status(self):
        status = "Message counts: "
        for msg_type, count in self.message_counts.items():
            status += f"{msg_type}: {count}, "
            # Reset the counter
            self.message_counts[msg_type] = 0
        
        self.get_logger().info(status)

def main(args=None):
    rclpy.init(args=args)
    node = SegwayObservationAdapter()
    
    # Use a multithreaded executor to improve performance
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    
    try:
        while rclpy.ok():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    
    rclpy.shutdown()

if __name__ == '__main__':
    main()