#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, JointState
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage
import numpy as np
import message_filters
from cv_bridge import CvBridge

class SegwayObservationAdapter(Node):
    def __init__(self):
        super().__init__('segway_observation_adapter')
        
        # Load parameters
        self.declare_parameter('sensors.camera.topics.color_image', '/camera/color/image_raw')
        self.declare_parameter('sensors.camera.topics.depth_image', '/camera/depth/image_rect_raw')
        self.declare_parameter('sensors.camera.topics.pointcloud', '/camera/depth/color/points')
        self.declare_parameter('navigation.topics.odometry', '/odom')
        self.declare_parameter('robot_state.topics.joint_states', '/joint_states')
        
        # Get parameters
        rgb_topic = self.get_parameter('sensors.camera.topics.color_image').value
        depth_topic = self.get_parameter('sensors.camera.topics.depth_image').value
        pointcloud_topic = self.get_parameter('sensors.camera.topics.pointcloud').value
        odom_topic = self.get_parameter('navigation.topics.odometry').value
        joint_topic = self.get_parameter('robot_state.topics.joint_states').value
        
        # Create CV bridge
        self.bridge = CvBridge()
        
        # Create synchronized subscribers for observation
        self.rgb_sub = message_filters.Subscriber(self, Image, rgb_topic)
        self.depth_sub = message_filters.Subscriber(self, Image, depth_topic)
        self.points_sub = message_filters.Subscriber(self, PointCloud2, pointcloud_topic)
        self.odom_sub = message_filters.Subscriber(self, Odometry, odom_topic)
        self.joint_sub = message_filters.Subscriber(self, JointState, joint_topic)
        
        # Use approximate time synchronization
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.points_sub, self.odom_sub, self.joint_sub],
            queue_size=10, slop=0.1
        )
        self.ts.registerCallback(self.observation_callback)
        
        # Create publisher for combined observation
        self.observation_pub = self.create_publisher(
            Image, '/stretch_ai/observation', 10
        )
        
        self.get_logger().info('Segway Observation Adapter is running')
        self.get_logger().info(f'Listening to RGB topic: {rgb_topic}')
        self.get_logger().info(f'Listening to depth topic: {depth_topic}')
    
    def observation_callback(self, rgb_msg, depth_msg, points_msg, odom_msg, joint_msg):
        try:
            # Process the synchronized messages to create an observation
            # This is a simplified example - adapt to what Stretch AI expects
            rgb_img = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            depth_img = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
            
            # Log success in receiving data
            self.get_logger().info('Received synchronized data from all topics')
            
            # Create observation message (simplified)
            observation_msg = self.bridge.cv2_to_imgmsg(rgb_img, "bgr8")
            observation_msg.header = rgb_msg.header
            
            # Publish observation
            self.observation_pub.publish(observation_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error in observation callback: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = SegwayObservationAdapter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()