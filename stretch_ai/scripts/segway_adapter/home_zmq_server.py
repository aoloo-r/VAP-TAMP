#!/usr/bin/env python3
# home_zmq_server.py - Version updated for RTAB-Map SLAM integration
#
# Key changes:
# 1. REMOVED Odometry subscription ('/odom').
# 2. MODIFIED the TF lookup to get the robot's base pose from the 'map' frame,
#    which is provided by RTAB-Map and corrected for drift.
# 3. PRESERVED custom depth image enhancement logic.

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.time import Time

from sensor_msgs.msg import Image, CameraInfo, JointState, PointCloud2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
import tf2_ros
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
from scipy.spatial.transform import Rotation as R

import numpy as np
import zmq
import time
import threading
import queue
from cv_bridge import CvBridge
import cv2
from sensor_msgs_py import point_cloud2
import stretch.utils.compression as compression
from typing import Tuple

def downsample_pointcloud(points, target_count=50000):
    if len(points) <= target_count:
        return points
    stride = max(1, len(points) // target_count)
    return points[::stride][:target_count]

def pointcloud2_to_xyz_array(msg: PointCloud2) -> np.ndarray:
    points = []
    for pt in point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
        if isinstance(pt, (tuple, list, np.ndarray)) and all(abs(coord) < 20.0 for coord in pt):
            points.append([float(coord) for coord in pt])
    return downsample_pointcloud(np.array(points, dtype=np.float32)) if points else np.zeros((0, 3), dtype=np.float32)

class PublisherThread(threading.Thread):
    def __init__(self, ctx, logger, navigation_callback=None):
        super().__init__()
        self.daemon = True
        self.logger = logger
        self.navigation_callback = navigation_callback
        self.obs_queue = queue.Queue(maxsize=1)
        self.state_queue = queue.Queue(maxsize=1)
        self.servo_queue = queue.Queue(maxsize=1)
        self.running = True
        self.obs_pub = ctx.socket(zmq.PUB)
        self.state_pub = ctx.socket(zmq.PUB)
        self.servo_pub = ctx.socket(zmq.PUB)
        # Action socket for receiving navigation commands (SUB to match client's PUB)
        self.action_sock = ctx.socket(zmq.SUB)
        self.action_sock.setsockopt(zmq.SUBSCRIBE, b'')  # Subscribe to all messages
        for pub in [self.obs_pub, self.state_pub, self.servo_pub]:
            pub.setsockopt(zmq.SNDHWM, 1)
        self.obs_pub.bind("tcp://0.0.0.0:4401")
        self.state_pub.bind("tcp://0.0.0.0:4403")
        self.servo_pub.bind("tcp://0.0.0.0:4404")
        self.action_sock.bind("tcp://0.0.0.0:4402")  # Action receiver for navigation goals

    def _put_in_queue(self, q, item):
        if q.full():
            try: q.get_nowait()
            except queue.Empty: pass
        try: q.put_nowait(item)
        except queue.Full: pass

    def put_obs(self, obs): self._put_in_queue(self.obs_queue, obs)
    def put_state(self, state): self._put_in_queue(self.state_queue, state)
    def put_servo(self, servo): self._put_in_queue(self.servo_queue, servo)
    def stop(self): self.running = False
    
    def run(self):
        self.logger.info("ZMQ PublisherThread started.")
        while self.running:
            try:
                obs = self.obs_queue.get(block=False)
                self.obs_pub.send_pyobj(obs)
            except queue.Empty: pass
            
            try:
                state = self.state_queue.get(block=False)
                self.state_pub.send_pyobj(state)
            except queue.Empty: pass

            try:
                servo = self.servo_queue.get(block=False)
                self.servo_pub.send_pyobj(servo)
            except queue.Empty: pass

            # Process navigation goal actions
            try:
                if self.action_sock.poll(timeout=0):
                    action = self.action_sock.recv_pyobj(flags=zmq.NOBLOCK)
                    self.logger.info(f"🔍 Received action message: {action}")
                    # Check if this is a navigation goal command
                    if action and 'navigation_goal' in action and self.navigation_callback:
                        goal = action['navigation_goal']
                        self.logger.info(f"✅ Received navigation goal: x={goal[0]:.2f}, y={goal[1]:.2f}, theta={goal[2]:.2f}")
                        self.navigation_callback(goal)
                    elif action:
                        self.logger.warning(f"⚠️ Received action but no navigation_goal key. Keys: {action.keys()}")
            except zmq.Again:
                pass
            except Exception as e:
                self.logger.error(f"❌ Error processing action: {e}")
            
            time.sleep(0.005)
        
        for pub in [self.obs_pub, self.state_pub, self.servo_pub, self.action_sock]:
            pub.close()
        self.logger.info("ZMQ PublisherThread stopped.")

class HomeZmqServer(Node):
    def __init__(self):
        super().__init__('home_zmq_server')
        self.target_width = 640
        self.target_height = 480
        
        self.original_width = None
        self.original_height = None

        # Create move_base goal publisher
        self.goal_publisher = self.create_publisher(PoseStamped, '/move_base_simple/goal', 10)
        self.get_logger().info("Created /move_base_simple/goal publisher")

        self.ctx = zmq.Context()
        self.publisher_thread = PublisherThread(self.ctx, self.get_logger(),
                                               navigation_callback=self.publish_navigation_goal)
        self.publisher_thread.start()
        
        self.bridge = CvBridge()
        self.latest_color_img = None
        self.latest_depth_img = None
        self.latest_camK = None
        self.latest_odom_pose = np.zeros(3, dtype=np.float32) # Raw odometry (backup)
        self.latest_amcl_pose = None  # AMCL localized pose in map frame (primary)
        self.latest_joint_positions = None
        self.latest_joint_velocities = None
        self.latest_joint_efforts = None
        self.latest_lidar_points = None
        self.latest_lidar_timestamp = None
        
        # Storage for additional robot topics
        self.additional_topic_data = {}
        self.discovered_topics = set()
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.latest_camera_pose = np.eye(4, dtype=np.float32) # Default identity

        self.step = 0
        self.frame_skip_counter = 0
        self.frame_skip_interval = 3  # Only process every 3rd frame to reduce clutter
        
        sensor_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=5, reliability=ReliabilityPolicy.RELIABLE)
        
        # Subscribe to core required topics
        self.create_subscription(Image, '/camera/color/image_raw', self.color_image_cb, sensor_qos)
        self.create_subscription(Image, '/camera/depth/image_rect_raw', self.depth_image_cb, sensor_qos)
        self.create_subscription(CameraInfo, '/camera/color/camera_info', self.camera_info_cb, sensor_qos)
        self.create_subscription(PointCloud2, '/camera/depth/color/points', self.pointcloud_cb, sensor_qos)
        # Subscribe to AMCL pose (localized position in map frame) - PRIMARY
        # AMCL publishes to /amcl_pose (PoseWithCovarianceStamped in ROS1)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.amcl_pose_cb, sensor_qos)
        # Subscribe to raw odometry as backup
        self.create_subscription(Odometry, '/odom', self.basic_odom_cb, sensor_qos)
        self.create_subscription(JointState, '/joint_states', self.joint_states_cb, sensor_qos)
        
        # Start topic discovery timer (runs once after initialization)
        self.discovery_timer = self.create_timer(2.0, self.discover_and_subscribe_to_robot_topics)
        
        self.timer = self.create_timer(0.033, self.timer_callback) # ~30Hz
        self.get_logger().info("ZMQ Server for SLAM initialized.")

    def color_image_cb(self, msg: Image):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if self.original_width is None:
                self.original_width, self.original_height = msg.width, msg.height
            
            if cv_img.shape[1] != self.target_width or cv_img.shape[0] != self.target_height:
                cv_img = cv2.resize(cv_img, (self.target_width, self.target_height), interpolation=cv2.INTER_AREA)
            
            self.latest_color_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            self.get_logger().error(f"Color image callback failed: {e}")

    def depth_image_cb(self, msg: Image):
        """Process depth image with hole filling and enhancement for reflective surfaces"""
        try:
            encoding = '16UC1' if msg.encoding == '16UC1' else '32FC1'
            cv_depth = self.bridge.imgmsg_to_cv2(msg, encoding)
            
            if encoding == '32FC1':
                cv_depth = (cv_depth * 1000).astype(np.uint16)
            
            if self.original_width is None:
                self.original_width, self.original_height = msg.width, msg.height
            
            cv_depth_enhanced = self.enhance_depth_image(cv_depth)
            
            if cv_depth_enhanced.shape[1] != self.target_width or cv_depth_enhanced.shape[0] != self.target_height:
                cv_depth_final = cv2.resize(cv_depth_enhanced, (self.target_width, self.target_height), 
                                        interpolation=cv2.INTER_NEAREST)
            else:
                cv_depth_final = cv_depth_enhanced
            
            self.latest_depth_img = cv_depth_final
            
        except Exception as e:
            self.get_logger().error(f"Depth image callback failed: {e}")

    def enhance_depth_image(self, depth_img: np.ndarray) -> np.ndarray:
        """Enhance depth image by filling holes, handling reflective surfaces, and filtering noise."""
        depth_enhanced = depth_img.copy().astype(np.float32)
        max_depth, min_depth = 4000, 100
        invalid_mask = (depth_enhanced == 0) | (depth_enhanced > max_depth) | (depth_enhanced < min_depth)
        
        depth_median = cv2.medianBlur(depth_enhanced.astype(np.uint16), 5).astype(np.float32)
        
        if np.any(invalid_mask):
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            depth_closed = cv2.morphologyEx(depth_median, cv2.MORPH_CLOSE, kernel_small)
            depth_enhanced[invalid_mask] = depth_closed[invalid_mask]
            invalid_mask = (depth_enhanced == 0) | (depth_enhanced > max_depth)
        
        if np.any(invalid_mask):
            depth_bilateral = cv2.bilateralFilter(depth_enhanced.astype(np.float32), d=9, sigmaColor=25, sigmaSpace=25)
            depth_enhanced[invalid_mask] = depth_bilateral[invalid_mask]
            invalid_mask = (depth_enhanced == 0)

        remaining_invalid = (depth_enhanced == 0).astype(np.uint8)
        if np.any(remaining_invalid):
            depth_inpainted = cv2.inpaint(depth_enhanced.astype(np.float32), remaining_invalid, inpaintRadius=7, flags=cv2.INPAINT_NS)
            depth_enhanced = depth_inpainted
        
        depth_smooth = cv2.bilateralFilter(depth_enhanced.astype(np.float32), d=5, sigmaColor=10, sigmaSpace=10)
        depth_final = np.clip(depth_smooth, min_depth, max_depth)
        depth_final = np.nan_to_num(depth_final, nan=0.0, posinf=max_depth, neginf=0.0)
        
        return depth_final.astype(np.uint16)

    def camera_info_cb(self, msg: CameraInfo):
        original_K = np.array(msg.k).reshape((3, 3))
        if self.original_width is not None and self.original_width > 0:
            scale_x = self.target_width / self.original_width
            scale_y = self.target_height / self.original_height
            K = original_K.copy()
            K[0, 0] *= scale_x; K[1, 1] *= scale_y; K[0, 2] *= scale_x; K[1, 2] *= scale_y
            self.latest_camK = K.astype(np.float32)
        else:
            self.latest_camK = original_K.astype(np.float32)

    def amcl_pose_cb(self, msg):
        """Callback for AMCL localized pose (map frame) - PRIMARY pose source

        Accepts either PoseWithCovarianceStamped (ROS1 AMCL standard) or PoseStamped
        """
        try:
            # Handle PoseWithCovarianceStamped (AMCL standard output)
            if hasattr(msg, 'pose') and hasattr(msg.pose, 'pose'):
                pos = msg.pose.pose.position
                quat = msg.pose.pose.orientation
            # Handle PoseStamped (if bridge converts it)
            elif hasattr(msg, 'pose'):
                pos = msg.pose.position
                quat = msg.pose.orientation
            else:
                self.get_logger().error(f"Unknown AMCL message type: {type(msg)}")
                return

            yaw = R.from_quat([quat.x, quat.y, quat.z, quat.w]).as_euler('xyz', degrees=False)[2]
            self.latest_amcl_pose = np.array([pos.x, pos.y, yaw], dtype=np.float32)

            # Log first successful AMCL reception
            if not hasattr(self, '_amcl_received'):
                self.get_logger().info(f"✅ AMCL pose received: ({pos.x:.2f}, {pos.y:.2f}, θ={yaw:.2f})")
                self._amcl_received = True
        except Exception as e:
            self.get_logger().error(f"AMCL pose callback failed: {e}")

    def basic_odom_cb(self, msg: Odometry):
        """Callback for basic odometry (backup only)"""
        try:
            pos = msg.pose.pose.position
            quat = msg.pose.pose.orientation
            yaw = R.from_quat([quat.x, quat.y, quat.z, quat.w]).as_euler('xyz', degrees=False)[2]
            self.latest_odom_pose = np.array([pos.x, pos.y, yaw], dtype=np.float32)
        except Exception as e:
            self.get_logger().error(f"Basic odometry callback failed: {e}")

    def joint_states_cb(self, msg: JointState):
        self.latest_joint_positions = np.array(msg.position, dtype=np.float32)
        self.latest_joint_velocities = np.array(msg.velocity, dtype=np.float32)
        self.latest_joint_efforts = np.array(msg.effort, dtype=np.float32)

    def pointcloud_cb(self, msg: PointCloud2):
        try:
            self.latest_lidar_points = pointcloud2_to_xyz_array(msg)
            self.latest_lidar_timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        except Exception as e:
            self.get_logger().error(f"Pointcloud callback failed: {e}")

    def discover_and_subscribe_to_robot_topics(self):
        """Discover and subscribe to all robot topics that have been bridged."""
        try:
            # Get list of all available topics
            topic_names_and_types = self.get_topic_names_and_types()
            
            # Filter for robot-related topics (excluding ROS2 system topics)
            robot_topics = []
            exclude_patterns = ['/rosout', '/parameter_events', '/tf', '/tf_static', '/clock']
            core_topics = ['/camera/color/image_raw', '/camera/depth/image_rect_raw', 
                          '/camera/color/camera_info', '/camera/depth/color/points', 
                          '/odom', '/joint_states']
            
            for topic_name, topic_types in topic_names_and_types:
                # Skip if it's a core topic we already subscribe to
                if topic_name in core_topics:
                    continue
                    
                # Skip ROS2 system topics
                if any(pattern in topic_name for pattern in exclude_patterns):
                    continue
                    
                # Skip if we've already discovered this topic
                if topic_name in self.discovered_topics:
                    continue
                    
                # Add robot topics that might be useful
                if any(keyword in topic_name.lower() for keyword in 
                      ['camera', 'sensor', 'laser', 'scan', 'imu', 'gps', 'cmd_vel']):
                    robot_topics.append((topic_name, topic_types[0]))
            
            if robot_topics:
                self.get_logger().info(f"Discovered {len(robot_topics)} additional robot topics:")
                
                sensor_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=5, 
                                      reliability=ReliabilityPolicy.RELIABLE)
                
                for topic_name, topic_type in robot_topics:
                    try:
                        self.get_logger().info(f"  -> Subscribing to {topic_name} ({topic_type})")
                        
                        # Create generic callback that stores the data
                        callback = lambda msg, tn=topic_name: self.generic_topic_callback(msg, tn)
                        
                        # Import the message type dynamically
                        module_name, class_name = topic_type.rsplit('/', 1)
                        module_parts = module_name.split('.')
                        
                        # Try to import the message type
                        if module_parts[0] == 'sensor_msgs':
                            from sensor_msgs.msg import LaserScan, Imu, NavSatFix
                            msg_class_map = {
                                'LaserScan': LaserScan,
                                'Imu': Imu, 
                                'NavSatFix': NavSatFix
                            }
                            if class_name in msg_class_map:
                                self.create_subscription(msg_class_map[class_name], topic_name, callback, sensor_qos)
                                self.discovered_topics.add(topic_name)
                        elif module_parts[0] == 'geometry_msgs':
                            from geometry_msgs.msg import Twist, PoseStamped
                            msg_class_map = {
                                'Twist': Twist,
                                'PoseStamped': PoseStamped
                            }
                            if class_name in msg_class_map:
                                self.create_subscription(msg_class_map[class_name], topic_name, callback, sensor_qos)
                                self.discovered_topics.add(topic_name)
                        else:
                            self.get_logger().debug(f"Unknown message type for {topic_name}: {topic_type}")
                            
                    except Exception as e:
                        self.get_logger().warn(f"Failed to subscribe to {topic_name}: {e}")
            else:
                self.get_logger().info("No additional robot topics discovered")
                
        except Exception as e:
            self.get_logger().error(f"Topic discovery failed: {e}")
        
        # Cancel this timer since it only needs to run once
        self.discovery_timer.cancel()

    def generic_topic_callback(self, msg, topic_name: str):
        """Generic callback for additional robot topics."""
        try:
            # Store timestamp and basic info
            timestamp = None
            if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
                timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            
            # Store the message data
            self.additional_topic_data[topic_name] = {
                'data': msg,
                'timestamp': timestamp,
                'last_updated': time.time()
            }
            
            # Log occasionally to show data is flowing
            if self.step % 300 == 0:  # Every ~10 seconds at 30Hz
                self.get_logger().debug(f"Received data on {topic_name}")
                
        except Exception as e:
            self.get_logger().error(f"Error in generic callback for {topic_name}: {e}")

    def publish_navigation_goal(self, goal_xyt):
        """Publish navigation goal to move_base.

        Args:
            goal_xyt: [x, y, theta] goal position in map frame
        """
        try:
            import math

            goal_msg = PoseStamped()
            goal_msg.header.frame_id = "map"
            goal_msg.header.stamp = self.get_clock().now().to_msg()

            # Set position
            goal_msg.pose.position.x = float(goal_xyt[0])
            goal_msg.pose.position.y = float(goal_xyt[1])
            goal_msg.pose.position.z = 0.0

            # Convert theta to quaternion
            theta = float(goal_xyt[2])
            goal_msg.pose.orientation.w = math.cos(theta / 2.0)
            goal_msg.pose.orientation.x = 0.0
            goal_msg.pose.orientation.y = 0.0
            goal_msg.pose.orientation.z = math.sin(theta / 2.0)

            self.goal_publisher.publish(goal_msg)
            self.get_logger().info(f"📍 Published navigation goal: ({goal_xyt[0]:.2f}, {goal_xyt[1]:.2f}, {math.degrees(theta):.1f}°)")

        except Exception as e:
            self.get_logger().error(f"Failed to publish navigation goal: {e}")

    def lookup_poses(self) -> Tuple[np.ndarray, np.ndarray]:
        """Looks up both the camera and robot base poses using AMCL (map frame) and TF."""
        camera_pose = self.latest_camera_pose

        # Use AMCL pose if available (map frame), otherwise fall back to odometry
        if self.latest_amcl_pose is not None:
            base_pose_xyt = self.latest_amcl_pose  # PRIMARY: AMCL localized pose in map frame
        else:
            base_pose_xyt = self.latest_odom_pose  # FALLBACK: raw odometry
            if not hasattr(self, '_amcl_warning_shown'):
                self.get_logger().warn("⚠️  AMCL pose not available, using raw odometry (may be inaccurate)")
                self._amcl_warning_shown = True

        try:
            # Get the camera pose relative to the robot's base from TF
            # Try multiple frame combinations to find what's available
            
            # First, check what frames are available
            available_frames = self.tf_buffer.all_frames_as_string()
            
            # Try different base frame options
            base_frame = None
            camera_frame = None
            
            # Check for base frames in order of preference
            for frame in ['base_footprint', 'base_link', 'base']:
                if frame in available_frames:
                    base_frame = frame
                    break
            
            # Check for camera frames in order of preference  
            for frame in ['camera_color_optical_frame', 'camera_depth_optical_frame', 'camera_link']:
                if frame in available_frames:
                    camera_frame = frame
                    break
            
            if base_frame and camera_frame:
                base_to_camera_trans = self.tf_buffer.lookup_transform(base_frame, camera_frame, Time())
                t, q = base_to_camera_trans.transform.translation, base_to_camera_trans.transform.rotation
                T = np.eye(4, dtype=np.float32)
                T[0:3, 0:3] = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
                T[0, 3], T[1, 3], T[2, 3] = t.x, t.y, t.z
                camera_pose = T
                self.latest_camera_pose = camera_pose
                
                # Log success once
                if not hasattr(self, '_tf_success_logged'):
                    self.get_logger().info(f'Successfully using TF: {base_frame} -> {camera_frame}')
                    self._tf_success_logged = True
            else:
                if not hasattr(self, '_tf_frames_logged'):
                    self.get_logger().warn(f'TF frames not available yet. Available: {available_frames[:200]}...', throttle_duration_sec=10)
                    self._tf_frames_logged = True

        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            if not hasattr(self, '_tf_error_logged'):
                self.get_logger().warn(f'Could not lookup camera transform: {e}', throttle_duration_sec=10)
                self._tf_error_logged = True
            
        return camera_pose, base_pose_xyt

    def timer_callback(self):
        # Skip frames to reduce overlapping observations
        self.frame_skip_counter += 1
        if self.frame_skip_counter < self.frame_skip_interval:
            return
        self.frame_skip_counter = 0
        
        camera_pose, base_pose_xyt = self.lookup_poses()

        if not all([self.latest_color_img is not None, self.latest_depth_img is not None, self.latest_camK is not None, self.latest_joint_positions is not None]):
            return

        self.step += 1
        
        obs = {'step': self.step}
        obs['rgb'] = compression.to_jpg(self.latest_color_img)
        obs['depth'] = compression.to_jp2(self.latest_depth_img)
        obs['rgb_height'] = self.target_height
        obs['rgb_width'] = self.target_width
        obs['camera_K'] = self.latest_camK
        obs['camera_pose'] = camera_pose
        obs['gps'] = base_pose_xyt[:2]
        obs['compass'] = base_pose_xyt[2:3]
        obs['joint'] = self.latest_joint_positions
        obs['joint_velocities'] = self.latest_joint_velocities
        if self.latest_lidar_points is not None:
            obs['lidar_points'] = self.latest_lidar_points
            obs['lidar_timestamp'] = self.latest_lidar_timestamp
        
        # Add additional robot topic data
        if self.additional_topic_data:
            obs['additional_topics'] = {}
            current_time = time.time()
            for topic_name, topic_info in self.additional_topic_data.items():
                # Only include recent data (within last 5 seconds)
                if current_time - topic_info['last_updated'] < 5.0:
                    # Convert ROS message to dict/basic types for ZMQ serialization
                    try:
                        obs['additional_topics'][topic_name] = {
                            'timestamp': topic_info['timestamp'],
                            'last_updated': topic_info['last_updated'],
                            'type': str(type(topic_info['data']).__name__)
                        }
                        
                        # Add specific data based on message type
                        msg = topic_info['data']
                        if hasattr(msg, 'data') and isinstance(msg.data, (int, float, str, bool)):
                            obs['additional_topics'][topic_name]['data'] = msg.data
                        elif hasattr(msg, 'linear') and hasattr(msg, 'angular'):  # Twist
                            obs['additional_topics'][topic_name]['linear'] = [msg.linear.x, msg.linear.y, msg.linear.z]
                            obs['additional_topics'][topic_name]['angular'] = [msg.angular.x, msg.angular.y, msg.angular.z]
                        elif hasattr(msg, 'ranges'):  # LaserScan
                            obs['additional_topics'][topic_name]['ranges_count'] = len(msg.ranges)
                            obs['additional_topics'][topic_name]['angle_min'] = msg.angle_min
                            obs['additional_topics'][topic_name]['angle_max'] = msg.angle_max
                            # Don't include full ranges array to avoid large messages
                        
                    except Exception as e:
                        self.get_logger().debug(f"Failed to serialize {topic_name}: {e}")
        
        self.publisher_thread.put_obs(obs)

        state = {'step': self.step, 'base_pose': base_pose_xyt, 'joint_positions': self.latest_joint_positions, 'joint_velocities': self.latest_joint_velocities, 'joint_efforts': self.latest_joint_efforts, 'control_mode': "navigation", 'at_goal': True, 'is_homed': True, 'is_runstopped': False}
        self.publisher_thread.put_state(state)

        servo = {'head_cam/color_image': obs['rgb'], 'head_cam/depth_image': obs['depth'], 'head_cam/depth_camera_K': self.latest_camK, 'head_cam/pose': obs['camera_pose'], 'head_cam/depth_scaling': 1.0, 'head_cam/image_scaling': 1.0, 'ee/pose': np.eye(4, dtype=np.float32), 'robot/config': self.latest_joint_positions}
        self.publisher_thread.put_servo(servo)

        # Log only once when observations start
        if self.step == 1:
            self.get_logger().info("✅ Observations started - publishing at ~30Hz")

def main(args=None):
    rclpy.init(args=args)
    node = HomeZmqServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down ZMQ server.")
    finally:
        node.publisher_thread.stop()
        node.publisher_thread.join(timeout=1.0)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
