#!/usr/bin/env python3
# check_nvblox_system.py - Comprehensive system check for nvblox + Stretch AI integration

import subprocess
import time
import sys

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import SingleThreadedExecutor
    from sensor_msgs.msg import Image, PointCloud2, CameraInfo
    from geometry_msgs.msg import PoseWithCovarianceStamped
    from nav_msgs.msg import Odometry
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False

class SystemChecker(Node):
    """Comprehensive system checker for nvblox integration."""
    
    def __init__(self):
        super().__init__('stretch_ai_system_checker')
        
        self.topic_data = {}
        self.required_topics = {
            # Camera topics
            '/camera0/color/image_raw': ('Image', 'RGB camera feed'),
            '/camera0/depth/image_rect_raw': ('Image', 'Depth camera feed'),  
            '/camera0/color/camera_info': ('CameraInfo', 'Camera calibration'),
            
            # Alternative camera topics (if main ones don't work)
            '/camera/color/image_raw': ('Image', 'RGB camera feed (alt)'),
            '/camera/depth/image_rect_raw': ('Image', 'Depth camera feed (alt)'),
            '/camera/color/camera_info': ('CameraInfo', 'Camera calibration (alt)'),
            
            # Robot state
            '/joint_states': ('JointState', 'Robot joint states'),
            '/odom': ('Odometry', 'Wheel odometry'),
            '/visual_slam/tracking/odometry': ('Odometry', 'Visual SLAM odometry'),
            
            # nvblox topics  
            '/nvblox_node/static_esdf_pointcloud': ('PointCloud2', 'nvblox 3D reconstruction'),
            '/nvblox_node/mesh': ('Mesh', 'nvblox mesh'),
            '/nvblox_node/pessimistic_static_esdf_pointcloud': ('PointCloud2', 'nvblox pessimistic map'),
            
            # TF
            '/tf': ('TFMessage', 'Transform broadcasts'),
            '/tf_static': ('TFMessage', 'Static transforms'),
        }
        
        self.subscribers = {}
        self.message_counts = {topic: 0 for topic in self.required_topics.keys()}
        
        # Create subscribers for topic monitoring
        self.create_topic_subscribers()
        
    def create_topic_subscribers(self):
        """Create subscribers to monitor topic activity."""
        
        # Generic callback to count messages
        def make_callback(topic_name):
            def callback(msg):
                self.message_counts[topic_name] += 1
                if topic_name not in self.topic_data:
                    self.topic_data[topic_name] = {
                        'first_message_time': time.time(),
                        'message_type': type(msg).__name__,
                        'active': True
                    }
            return callback
        
        # Subscribe to each topic (with error handling)
        for topic, (msg_type, description) in self.required_topics.items():
            try:
                if msg_type == 'Image':
                    from sensor_msgs.msg import Image
                    self.subscribers[topic] = self.create_subscription(
                        Image, topic, make_callback(topic), 1)
                elif msg_type == 'PointCloud2':
                    from sensor_msgs.msg import PointCloud2
                    self.subscribers[topic] = self.create_subscription(
                        PointCloud2, topic, make_callback(topic), 1)
                elif msg_type == 'CameraInfo':
                    from sensor_msgs.msg import CameraInfo
                    self.subscribers[topic] = self.create_subscription(
                        CameraInfo, topic, make_callback(topic), 1)
                elif msg_type == 'Odometry':
                    from nav_msgs.msg import Odometry
                    self.subscribers[topic] = self.create_subscription(
                        Odometry, topic, make_callback(topic), 1)
                elif msg_type == 'JointState':
                    from sensor_msgs.msg import JointState
                    self.subscribers[topic] = self.create_subscription(
                        JointState, topic, make_callback(topic), 1)
                # TF and Mesh topics handled separately
                    
            except Exception as e:
                self.get_logger().warn(f"Could not subscribe to {topic}: {e}")
    
    def run_diagnostics(self):
        """Run comprehensive system diagnostics."""
        
        print("\n" + "="*80)
        print("STRETCH AI + NVBLOX SYSTEM DIAGNOSTIC")
        print("="*80)
        
        # 1. Check ROS2 installation
        print("\n1. ROS2 System Check:")
        if HAS_ROS2:
            print("  ✓ ROS2 Python libraries available")
        else:
            print("  ✗ ROS2 Python libraries missing")
            return False
        
        # 2. Check available topics
        print("\n2. Available Topics:")
        available_topics = self.get_available_topics()
        
        essential_found = 0
        essential_topics = [
            '/camera0/color/image_raw',
            '/camera0/depth/image_rect_raw', 
            '/nvblox_node/static_esdf_pointcloud',
            '/odom'
        ]
        
        for topic in essential_topics:
            if topic in available_topics:
                print(f"  ✓ {topic}")
                essential_found += 1
            else:
                print(f"  ✗ {topic} (MISSING)")
        
        print(f"\nEssential topics found: {essential_found}/{len(essential_topics)}")
        
        # 3. Check topic activity (monitor for 10 seconds)
        print(f"\n3. Topic Activity Check (monitoring for 10 seconds)...")
        
        executor = SingleThreadedExecutor()
        executor.add_node(self)
        
        start_time = time.time()
        timeout = 10.0
        
        while time.time() - start_time < timeout:
            executor.spin_once(timeout_sec=0.1)
            
            # Print progress
            elapsed = time.time() - start_time
            print(f"\r  Progress: {elapsed:.1f}/{timeout:.1f}s", end="", flush=True)
        
        print("")  # New line after progress
        
        # 4. Report topic activity
        print("\n4. Topic Activity Results:")
        active_count = 0
        
        for topic, (msg_type, description) in self.required_topics.items():
            count = self.message_counts.get(topic, 0)
            rate = count / timeout if timeout > 0 else 0
            
            if count > 0:
                print(f"  ✓ {topic:<50} {count:3d} msgs ({rate:4.1f} Hz) - {description}")
                active_count += 1
            elif topic in available_topics:
                print(f"  ⚠ {topic:<50} {count:3d} msgs (INACTIVE) - {description}")
            else:
                print(f"  ✗ {topic:<50} NOT AVAILABLE - {description}")
        
        # 5. Check TF transforms
        print(f"\n5. TF Transform Check:")
        self.check_tf_transforms()
        
        # 6. Check processes
        print(f"\n6. Process Check:")
        self.check_running_processes()
        
        # 7. Summary and recommendations
        print(f"\n" + "="*80)
        print("SUMMARY & RECOMMENDATIONS")
        print("="*80)
        
        if essential_found >= 3 and active_count >= 5:
            print("✓ System appears ready for Stretch AI mapping!")
            print("\nNext steps:")
            print("  1. Run: python /home/aoloo/code/stretch_ai/scripts/get_camera_transform.py")
            print("  2. Update camera_transform in fixed_nvblox_config.yaml")
            print("  3. Run: python /home/aoloo/code/stretch_ai/src/stretch/app/fixed_nvblox_mapping.py")
            return True
        else:
            print("✗ System not ready. Issues found:")
            if essential_found < 3:
                print(f"  - Missing essential topics ({essential_found}/4 found)")
                print("    Make sure nvblox and camera bridge are running")
            if active_count < 5:
                print(f"  - Low topic activity ({active_count} active topics)")
                print("    Check that sensors are publishing data")
            
            print(f"\nTroubleshooting:")
            print(f"  1. Start nvblox: ./scripts/run_nvblox.sh")
            print(f"  2. Start camera bridge: python3 ultimate_camera_bridge.py")
            print(f"  3. Check with: ros2 topic list | grep -E '(camera|nvblox|odom)'")
            return False
    
    def get_available_topics(self):
        """Get list of available ROS2 topics."""
        try:
            result = subprocess.run(['ros2', 'topic', 'list'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return result.stdout.strip().split('\n')
            else:
                print(f"  ✗ Failed to get topic list: {result.stderr}")
                return []
        except Exception as e:
            print(f"  ✗ Error getting topics: {e}")
            return []
    
    def check_tf_transforms(self):
        """Check TF transform availability."""
        try:
            # Check if TF is publishing
            result = subprocess.run(['ros2', 'topic', 'echo', '/tf', '--once'], 
                                  capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                print("  ✓ TF transforms are being published")
                
                # Try to check specific transforms
                essential_frames = ['odom', 'base_footprint', 'camera0_link']
                print("  Essential frames check:")
                
                for frame in essential_frames:
                    try:
                        tf_result = subprocess.run(
                            ['ros2', 'run', 'tf2_ros', 'tf2_echo', 'odom', frame],
                            capture_output=True, text=True, timeout=2)
                        if tf_result.returncode == 0:
                            print(f"    ✓ odom -> {frame}")
                        else:
                            print(f"    ✗ odom -> {frame} (transform not available)")
                    except:
                        print(f"    ⚠ odom -> {frame} (timeout checking)")
                        
            else:
                print("  ✗ TF transforms not available")
                
        except Exception as e:
            print(f"  ⚠ Could not check TF transforms: {e}")
    
    def check_running_processes(self):
        """Check for relevant running processes."""
        processes_to_check = [
            ('nvblox', 'nvblox mapping node'),
            ('realsense', 'RealSense camera driver'),
            ('camera_bridge', 'Camera bridge script'),
            ('isaac_ros', 'Isaac ROS container'),
        ]
        
        try:
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            process_list = result.stdout.lower()
            
            for process_name, description in processes_to_check:
                if process_name in process_list:
                    print(f"  ✓ {description} appears to be running")
                else:
                    print(f"  ⚠ {description} not detected in process list")
                    
        except Exception as e:
            print(f"  ⚠ Could not check processes: {e}")


def main():
    """Main diagnostic function."""
    
    if not HAS_ROS2:
        print("Error: ROS2 not available. Install ros2-humble or equivalent.")
        sys.exit(1)
    
    rclpy.init()
    
    try:
        checker = SystemChecker()
        success = checker.run_diagnostics()
        
        print(f"\n" + "="*80)
        if success:
            print("SYSTEM CHECK PASSED - Ready for fixed nvblox mapping!")
        else:
            print("SYSTEM CHECK FAILED - Fix issues above before proceeding")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\nDiagnostic interrupted by user")
    except Exception as e:
        print(f"\nDiagnostic failed with error: {e}")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()