#!/usr/bin/env python3
# get_camera_transform.py - Get the actual camera transform from your ROS2 system

import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
import numpy as np
from scipy.spatial.transform import Rotation as R
import time

class CameraTransformGetter(Node):
    """Get camera transform from TF for Stretch AI configuration."""
    
    def __init__(self):
        super().__init__('camera_transform_getter')
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        print("Camera Transform Getter initialized")
        print("Waiting for TF transforms...")
        
        # Wait for TF to populate
        time.sleep(3.0)
        
    def get_transform_matrix(self, parent_frame: str, child_frame: str):
        """Get 4x4 transformation matrix between two frames."""
        try:
            transform = self.tf_buffer.lookup_transform(
                parent_frame, child_frame, rclpy.time.Time())
            
            # Extract translation and rotation
            t = transform.transform.translation
            r = transform.transform.rotation
            
            # Convert quaternion to rotation matrix
            rot_matrix = R.from_quat([r.x, r.y, r.z, r.w]).as_matrix()
            
            # Create 4x4 transformation matrix
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rot_matrix
            transform_matrix[:3, 3] = [t.x, t.y, t.z]
            
            return transform_matrix, True
            
        except Exception as e:
            self.get_logger().error(f"Failed to lookup transform {parent_frame}->{child_frame}: {e}")
            return None, False
    
    def print_yaml_config(self, transform_matrix, parent_frame, child_frame):
        """Print the transform in YAML format for configuration."""
        print(f"\n# Camera transform from {parent_frame} to {child_frame}")
        print("camera_transform: [")
        for i in range(4):
            row_str = "  " + ", ".join([f"{transform_matrix[i,j]:8.4f}" for j in range(4)])
            if i < 3:
                row_str += ","
            print(row_str)
        print("]")
        
        # Also print as flattened list
        flat = transform_matrix.flatten()
        print(f"\n# Flattened version (16 elements):")
        print("camera_transform: [", end="")
        for i, val in enumerate(flat):
            print(f"{val:.4f}", end="")
            if i < len(flat) - 1:
                print(", ", end="")
        print("]")
    
    def run(self):
        """Main execution to get all relevant transforms."""
        
        print("\n" + "="*60)
        print("CAMERA TRANSFORM CALIBRATION FOR STRETCH AI")
        print("="*60)
        
        # Check available frames
        print("\nChecking for required coordinate frames...")
        
        required_frames = ["odom", "base_footprint", "camera0_link"]
        available_frames = []
        
        for frame in required_frames:
            try:
                # Try to get a transform to check if frame exists
                self.tf_buffer.lookup_transform("odom", frame, rclpy.time.Time())
                available_frames.append(frame)
                print(f"  ✓ {frame}")
            except:
                print(f"  ✗ {frame} (not available)")
        
        if len(available_frames) < len(required_frames):
            print(f"\nWarning: Not all required frames are available.")
            print(f"Make sure your nvblox system is running with:")
            print(f"  ros2 launch nvblox_examples_bringup realsense_example.launch.py")
        
        print(f"\nGetting coordinate transformations...")
        
        # Get transforms
        transforms_to_get = [
            ("odom", "base_footprint", "Global to robot base"),
            ("base_footprint", "camera0_link", "Robot base to camera"),
            ("odom", "camera0_link", "Global to camera (direct)")
        ]
        
        all_transforms = {}
        
        for parent, child, description in transforms_to_get:
            print(f"\n{description} ({parent} -> {child}):")
            
            transform_matrix, success = self.get_transform_matrix(parent, child)
            
            if success:
                all_transforms[(parent, child)] = transform_matrix
                
                # Print the transform details
                t = transform_matrix[:3, 3]
                R_mat = transform_matrix[:3, :3]
                euler = R.from_matrix(R_mat).as_euler('xyz', degrees=True)
                
                print(f"  Translation: [{t[0]:7.4f}, {t[1]:7.4f}, {t[2]:7.4f}] meters")
                print(f"  Rotation (XYZ Euler): [{euler[0]:6.1f}°, {euler[1]:6.1f}°, {euler[2]:6.1f}°]")
                
                # Print matrix
                print("  Transform Matrix:")
                for i in range(4):
                    row_str = "    [" + " ".join([f"{transform_matrix[i,j]:8.4f}" for j in range(4)]) + "]"
                    print(row_str)
            else:
                print(f"  ✗ Transform not available")
        
        # Generate configuration
        if ("base_footprint", "camera0_link") in all_transforms:
            print(f"\n" + "="*60)
            print("STRETCH AI CONFIGURATION")
            print("="*60)
            
            camera_transform = all_transforms[("base_footprint", "camera0_link")]
            self.print_yaml_config(camera_transform, "base_footprint", "camera0_link")
            
            print(f"\n# Copy the camera_transform above into your config file:")
            print(f"# /path/to/stretch_ai/src/stretch/config/fixed_nvblox_config.yaml")
            
        else:
            print(f"\nError: Could not get base_footprint -> camera0_link transform")
            print(f"This is required for Stretch AI mapping!")
        
        # Additional diagnostic info
        print(f"\n" + "="*60)
        print("DIAGNOSTIC INFORMATION")
        print("="*60)
        
        print(f"\nTo manually check transforms, use:")
        print(f"  ros2 run tf2_ros tf2_echo odom base_footprint")
        print(f"  ros2 run tf2_ros tf2_echo base_footprint camera0_link")
        print(f"  ros2 run tf2_ros tf2_echo odom camera0_link")
        
        print(f"\nTo visualize TF tree:")
        print(f"  ros2 run tf2_tools view_frames.py")
        print(f"  # This generates frames.pdf with the complete transform tree")
        
        print(f"\nTo check topic availability:")
        print(f"  ros2 topic list | grep -E '(camera|nvblox|odom|joint)'")
        
        return len(all_transforms) > 0


def main():
    """Main function."""
    rclpy.init()
    
    try:
        node = CameraTransformGetter()
        success = node.run()
        
        if success:
            print(f"\n✓ Camera transform calibration complete!")
        else:
            print(f"\n✗ Camera transform calibration failed!")
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()