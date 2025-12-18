#!/usr/bin/env python3
"""
Direct semantic 3D goal to 2D navigation using robot's map.
Uses /map from robot, /amcl_pose for localization, and transforms 3D voxel goals to 2D.
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
import numpy as np
from typing import Optional, Tuple
import time


class SemanticTo2DNavigator(Node):
    def __init__(self):
        super().__init__('semantic_to_2d_navigator')

        # State
        self.occupancy_map: Optional[OccupancyGrid] = None
        self.robot_pose: Optional[PoseStamped] = None
        self.map_ready = False

        # Subscribers
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

        self.amcl_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.amcl_callback,
            10
        )

        # Publisher for navigation goals
        self.goal_pub = self.create_publisher(
            PoseStamped,
            '/move_base_simple/goal',
            10
        )

        self.get_logger().info("Semantic to 2D Navigator initialized")
        self.get_logger().info("Waiting for /map and /amcl_pose...")

    def map_callback(self, msg: OccupancyGrid):
        """Receive the robot's 2D occupancy grid map"""
        if not self.map_ready:
            self.get_logger().info(f"✅ Map received: {msg.info.width}x{msg.info.height}, resolution={msg.info.resolution}m/px")
            self.get_logger().info(f"   Map origin: ({msg.info.origin.position.x:.2f}, {msg.info.origin.position.y:.2f})")
            self.map_ready = True

        self.occupancy_map = msg

    def amcl_callback(self, msg: PoseWithCovarianceStamped):
        """Receive robot's localized pose from AMCL"""
        # Convert to PoseStamped for easier handling
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose = msg.pose.pose
        self.robot_pose = pose

    def world_to_map_coords(self, world_x: float, world_y: float) -> Tuple[int, int]:
        """
        Convert world coordinates (meters) to map pixel coordinates.

        Args:
            world_x, world_y: Position in world/map frame (meters)

        Returns:
            (map_x, map_y): Pixel coordinates in occupancy grid
        """
        if self.occupancy_map is None:
            raise ValueError("Map not available")

        info = self.occupancy_map.info

        # Transform world -> map pixel
        # world = origin + pixel * resolution
        # pixel = (world - origin) / resolution
        pixel_x = int((world_x - info.origin.position.x) / info.resolution)
        pixel_y = int((world_y - info.origin.position.y) / info.resolution)

        return pixel_x, pixel_y

    def map_to_world_coords(self, map_x: int, map_y: int) -> Tuple[float, float]:
        """
        Convert map pixel coordinates to world coordinates (meters).

        Args:
            map_x, map_y: Pixel coordinates in occupancy grid

        Returns:
            (world_x, world_y): Position in world/map frame (meters)
        """
        if self.occupancy_map is None:
            raise ValueError("Map not available")

        info = self.occupancy_map.info

        # Transform map pixel -> world
        world_x = info.origin.position.x + (map_x + 0.5) * info.resolution
        world_y = info.origin.position.y + (map_y + 0.5) * info.resolution

        return world_x, world_y

    def is_valid_goal(self, map_x: int, map_y: int, check_radius: int = 5) -> bool:
        """
        Check if a map position is valid for navigation (in free space).

        Args:
            map_x, map_y: Pixel coordinates to check
            check_radius: Radius in pixels to check around the point

        Returns:
            bool: True if position is valid (free space)
        """
        if self.occupancy_map is None:
            return False

        info = self.occupancy_map.info
        data = np.array(self.occupancy_map.data).reshape((info.height, info.width))

        # Check bounds
        if not (0 <= map_x < info.width and 0 <= map_y < info.height):
            self.get_logger().warn(f"Goal ({map_x}, {map_y}) out of map bounds")
            return False

        # Check if center is free (0-50 = free, >50 = obstacle, -1 = unknown)
        center_cost = data[map_y, map_x]
        if center_cost > 50:
            self.get_logger().warn(f"Goal at ({map_x}, {map_y}) is in obstacle (cost={center_cost})")
            return False

        if center_cost < 0:
            self.get_logger().warn(f"Goal at ({map_x}, {map_y}) is in unknown space")
            return False

        # Check surrounding area for obstacles
        y_min = max(0, map_y - check_radius)
        y_max = min(info.height, map_y + check_radius + 1)
        x_min = max(0, map_x - check_radius)
        x_max = min(info.width, map_x + check_radius + 1)

        region = data[y_min:y_max, x_min:x_max]
        obstacles = np.sum(region > 50)

        if obstacles > (region.size * 0.3):  # More than 30% obstacles
            self.get_logger().warn(f"Goal has too many nearby obstacles ({obstacles} cells)")
            return False

        self.get_logger().info(f"✅ Goal at ({map_x}, {map_y}) is valid (cost={center_cost})")
        return True

    def find_nearest_free_space(self, map_x: int, map_y: int, max_search_radius: int = 20) -> Optional[Tuple[int, int]]:
        """
        Find nearest free space if the goal is in an obstacle.

        Args:
            map_x, map_y: Starting pixel coordinates
            max_search_radius: Maximum search radius in pixels

        Returns:
            (map_x, map_y): Nearest free space coordinates, or None if not found
        """
        if self.occupancy_map is None:
            return None

        info = self.occupancy_map.info
        data = np.array(self.occupancy_map.data).reshape((info.height, info.width))

        # Spiral search outward from goal
        for radius in range(1, max_search_radius + 1):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    # Only check perimeter of current radius
                    if abs(dx) != radius and abs(dy) != radius:
                        continue

                    check_x = map_x + dx
                    check_y = map_y + dy

                    if self.is_valid_goal(check_x, check_y, check_radius=3):
                        self.get_logger().info(f"✅ Found free space at ({check_x}, {check_y}), {radius} pixels away")
                        return check_x, check_y

        self.get_logger().error(f"❌ No free space found within {max_search_radius} pixels")
        return None

    def navigate_to_3d_goal(self, voxel_x: float, voxel_y: float, voxel_z: float = 0.0):
        """
        Navigate to a 3D voxel goal by transforming to 2D map coordinates.

        Args:
            voxel_x, voxel_y, voxel_z: 3D position from voxel map
        """
        if not self.map_ready or self.occupancy_map is None:
            self.get_logger().error("❌ Map not ready")
            return False

        if self.robot_pose is None:
            self.get_logger().error("❌ Robot pose not available from AMCL")
            return False

        # Step 1: Project 3D voxel goal to 2D (just use x, y)
        self.get_logger().info(f"\n{'='*60}")
        self.get_logger().info(f"📍 3D Goal (voxel frame): ({voxel_x:.2f}, {voxel_y:.2f}, {voxel_z:.2f})")

        # Step 2: Transform to map coordinates (assuming voxel frame == map frame for now)
        # TODO: Add proper coordinate transformation if voxel map has different origin
        world_x, world_y = voxel_x, voxel_y
        self.get_logger().info(f"🗺️  World coordinates: ({world_x:.2f}, {world_y:.2f})")

        # Step 3: Convert to map pixel coordinates
        map_x, map_y = self.world_to_map_coords(world_x, world_y)
        self.get_logger().info(f"🔢 Map pixel coords: ({map_x}, {map_y})")

        # Step 4: Validate goal is in free space
        if not self.is_valid_goal(map_x, map_y):
            self.get_logger().warn("⚠️  Goal in obstacle, searching for nearest free space...")
            result = self.find_nearest_free_space(map_x, map_y)
            if result is None:
                self.get_logger().error("❌ Cannot find valid navigation goal")
                return False
            map_x, map_y = result
            world_x, world_y = self.map_to_world_coords(map_x, map_y)
            self.get_logger().info(f"✅ Using adjusted goal: ({world_x:.2f}, {world_y:.2f})")

        # Step 5: Calculate orientation (face toward goal from current position)
        robot_x = self.robot_pose.pose.position.x
        robot_y = self.robot_pose.pose.position.y
        goal_theta = np.arctan2(world_y - robot_y, world_x - robot_x)

        distance = np.sqrt((world_x - robot_x)**2 + (world_y - robot_y)**2)
        self.get_logger().info(f"📏 Distance to goal: {distance:.2f}m")
        self.get_logger().info(f"🧭 Goal orientation: {np.degrees(goal_theta):.1f}°")

        # Step 6: Send navigation goal
        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = world_x
        goal.pose.position.y = world_y
        goal.pose.position.z = 0.0

        # Convert theta to quaternion
        goal.pose.orientation.w = np.cos(goal_theta / 2.0)
        goal.pose.orientation.x = 0.0
        goal.pose.orientation.y = 0.0
        goal.pose.orientation.z = np.sin(goal_theta / 2.0)

        self.goal_pub.publish(goal)
        self.get_logger().info(f"🚀 Navigation goal published to /move_base_simple/goal")
        self.get_logger().info(f"{'='*60}\n")

        return True

    def get_current_position(self) -> Optional[Tuple[float, float, float]]:
        """Get current robot position (x, y, theta) from AMCL"""
        if self.robot_pose is None:
            return None

        x = self.robot_pose.pose.position.x
        y = self.robot_pose.pose.position.y

        # Extract yaw from quaternion
        quat = self.robot_pose.pose.orientation
        siny_cosp = 2 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1 - 2 * (quat.y * quat.y + quat.z * quat.z)
        theta = np.arctan2(siny_cosp, cosy_cosp)

        return x, y, theta


def main():
    rclpy.init()
    navigator = SemanticTo2DNavigator()

    # Wait for map and localization
    print("Waiting for map and localization...")
    while not navigator.map_ready or navigator.robot_pose is None:
        rclpy.spin_once(navigator, timeout_sec=0.1)
        time.sleep(0.1)

    print("\n✅ Map and localization ready!")

    # Get current position
    pos = navigator.get_current_position()
    if pos:
        print(f"📍 Robot at: ({pos[0]:.2f}, {pos[1]:.2f}), θ={np.degrees(pos[2]):.1f}°")

    # Example: Navigate to a 3D semantic goal
    # Replace these with actual 3D voxel coordinates from your semantic map
    print("\n🔍 Enter 3D voxel goal coordinates:")
    try:
        voxel_x = float(input("  X (meters): "))
        voxel_y = float(input("  Y (meters): "))
        voxel_z = float(input("  Z (meters, optional): ") or "0.0")

        navigator.navigate_to_3d_goal(voxel_x, voxel_y, voxel_z)

        # Keep spinning to maintain subscriptions
        print("\nNavigating... Press Ctrl+C to exit")
        rclpy.spin(navigator)

    except KeyboardInterrupt:
        print("\nShutdown requested")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        navigator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()