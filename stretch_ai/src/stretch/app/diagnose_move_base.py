#!/usr/bin/env python3
"""
Diagnostic tool to monitor move_base behavior and identify why robot only rotates.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path, OccupancyGrid
from actionlib_msgs.msg import GoalStatusArray
import numpy as np
from datetime import datetime


class MoveBaseDiagnostic(Node):
    def __init__(self):
        super().__init__('move_base_diagnostic')

        # Subscribers
        self.status_sub = self.create_subscription(
            GoalStatusArray, '/move_base/status', self.status_callback, 10
        )
        self.global_plan_sub = self.create_subscription(
            Path, '/move_base/NavfnROS/plan', self.global_plan_callback, 10
        )
        self.local_plan_sub = self.create_subscription(
            Path, '/move_base/DWAPlannerROS/local_plan', self.local_plan_callback, 10
        )
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10
        )
        self.local_costmap_sub = self.create_subscription(
            OccupancyGrid, '/move_base/local_costmap/costmap', self.local_costmap_callback, 10
        )
        self.global_costmap_sub = self.create_subscription(
            OccupancyGrid, '/move_base/global_costmap/costmap', self.global_costmap_callback, 10
        )

        # State tracking
        self.last_status = None
        self.global_plan_length = 0
        self.local_plan_length = 0
        self.cmd_vel_history = []
        self.rotation_only_count = 0
        self.translation_count = 0
        self.local_costmap_obstacles = 0
        self.global_costmap_obstacles = 0

        self.get_logger().info("Move_base diagnostic started. Monitoring all topics...")

    def status_callback(self, msg):
        """Monitor move_base goal status"""
        if not msg.status_list:
            return

        for status in msg.status_list:
            status_text = self.get_status_text(status.status)
            if status.status != self.last_status:
                self.get_logger().info(f"[STATUS] {status_text}: {status.text}")
                self.last_status = status.status

                # Check for failures
                if status.status == 4:  # ABORTED
                    self.get_logger().error(f"❌ Goal ABORTED: {status.text}")
                    self.print_summary()
                elif status.status == 5:  # REJECTED
                    self.get_logger().error(f"❌ Goal REJECTED: {status.text}")
                    self.print_summary()

    def global_plan_callback(self, msg):
        """Monitor global path from NavfnROS"""
        self.global_plan_length = len(msg.poses)
        if self.global_plan_length == 0:
            self.get_logger().error("❌ No global path generated!")
        else:
            self.get_logger().info(f"✅ Global path: {self.global_plan_length} waypoints")

    def local_plan_callback(self, msg):
        """Monitor local path from DWA planner"""
        self.local_plan_length = len(msg.poses)
        if self.local_plan_length == 0:
            self.get_logger().warn("⚠️  No local path generated!")
        else:
            self.get_logger().info(f"✅ Local path: {self.local_plan_length} waypoints")

    def cmd_vel_callback(self, msg):
        """Monitor velocity commands to see if robot is moving"""
        linear = abs(msg.linear.x)
        angular = abs(msg.angular.z)

        self.cmd_vel_history.append((linear, angular))
        if len(self.cmd_vel_history) > 50:
            self.cmd_vel_history.pop(0)

        # Detect rotation-only behavior
        if angular > 0.01 and linear < 0.01:
            self.rotation_only_count += 1
            if self.rotation_only_count % 20 == 0:
                self.get_logger().warn(
                    f"⚠️  Rotation only! angular={angular:.3f}, linear={linear:.3f}"
                )
        elif linear > 0.01:
            self.translation_count += 1
            self.get_logger().info(f"✅ Moving forward! linear={linear:.3f}")

    def local_costmap_callback(self, msg):
        """Analyze local costmap for obstacles"""
        data = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        # Count cells with high cost (>50 = obstacle)
        obstacles = np.sum(data > 50)
        self.local_costmap_obstacles = obstacles

        if obstacles > (msg.info.height * msg.info.width * 0.8):
            self.get_logger().error(
                f"❌ Local costmap heavily occupied: {obstacles} obstacle cells!"
            )

    def global_costmap_callback(self, msg):
        """Analyze global costmap for obstacles"""
        data = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        obstacles = np.sum(data > 50)
        self.global_costmap_obstacles = obstacles

    def get_status_text(self, status):
        """Convert status code to readable text"""
        status_map = {
            0: "PENDING",
            1: "ACTIVE",
            2: "PREEMPTED",
            3: "SUCCEEDED",
            4: "ABORTED",
            5: "REJECTED",
            6: "PREEMPTING",
            7: "RECALLING",
            8: "RECALLED",
            9: "LOST"
        }
        return status_map.get(status, f"UNKNOWN({status})")

    def print_summary(self):
        """Print diagnostic summary"""
        self.get_logger().info("\n" + "="*60)
        self.get_logger().info("MOVE_BASE DIAGNOSTIC SUMMARY")
        self.get_logger().info("="*60)
        self.get_logger().info(f"Global plan length: {self.global_plan_length} waypoints")
        self.get_logger().info(f"Local plan length: {self.local_plan_length} waypoints")
        self.get_logger().info(f"Rotation-only commands: {self.rotation_only_count}")
        self.get_logger().info(f"Translation commands: {self.translation_count}")
        self.get_logger().info(f"Local costmap obstacles: {self.local_costmap_obstacles} cells")
        self.get_logger().info(f"Global costmap obstacles: {self.global_costmap_obstacles} cells")

        # Diagnose issue
        self.get_logger().info("\n🔍 DIAGNOSIS:")
        if self.global_plan_length == 0:
            self.get_logger().error("❌ No global path - goal may be in obstacle or unreachable")
        elif self.local_plan_length == 0:
            self.get_logger().error("❌ No local path - local costmap may be blocking movement")
        elif self.rotation_only_count > 100 and self.translation_count < 10:
            self.get_logger().error("❌ Robot stuck rotating - likely costmap inflation too aggressive")

        self.get_logger().info("="*60 + "\n")


def main():
    rclpy.init()
    node = MoveBaseDiagnostic()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.print_summary()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()