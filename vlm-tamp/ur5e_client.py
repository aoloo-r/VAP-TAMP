#!/usr/bin/env python3
"""
UR5e ROS2 Client for VLM-TAMP

This client runs on the desktop and communicates with UR5eController.py
running on the robot via rosbridge.

The UR5eController expects commands in a simple topic-based interface.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import time
import threading


class UR5eClient(Node):
    """
    ROS2 client that publishes commands to UR5eController via rosbridge.

    The UR5eController on the robot subscribes to /ur5e/command topic
    and executes pickup/place actions.
    """

    def __init__(self):
        super().__init__('ur5e_client')

        # Publisher for UR5e commands
        self.command_pub = self.create_publisher(
            String,
            '/ur5e/command',
            10
        )

        # Subscriber for UR5e status/feedback
        self.status_sub = self.create_subscription(
            String,
            '/ur5e/status',
            self._status_callback,
            10
        )

        # Status tracking
        self.current_status = "idle"
        self.command_success = False
        self.command_complete = threading.Event()

        self.get_logger().info("UR5e Client initialized")

    def _status_callback(self, msg):
        """Handle status updates from UR5eController."""
        status_data = json.loads(msg.data)
        self.current_status = status_data.get("status", "unknown")

        if self.current_status in ["success", "failed"]:
            self.command_success = (self.current_status == "success")
            self.command_complete.set()

        self.get_logger().info(f"UR5e status: {self.current_status}")

    def pickup(self, object_info, timeout=60.0):
        """
        Execute pickup action.

        Args:
            object_info: Dict with keys:
                - object_name: str
                - offset: [x, y, z] list
            timeout: Max time to wait (seconds)

        Returns:
            bool: Success status
        """
        self.get_logger().info(f"Sending pickup command: {object_info}")

        command = {
            "action": "pickup",
            "object_info": object_info
        }

        # Reset status
        self.command_complete.clear()
        self.command_success = False

        # Publish command
        msg = String()
        msg.data = json.dumps(command)
        self.command_pub.publish(msg)

        # Wait for completion
        success = self.command_complete.wait(timeout=timeout)

        if not success:
            self.get_logger().error(f"Pickup timed out after {timeout}s")
            return False

        return self.command_success

    def place(self, object_info, timeout=60.0):
        """
        Execute place action.

        Args:
            object_info: Dict with keys:
                - object_name: str
                - target: dict with x, y, z, ox, oy, oz, ow
                - offset: [x, y, z] list
            timeout: Max time to wait (seconds)

        Returns:
            bool: Success status
        """
        self.get_logger().info(f"Sending place command: {object_info}")

        command = {
            "action": "place",
            "object_info": object_info
        }

        # Reset status
        self.command_complete.clear()
        self.command_success = False

        # Publish command
        msg = String()
        msg.data = json.dumps(command)
        self.command_pub.publish(msg)

        # Wait for completion
        success = self.command_complete.wait(timeout=timeout)

        if not success:
            self.get_logger().error(f"Place timed out after {timeout}s")
            return False

        return self.command_success

    def move_to_init(self, timeout=15.0):
        """
        Move arm to initial pose.

        Args:
            timeout: Max time to wait (seconds)

        Returns:
            bool: Success status
        """
        self.get_logger().info("Sending move_to_init command")

        command = {
            "action": "move_to_init"
        }

        # Reset status
        self.command_complete.clear()
        self.command_success = False

        # Publish command
        msg = String()
        msg.data = json.dumps(command)
        self.command_pub.publish(msg)

        # Wait for completion
        success = self.command_complete.wait(timeout=timeout)

        if not success:
            self.get_logger().error(f"Move to init timed out after {timeout}s")
            return False

        return self.command_success


class UR5eClientWrapper:
    """
    Wrapper that manages ROS2 initialization and provides simple interface.

    Usage:
        client = UR5eClientWrapper()
        success = client.pickup({"object_name": "bottle", "offset": [0.065, 0.06, 0.08]})
    """

    def __init__(self):
        # Initialize ROS2 if needed
        if not rclpy.ok():
            rclpy.init()

        # Create node
        self.node = UR5eClient()

        # Start spinner thread
        self.spin_thread = threading.Thread(target=self._spin, daemon=True)
        self.spin_thread.start()

        print("✅ UR5e Client ready")

    def _spin(self):
        """Spin ROS2 node in background."""
        while rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.01)

    def pickup(self, object_info, timeout=60.0):
        """Execute pickup."""
        return self.node.pickup(object_info, timeout)

    def place(self, object_info, timeout=60.0):
        """Execute place."""
        return self.node.place(object_info, timeout)

    def move_to_init(self, timeout=15.0):
        """Move to initial pose."""
        return self.node.move_to_init(timeout)

    def __del__(self):
        """Cleanup."""
        if hasattr(self, 'node'):
            self.node.destroy_node()


# Simple test
if __name__ == '__main__':
    client = UR5eClientWrapper()

    print("\n🧪 Testing UR5e client...")
    print("Make sure:")
    print("1. UR5eController.py is running on the robot")
    print("2. rosbridge is running")
    print("3. The robot arm is in a safe position")

    input("\nPress Enter to test move_to_init...")
    success = client.move_to_init()
    print(f"Move to init: {'✅ Success' if success else '❌ Failed'}")

    print("\n✅ Test complete")
