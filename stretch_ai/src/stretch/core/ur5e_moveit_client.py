#!/usr/bin/env python3
"""
UR5e MoveIt Client for Stretch AI

This client communicates with the UR5eController running on EOS ONE
via ROS bridge. It provides a simple interface for pick and place operations.
"""

import time
import numpy as np
from typing import Optional, Dict, Any, List
import threading

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import Pose
from std_srvs.srv import Trigger


class UR5eMoveItClient(Node):
    """
    ROS2 client for UR5e MoveIt controller.

    Communicates with UR5eController.py running on the robot via ROS bridge.
    """

    # Named joint configurations (from UR5eController.py)
    INITIAL_JOINT = [0.20, -2.19, 2.04, -1.42, -1.57, 0.19]
    JOINT_FOR_SEARCH = [0, -0.9, 0.5, -1.57, -1.57, 0]
    INITIAL_JOINT_FOR_PICKUP = [0, -0.9, 0.5, -1.2, -1.57, 0]

    # Configuration constants
    Z_OFFSET_FROM_TABLE = 1.1
    Z_OFFSET_FROM_OBJECT = 0.97
    Z_OFFSET = 0.03

    def __init__(self):
        super().__init__('ur5e_moveit_client')

        # Publishers for custom commands to UR5eController
        # Note: These topics should be set up in UR5eController to listen for commands
        self.pickup_cmd_pub = self.create_publisher(
            Pose,
            '/ur5e/pickup_command',
            10
        )

        self.place_cmd_pub = self.create_publisher(
            Pose,
            '/ur5e/place_command',
            10
        )

        # Service clients for gripper control
        self.gripper_open_client = self.create_client(Trigger, '/ur5e/gripper/open')
        self.gripper_close_client = self.create_client(Trigger, '/ur5e/gripper/close')

        # Service client for moving to initial pose
        self.move_to_init_client = self.create_client(Trigger, '/ur5e/move_to_init')

        # Track operation status
        self._operation_complete = threading.Event()
        self._operation_success = False

        self.get_logger().info("UR5e MoveIt Client initialized")

    def wait_for_services(self, timeout: float = 5.0) -> bool:
        """Wait for all services to be available."""
        services = [
            self.gripper_open_client,
            self.gripper_close_client,
            self.move_to_init_client
        ]

        all_ready = True
        for service in services:
            if not service.wait_for_service(timeout_sec=timeout):
                self.get_logger().warning(f"Service {service.srv_name} not available")
                all_ready = False

        return all_ready

    def open_gripper(self, blocking: bool = True, timeout: float = 5.0) -> bool:
        """Open the Robotiq gripper."""
        if not self.gripper_open_client.service_is_ready():
            self.get_logger().error("Gripper open service not available")
            return False

        request = Trigger.Request()
        future = self.gripper_open_client.call_async(request)

        if blocking:
            rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
            if future.done():
                response = future.result()
                return response.success
            return False

        return True

    def close_gripper(self, blocking: bool = True, timeout: float = 5.0) -> bool:
        """Close the Robotiq gripper."""
        if not self.gripper_close_client.service_is_ready():
            self.get_logger().error("Gripper close service not available")
            return False

        request = Trigger.Request()
        future = self.gripper_close_client.call_async(request)

        if blocking:
            rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
            if future.done():
                response = future.result()
                return response.success
            return False

        return True

    def move_to_init(self, blocking: bool = True, timeout: float = 10.0) -> bool:
        """Move arm to initial pose."""
        if not self.move_to_init_client.service_is_ready():
            self.get_logger().error("Move to init service not available")
            return False

        request = Trigger.Request()
        future = self.move_to_init_client.call_async(request)

        if blocking:
            rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
            if future.done():
                response = future.result()
                return response.success
            return False

        return True

    def pickup_object(
        self,
        object_info: Dict[str, Any],
        blocking: bool = True,
        timeout: float = 30.0
    ) -> bool:
        """
        Execute pickup operation.

        Args:
            object_info: Dictionary with:
                - offset: [x, y, z] offset for grasping
            blocking: Wait for completion
            timeout: Timeout in seconds

        Returns:
            Success status
        """
        # The UR5eController will handle the full pickup sequence
        # We just need to send the command with object info

        # For now, we'll use a simple message-based approach
        # In production, you'd want to set up a proper action interface

        self.get_logger().info(f"Sending pickup command with offset {object_info['offset']}")

        # This is a placeholder - you need to implement the actual
        # message passing to UR5eController
        # Option 1: Use ROS services
        # Option 2: Use ROS actions
        # Option 3: Use topics with acknowledgment

        return True

    def place_object(
        self,
        object_info: Dict[str, Any],
        blocking: bool = True,
        timeout: float = 30.0
    ) -> bool:
        """
        Execute place operation.

        Args:
            object_info: Dictionary with:
                - target: pose dictionary with x, y, z, ox, oy, oz, ow
                - offset: [x, y, z] offset for placing
            blocking: Wait for completion
            timeout: Timeout in seconds

        Returns:
            Success status
        """
        self.get_logger().info(f"Sending place command to {object_info['target']}")

        # Similar to pickup, this needs proper implementation
        # based on your UR5eController setup

        return True


class UR5eMoveItClientWrapper:
    """
    Wrapper for UR5eMoveItClient that manages ROS2 initialization
    and provides a simpler interface for the robot agent.
    """

    def __init__(self):
        """Initialize the UR5e MoveIt client."""
        if not rclpy.ok():
            rclpy.init()

        self.node = UR5eMoveItClient()

        # Start ROS2 spinner thread
        self.ros_thread = threading.Thread(target=self._spin_ros, daemon=True)
        self.ros_thread.start()

        # Wait for services
        print("Waiting for UR5e services...")
        if self.node.wait_for_services(timeout=5.0):
            print("✅ UR5e services are ready")
        else:
            print("⚠️  Some UR5e services are not available")

    def _spin_ros(self):
        """Spin ROS2 node in background thread."""
        while rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.01)

    def open_gripper(self, blocking: bool = True, timeout: float = 5.0) -> bool:
        """Open the gripper."""
        return self.node.open_gripper(blocking=blocking, timeout=timeout)

    def close_gripper(self, blocking: bool = True, timeout: float = 5.0) -> bool:
        """Close the gripper."""
        return self.node.close_gripper(blocking=blocking, timeout=timeout)

    def move_to_init(self, blocking: bool = True, timeout: float = 10.0) -> bool:
        """Move to initial pose."""
        return self.node.move_to_init(blocking=blocking, timeout=timeout)

    def pickup_object(
        self,
        object_info: Dict[str, Any],
        blocking: bool = True,
        timeout: float = 30.0
    ) -> bool:
        """Execute pickup."""
        return self.node.pickup_object(object_info, blocking=blocking, timeout=timeout)

    def place_object(
        self,
        object_info: Dict[str, Any],
        blocking: bool = True,
        timeout: float = 30.0
    ) -> bool:
        """Execute place."""
        return self.node.place_object(object_info, blocking=blocking, timeout=timeout)

    def __del__(self):
        """Cleanup on destruction."""
        if hasattr(self, 'node'):
            self.node.destroy_node()
