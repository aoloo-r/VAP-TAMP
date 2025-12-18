# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# Segway Robot Interface for Stretch AI
# This provides a robot interface specifically for Segway robots

import numpy as np
from typing import Optional, Dict, Any
from stretch.core.interfaces import Observations
from stretch.core.robot import AbstractRobotClient
from stretch.motion import Footprint, RobotModel
from stretch.utils.logger import Logger

logger = Logger(__name__)


class SegwayRobotModel:
    """Robot model specifically for Segway robots."""

    def __init__(self,
                 width: float = 0.25,
                 length: float = 0.30,
                 height: float = 0.1):
        """Initialize Segway robot model.

        Args:
            width: Robot width in meters (default 25cm)
            length: Robot length in meters (default 30cm)
            height: Robot height in meters (default 10cm for base)
        """
        self.width = width
        self.length = length
        self.height = height

        # Create footprint for collision detection
        self._footprint = Footprint(
            width=width,
            length=length,
            width_offset=0.0,
            length_offset=0.0
        )

        logger.info(f"Segway robot model: {width}m x {length}m x {height}m")

    def get_footprint(self) -> Footprint:
        """Get robot footprint for collision detection."""
        return self._footprint


class SegwayRobotClient(AbstractRobotClient):
    """
    Robot interface for Segway robots using ROS bridge.

    This provides a simplified interface that works with Segway's
    navigation capabilities while maintaining compatibility with
    Stretch AI's planning and mapping systems.
    """

    def __init__(self,
                 robot_width: float = 0.25,
                 robot_length: float = 0.30,
                 robot_height: float = 0.1):
        """Initialize Segway robot client.

        Args:
            robot_width: Segway robot width in meters
            robot_length: Segway robot length in meters
            robot_height: Segway robot height in meters
        """
        super().__init__()

        # Create robot model
        self.robot_model = SegwayRobotModel(robot_width, robot_length, robot_height)

        # Robot state
        self._base_pose = np.array([0.0, 0.0, 0.0])  # [x, y, theta]
        self._last_observation = None

        logger.info("Segway robot client initialized")

    def get_robot_model(self) -> SegwayRobotModel:
        """Get the Segway robot model."""
        return self.robot_model

    def get_base_pose(self) -> np.ndarray:
        """Get current robot base pose [x, y, theta]."""
        return self._base_pose.copy()

    def set_base_pose(self, pose: np.ndarray):
        """Set robot base pose (for simulation/testing)."""
        self._base_pose = pose.copy()
        logger.debug(f"Set base pose to {pose}")

    def move_base_to(self, xyt: np.ndarray, blocking: bool = True, **kwargs) -> bool:
        """Move robot base to target pose.

        Args:
            xyt: Target pose [x, y, theta]
            blocking: Whether to wait for completion
            **kwargs: Additional parameters

        Returns:
            bool: True if successful
        """
        logger.info(f"Moving base to {xyt} (blocking={blocking})")

        # For now, just update internal state
        # In real implementation, this would send commands to robot
        self._base_pose = np.array(xyt)

        return True

    def navigate_to_semantic_instance(self, target_instance, **kwargs) -> bool:
        """Navigate to a semantic instance (implemented by subclasses)."""
        logger.warning("navigate_to_semantic_instance not implemented for base SegwayRobotClient")
        return False

    def get_observation(self) -> Optional[Observations]:
        """Get current robot observations."""
        return self._last_observation

    def head_to(self, pan: float, tilt: float, blocking: bool = True, **kwargs) -> bool:
        """Move robot head (if available)."""
        logger.info(f"Head movement requested: pan={pan}, tilt={tilt}")
        # Segway might not have head - return success anyway
        return True

    def arm_to(self, joints: Dict[str, float], blocking: bool = True, **kwargs) -> bool:
        """Move robot arm (if available)."""
        logger.info(f"Arm movement requested: {joints}")
        # Segway might not have arm - return success anyway
        return True

    def stop(self):
        """Stop all robot motion."""
        logger.info("Stopping robot")
        # Implementation would send stop commands to robot
        pass

    def switch_to_navigation_mode(self):
        """Switch robot to navigation mode."""
        logger.info("Switching to navigation mode")
        return True

    def switch_to_manipulation_mode(self):
        """Switch robot to manipulation mode."""
        logger.info("Switching to manipulation mode")
        return True

    def move_to_nav_posture(self):
        """Move robot to navigation posture."""
        logger.info("Moving to navigation posture")
        return True