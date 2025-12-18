#!/usr/bin/env python3
"""
UR5e Robot Client for Stretch AI

Integrates UR5e arm with Segway base into Stretch AI framework.
Uses ROS2 for communication with UR5e and existing ZMQ client for Segway base.
"""

import time
import numpy as np
from typing import Optional, Tuple, List
import threading

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient

from stretch.core.robot import AbstractRobotClient
from stretch.core.interfaces import Observations
from stretch.motion.robot import RobotModel
from stretch.agent.zmq_client import HomeRobotZmqClient


class UR5eKinematics(RobotModel):
    """
    UR5e kinematics model.

    Provides FK/IK and robot model interface for UR5e arm.
    """

    # UR5e joint names in standard order
    UR5E_JOINTS = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint"
    ]

    def __init__(self):
        """Initialize UR5e kinematics."""
        self.dof = 6
        self.joint_limits_lower = np.array([-2*np.pi, -2*np.pi, -np.pi, -2*np.pi, -2*np.pi, -2*np.pi])
        self.joint_limits_upper = np.array([2*np.pi, 2*np.pi, np.pi, 2*np.pi, 2*np.pi, 2*np.pi])

    def get_dof(self) -> int:
        """Get degrees of freedom."""
        return self.dof

    def set_config(self, q):
        """Set configuration (not used for UR5e)."""
        pass

    def get_footprint(self):
        """
        Get robot footprint.

        Since UR5e is mounted on Segway, use Segway footprint.
        """
        from stretch.utils.geometry import Footprint
        # Segway footprint: 0.25m radius
        return Footprint(
            width=0.5,
            length=0.5,
            width_offset=0.0,
            length_offset=0.0
        )


class UR5eROS2Node(Node):
    """ROS2 node for UR5e communication."""

    def __init__(self):
        super().__init__('ur5e_stretch_ai_client')

        # Joint state subscriber
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self._joint_state_callback,
            10
        )

        # Trajectory action client
        self.trajectory_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/scaled_pos_joint_traj_controller/follow_joint_trajectory'
        )

        # Current joint state
        self.current_joint_positions = None
        self.current_joint_velocities = None
        self.current_joint_efforts = None
        self.joint_names = None
        self._joint_state_lock = threading.Lock()

        # Gripper state (if using Robotiq or other gripper)
        self.gripper_position = 0.0  # 0 = open, 1 = closed

        self.get_logger().info("UR5e ROS2 node initialized")

    def _joint_state_callback(self, msg: JointState):
        """Callback for joint state updates."""
        with self._joint_state_lock:
            self.joint_names = list(msg.name)
            self.current_joint_positions = np.array(msg.position)
            self.current_joint_velocities = np.array(msg.velocity)
            self.current_joint_efforts = np.array(msg.effort)

    def get_joint_state(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Get current joint state.

        Returns:
            (positions, velocities, efforts) or None if no data
        """
        with self._joint_state_lock:
            if self.current_joint_positions is None:
                return None
            return (
                self.current_joint_positions.copy(),
                self.current_joint_velocities.copy() if self.current_joint_velocities is not None else None,
                self.current_joint_efforts.copy() if self.current_joint_efforts is not None else None
            )

    def get_ur5e_joint_positions(self) -> Optional[np.ndarray]:
        """
        Get UR5e joint positions in standard order.

        Returns:
            6D array [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]
        """
        with self._joint_state_lock:
            if self.joint_names is None or self.current_joint_positions is None:
                return None

            # Extract UR5e joints in correct order
            ur5e_positions = np.zeros(6)
            for i, joint_name in enumerate(UR5eKinematics.UR5E_JOINTS):
                if joint_name in self.joint_names:
                    idx = self.joint_names.index(joint_name)
                    ur5e_positions[i] = self.current_joint_positions[idx]

            return ur5e_positions

    def send_trajectory(self, positions: np.ndarray, duration: float = 2.0) -> bool:
        """
        Send trajectory to UR5e.

        Args:
            positions: 6D joint positions
            duration: Time to execute trajectory (seconds)

        Returns:
            Success status
        """
        if not self.trajectory_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().error("Trajectory action server not available")
            return False

        # Create trajectory message
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = UR5eKinematics.UR5E_JOINTS

        point = JointTrajectoryPoint()
        point.positions = positions.tolist()
        point.time_from_start.sec = int(duration)
        point.time_from_start.nanosec = int((duration - int(duration)) * 1e9)

        goal_msg.trajectory.points = [point]

        # Send goal
        future = self.trajectory_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)

        if not future.done():
            self.get_logger().error("Failed to send trajectory goal")
            return False

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Trajectory goal rejected")
            return False

        # Wait for result
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=duration + 2.0)

        if not result_future.done():
            self.get_logger().error("Trajectory execution timed out")
            return False

        result = result_future.result().result
        success = result.error_code == FollowJointTrajectory.Result.SUCCESSFUL

        return success


class UR5eSegwayRobotClient(AbstractRobotClient):
    """
    Combined UR5e + Segway robot client for Stretch AI.

    Integrates:
    - UR5e arm for manipulation via ROS2
    - Segway base for navigation via ZMQ
    """

    def __init__(
        self,
        robot_ip: str = "172.20.10.3",
        parameters = None,
        use_remote_computer: bool = True,
        enable_rerun_server: bool = False
    ):
        """
        Initialize UR5e + Segway robot client.

        Args:
            robot_ip: IP address for Segway base ZMQ connection
            parameters: Robot parameters
            use_remote_computer: Connect to remote robot
            enable_rerun_server: Enable rerun visualization
        """
        # Initialize ROS2 for UR5e
        if not rclpy.ok():
            rclpy.init()

        self.ur5e_node = UR5eROS2Node()

        # Start ROS2 spinner thread
        self.ros_thread = threading.Thread(target=self._spin_ros, daemon=True)
        self.ros_thread.start()

        # Initialize Segway base client via ZMQ
        self.base_client = HomeRobotZmqClient(
            robot_ip=robot_ip,
            parameters=parameters,
            use_remote_computer=use_remote_computer,
            enable_rerun_server=enable_rerun_server
        )

        # Robot model
        self._robot_model = UR5eKinematics()

        # State
        self._control_mode = "NAVIGATION"

        print("✅ UR5e + Segway robot client initialized")

    def _spin_ros(self):
        """Spin ROS2 node in background thread."""
        while rclpy.ok():
            rclpy.spin_once(self.ur5e_node, timeout_sec=0.01)

    # ========================================================================
    # Navigation methods (delegated to Segway base)
    # ========================================================================

    def navigate_to_goal(self, x: float, y: float, theta: float) -> bool:
        """Navigate to goal pose."""
        return self.base_client.navigate_to_goal(x, y, theta)

    def move_base_to(self, xyt, relative=False, blocking=False, verbose=False, timeout=None):
        """Move base to pose."""
        return self.base_client.move_base_to(xyt, relative, blocking, verbose, timeout)

    def execute_trajectory(self, trajectory: List[np.ndarray], **kwargs):
        """Execute base trajectory."""
        return self.base_client.execute_trajectory(trajectory, **kwargs)

    def get_base_pose(self) -> np.ndarray:
        """Get current base pose [x, y, theta]."""
        return self.base_client.get_base_pose()

    def at_goal(self) -> bool:
        """Check if at navigation goal."""
        return self.base_client.at_goal()

    # ========================================================================
    # UR5e Manipulation methods
    # ========================================================================

    def arm_to(self, joint_angles: np.ndarray, blocking: bool = True, timeout: float = 10.0) -> bool:
        """
        Move UR5e arm to joint configuration.

        Args:
            joint_angles: 6D joint angles [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]
            blocking: Wait for completion
            timeout: Timeout in seconds

        Returns:
            Success status
        """
        # Send trajectory to UR5e
        duration = 3.0  # Default trajectory duration
        success = self.ur5e_node.send_trajectory(joint_angles, duration=duration)

        if blocking:
            time.sleep(duration)

        return success

    def get_joint_positions(self, timeout: float = 5.0) -> Optional[np.ndarray]:
        """
        Get current UR5e joint positions.

        Returns:
            6D joint positions or None
        """
        return self.ur5e_node.get_ur5e_joint_positions()

    def get_joint_state(self, timeout: float = 5.0) -> Optional[Tuple]:
        """Get full joint state."""
        return self.ur5e_node.get_joint_state()

    def open_gripper(self, blocking: bool = True, timeout: float = 10.0, verbose: bool = False) -> bool:
        """
        Open gripper.

        TODO: Implement gripper control based on your gripper type
        (Robotiq, OnRobot, etc.)
        """
        print("⚠️  Gripper control not yet implemented")
        return True

    def close_gripper(self, loose: bool = False, blocking: bool = True, timeout: float = 10.0, verbose: bool = False) -> bool:
        """
        Close gripper.

        TODO: Implement gripper control
        """
        print("⚠️  Gripper control not yet implemented")
        return True

    # ========================================================================
    # Mode switching
    # ========================================================================

    def switch_to_navigation_mode(self):
        """Switch to navigation mode."""
        self._control_mode = "NAVIGATION"
        return self.base_client.switch_to_navigation_mode()

    def switch_to_manipulation_mode(self):
        """Switch to manipulation mode."""
        self._control_mode = "MANIPULATION"
        return self.base_client.switch_to_manipulation_mode()

    def in_manipulation_mode(self) -> bool:
        """Check if in manipulation mode."""
        return self._control_mode == "MANIPULATION"

    def in_navigation_mode(self) -> bool:
        """Check if in navigation mode."""
        return self._control_mode == "NAVIGATION"

    # ========================================================================
    # Camera/Head (use Segway's camera)
    # ========================================================================

    def get_observation(self) -> Observations:
        """Get current observation from cameras."""
        return self.base_client.get_observation()

    def head_to(self, head_pan, head_tilt, blocking=False, timeout=10.0):
        """Control head (not applicable for UR5e)."""
        return True

    # ========================================================================
    # Robot model
    # ========================================================================

    def get_robot_model(self) -> RobotModel:
        """Get robot kinematics model."""
        return self._robot_model

    def get_footprint(self):
        """Get robot footprint."""
        return self._robot_model.get_footprint()

    # ========================================================================
    # Map operations (delegated to base)
    # ========================================================================

    def save_map(self, filename: str):
        """Save map."""
        return self.base_client.save_map(filename)

    def load_map(self, filename: str):
        """Load map."""
        return self.base_client.load_map(filename)

    def reset(self):
        """Reset robot state."""
        return self.base_client.reset()

    def get_pose_graph(self) -> np.ndarray:
        """Get pose graph."""
        return self.base_client.get_pose_graph()

    def move_to_nav_posture(self):
        """Move to navigation posture."""
        # TODO: Move arm to safe navigation configuration
        return True

    def move_to_manip_posture(self):
        """Move to manipulation posture."""
        # TODO: Move arm to ready-to-grasp configuration
        return True

    def __del__(self):
        """Cleanup on destruction."""
        if hasattr(self, 'ur5e_node'):
            self.ur5e_node.destroy_node()
        rclpy.shutdown()
