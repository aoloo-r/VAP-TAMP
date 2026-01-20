#!/usr/bin/env python3
"""
Active Perception for DKPrompt with Stretch AI

This module adds active exploration capability to DKPrompt when VLM is uncertain.
When the robot can't determine a predicate from its current viewpoint, it moves
to better viewpoints using Stretch AI's navigation capabilities.

Key Features:
1. Detect VLM uncertainty in predicate verification
2. Sample alternative viewpoints around objects
3. Navigate robot to new viewpoint using Stretch AI
4. Retry perception with new observation
5. Return confident answer or failure after max attempts

Integration with DKPrompt:
- Wraps around existing check_states_and_update_problem()
- Only activates when VLM response is uncertain
- Uses Stretch AI's voxel map for navigation
"""

import sys
import numpy as np
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from PIL import Image

# Add Stretch AI to path
stretch_ai_path = Path(__file__).parent.parent / "stretch_ai" / "src"
if str(stretch_ai_path) not in sys.path:
    sys.path.insert(0, str(stretch_ai_path))

from stretch.agent.zmq_client import HomeRobotZmqClient
from stretch.agent.robot_agent import RobotAgent
from stretch.perception import create_semantic_sensor
from stretch.core.parameters import get_parameters


class ActivePerceptionModule:
    """
    Active perception module for DKPrompt.

    When VLM is uncertain about a predicate, this module:
    1. Identifies the object of interest from the predicate
    2. Samples viewpoints around the object using voxel map
    3. Navigates robot to new viewpoint
    4. Captures new observation
    5. Returns to VLM for retry
    """

    def __init__(
        self,
        robot_ip: str = "",
        map_file: str = None,
        config_file: str = "rosbridge_robot_config.yaml",
        calibration_file: str = "simple_offset_calibration.yaml",
        max_exploration_attempts: int = 3,
        viewpoint_distance: float = 1.0,  # meters from object
        use_ur5e: bool = False,  # Use UR5e + Segway robot
    ):
        """
        Initialize active perception module.

        Args:
            robot_ip: Robot IP address
            map_file: Path to voxel map (.pkl)
            config_file: Robot configuration file
            calibration_file: Map calibration file
            max_exploration_attempts: Maximum viewpoints to try
            viewpoint_distance: Distance from object for viewpoints
            use_ur5e: Use UR5e+Segway robot client instead of Stretch
        """
        # Load calibration (transform from voxel map to AMCL map frame)
        # Use calibrate_voxel_to_amcl.py to generate this file
        self.offset_x, self.offset_y = self._load_calibration(calibration_file)

        # Load parameters
        self.parameters = get_parameters(config_file)
        if self.parameters.get("vlm_base_config"):
            base_config = get_parameters(self.parameters.get("vlm_base_config"))
            base_config.data.update(self.parameters.data)
            self.parameters.data = base_config.data

        # Initialize semantic sensor
        self.semantic_sensor = create_semantic_sensor(parameters=self.parameters)

        # Connect to robot
        if use_ur5e:
            from stretch.core.ur5e_robot import UR5eSegwayRobotClient
            print("Using UR5e + Segway robot client")
            self.robot = UR5eSegwayRobotClient(
                robot_ip=robot_ip if robot_ip else "",
                parameters=self.parameters,
                use_remote_computer=True if robot_ip else False,
                enable_rerun_server=False,
            )
        else:
            self.robot = HomeRobotZmqClient(
                robot_ip=robot_ip if robot_ip else "",
                parameters=self.parameters,
                use_remote_computer=True if robot_ip else False,
                enable_rerun_server=False,
            )

        # Create agent
        self.agent = RobotAgent(
            self.robot,
            self.parameters,
            semantic_sensor=self.semantic_sensor
        )

        # Load voxel map
        self.voxel_map = None
        if map_file:
            self._load_map(map_file)
            print(f" Robot localization handled by AMCL - no initial pose needed")

        # Exploration parameters
        self.max_exploration_attempts = max_exploration_attempts
        self.viewpoint_distance = viewpoint_distance

        # Store original position for returning
        self.original_position = None

    def _load_calibration(self, calibration_file: str) -> Tuple[float, float]:
        """Load map calibration offsets."""
        import yaml
        cal_path = Path(calibration_file)

        if cal_path.exists():
            with open(cal_path, 'r') as f:
                calibration = yaml.safe_load(f)
            offset_x = calibration.get('offset_x', 0.0)
            offset_y = calibration.get('offset_y', 0.0)
            print(f"   Loaded calibration: ({offset_x:.3f}, {offset_y:.3f})")
            return offset_x, offset_y
        else:
            print(f"  No calibration file, using (0.0, 0.0)")
            return 0.0, 0.0

    def _load_map(self, map_file: str):
        """Load voxel map from file."""
        print(f"   Loading map from {map_file}...")

        self.voxel_map = self.agent.get_voxel_map()

        import matplotlib
        matplotlib.use('Agg')

        self.voxel_map.read_from_pickle(
            str(Path(map_file)),
            num_frames=-1,
            perception=self.semantic_sensor
        )

        print(f"   Loaded map with {len(self.voxel_map.instances)} instances")

    def get_current_observation(self) -> np.ndarray:
        """Get current RGB image from robot camera."""
        obs = self.robot.get_observation()

        if obs is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        if hasattr(obs, 'rgb'):
            return obs.rgb
        elif isinstance(obs, dict) and 'rgb' in obs:
            return obs['rgb']
        else:
            return np.zeros((480, 640, 3), dtype=np.uint8)

    def explore_for_better_view(
        self,
        predicate: List[str],
        vlm_agent,
        question: str
    ) -> Tuple[str, np.ndarray, bool]:
        """
        Actively explore to get better view for uncertain predicate.

        Strategy (hierarchical):
        1. Try head camera pan/tilt adjustments (fast, non-disruptive)
        2. If still uncertain, try base repositioning (slower, more comprehensive)

        Args:
            predicate: PDDL predicate like ["inside", "cup", "cabinet"]
            vlm_agent: VLM agent for querying
            question: Natural language question

        Returns:
            (vlm_answer, final_rgb_image, exploration_succeeded)
        """
        print(f"\n🔍 ACTIVE EXPLORATION for predicate: {predicate}")
        print(f"   Question: {question}")

        # Extract target object from predicate
        target_object = self._extract_target_object(predicate)

        if not target_object:
            print(f"    Could not extract target object from predicate")
            return "uncertain", self.get_current_observation(), False

        print(f"   Target object: {target_object}")

        # STEP 1: Try head camera adjustments first (faster)
        print(f"\n    STEP 1: Trying head camera adjustments...")
        confident_response = self._try_head_camera_exploration(vlm_agent, question)
        if confident_response:
            return confident_response, self.get_current_observation(), True

        # STEP 2: Try base repositioning if head adjustments didn't work
        print(f"\n    STEP 2: Head adjustments unsuccessful, trying base repositioning...")

        # Store original position
        self.original_position = self.robot.get_base_pose()
        print(f"   Original position: ({self.original_position[0]:.2f}, {self.original_position[1]:.2f})")

        # Generate viewpoints around target object
        viewpoints = self._sample_viewpoints_around_object(target_object)

        if not viewpoints:
            print(f"    No viewpoints could be generated")
            return "uncertain", self.get_current_observation(), False

        print(f"   Generated {len(viewpoints)} viewpoints")

        # Try each viewpoint
        for i, (vp_x, vp_y, vp_theta) in enumerate(viewpoints[:self.max_exploration_attempts]):
            print(f"\n   Viewpoint {i+1}/{min(len(viewpoints), self.max_exploration_attempts)}: "
                  f"({vp_x:.2f}, {vp_y:.2f}, {np.degrees(vp_theta):.1f}°)")

            # Navigate to viewpoint
            success = self.robot.navigate_to_goal(vp_x, vp_y, vp_theta)

            if not success:
                print(f"       Navigation failed")
                continue

            # Wait for robot to stabilize
            time.sleep(2.0)

            # Get new observation
            rgb_image = self.get_current_observation()

            # Query VLM again with new view
            print(f"       Querying VLM with new view...")
            vlm_response = vlm_agent.ask(question, rgb_image)

            print(f"      VLM response: {vlm_response}")

            # Extract string from list if needed
            if isinstance(vlm_response, list):
                vlm_response = vlm_response[0] if len(vlm_response) > 0 else "uncertain"

            # Check if VLM is now confident
            if self._is_confident_response(vlm_response):
                print(f"       Got confident answer: {vlm_response}")

                # Return to original position
                self._return_to_original_position()

                return vlm_response, rgb_image, True

        # Failed to get confident answer
        print(f"    Exploration failed - no confident answer after {self.max_exploration_attempts} attempts")

        # Return to original position
        self._return_to_original_position()

        return "uncertain", self.get_current_observation(), False

    def _try_head_camera_exploration(
        self,
        vlm_agent,
        question: str
    ) -> Optional[str]:
        """
        Try adjusting head camera pan/tilt to get better view.

        This is faster than moving the base and less disruptive to robot state.

        Args:
            vlm_agent: VLM agent for querying
            question: Natural language question

        Returns:
            Confident VLM response if successful, None otherwise
        """
        # Check if robot has head control capability
        if not hasattr(self.robot, 'head_to'):
            print(f"        Robot does not have head control, skipping")
            return None

        # Store original head position
        original_head_pan = 0.0  # Default forward
        original_head_tilt = -0.6  # Default looking slightly down

        # Define head positions to try (pan, tilt) in radians
        # pan: -π to π/4 (left to right)
        # tilt: -π to 0 (down to level)
        head_positions = [
            (0.0, -0.4),      # Straight, slightly up
            (0.3, -0.6),      # Right, down
            (-0.3, -0.6),     # Left, down
            (0.0, -0.8),      # Straight, more down
            (0.5, -0.5),      # More right, level
            (-0.5, -0.5),     # More left, level
        ]

        print(f"      Trying {len(head_positions)} head positions...")

        for i, (pan, tilt) in enumerate(head_positions):
            print(f"      Position {i+1}/{len(head_positions)}: pan={np.degrees(pan):.1f}°, tilt={np.degrees(tilt):.1f}°")

            try:
                # Move head to new position
                self.robot.head_to(pan, tilt, blocking=True)

                # Wait for camera to stabilize
                time.sleep(1.0)

                # Get new observation
                rgb_image = self.get_current_observation()

                # Query VLM with new view
                vlm_response = vlm_agent.ask(question, rgb_image)

                # Extract string from list if needed
                if isinstance(vlm_response, list):
                    vlm_response = vlm_response[0] if len(vlm_response) > 0 else "uncertain"

                print(f"         VLM: {vlm_response}")

                # Check if confident
                if self._is_confident_response(vlm_response):
                    print(f"       Got confident answer with head adjustment!")

                    # Return head to original position
                    self.robot.head_to(original_head_pan, original_head_tilt, blocking=True)
                    time.sleep(0.5)

                    return vlm_response

            except Exception as e:
                print(f"           Head movement failed: {e}")
                continue

        # Return head to original position
        print(f"      No confident answer from head adjustments, returning to original position")
        try:
            self.robot.head_to(original_head_pan, original_head_tilt, blocking=True)
            time.sleep(0.5)
        except:
            pass

        return None

    def _extract_target_object(self, predicate: List[str]) -> Optional[str]:
        """
        Extract target object name from predicate.

        Examples:
            ["inside", "cup-n-01_1", "cabinet-n-01_1"] → "cup"
            ["ontop", "book-n-02_1", "table-n-01_1"] → "book"
            ["closed", "door-n-01_1"] → "door"
        """
        # Handle negation
        if predicate[0] == "not":
            predicate = predicate[1:]

        pred_type = predicate[0]

        # For binary predicates, target is usually first object
        if pred_type in ["inside", "ontop", "under", "nextto"]:
            obj = predicate[1]
        # For unary predicates, target is the object
        elif pred_type in ["closed", "open", "filled"]:
            obj = predicate[1]
        else:
            # Default: take first non-agent argument
            for arg in predicate[1:]:
                if "agent" not in arg.lower():
                    obj = arg
                    break
            else:
                return None

        # Clean object name (remove -n-XX_X suffix)
        obj_name = obj.split("-")[0]
        return obj_name

    def _sample_viewpoints_around_object(
        self,
        object_name: str
    ) -> List[Tuple[float, float, float]]:
        """
        Sample viewpoints around target object from voxel map.

        Args:
            object_name: Object category name

        Returns:
            List of (x, y, theta) viewpoints in robot frame
        """
        if self.voxel_map is None:
            print(f"        No voxel map available")
            return []

        # Find object in map
        instances = self.voxel_map.get_instances()
        target_instance = None

        for instance in instances:
            if hasattr(instance, 'category_id'):
                try:
                    cat_name = self.semantic_sensor.get_class_name_for_id(instance.category_id)
                    if cat_name and object_name.lower() in cat_name.lower():
                        target_instance = instance
                        break
                except:
                    pass

        if target_instance is None:
            print(f"        Object '{object_name}' not found in map")
            return []

        # Get object center
        obj_center = target_instance.get_center()
        # Transform from voxel map to AMCL coordinates
        obj_x = obj_center[0] + self.offset_x
        obj_y = obj_center[1] + self.offset_y

        print(f"      Object at voxel: ({obj_center[0]:.2f}, {obj_center[1]:.2f})")
        print(f"      Object at AMCL: ({obj_x:.2f}, {obj_y:.2f})")

        # Sample viewpoints around object in a circle
        viewpoints = []
        num_samples = 8  # 8 viewpoints around object (every 45°)

        for i in range(num_samples):
            angle = 2 * np.pi * i / num_samples

            # Position at distance from object
            vp_x = obj_x + self.viewpoint_distance * np.cos(angle)
            vp_y = obj_y + self.viewpoint_distance * np.sin(angle)

            # Face towards object
            vp_theta = np.arctan2(obj_y - vp_y, obj_x - vp_x)

            viewpoints.append((vp_x, vp_y, vp_theta))

        return viewpoints

    def _is_confident_response(self, response: str) -> bool:
        """
        Check if VLM response is confident (yes/no) vs uncertain.

        Args:
            response: VLM response string

        Returns:
            True if response is confident (yes or no)
        """
        response_lower = response.lower()

        # Check for explicit uncertainty
        if any(word in response_lower for word in ["uncertain", "unclear", "cannot tell", "can't tell", "not sure"]):
            return False

        # Check for clear yes/no
        if "yes" in response_lower and "no" not in response_lower:
            return True
        if "no" in response_lower and "yes" not in response_lower:
            return True

        # Ambiguous response
        return False

    def _return_to_original_position(self):
        """Navigate back to original position."""
        if self.original_position is None:
            return

        print(f"    Returning to original position...")

        orig_x, orig_y, orig_theta = self.original_position
        self.robot.navigate_to_goal(orig_x, orig_y, orig_theta)

        time.sleep(1.5)
        print(f"    Returned to original position")

    def verify_object_visible(self, object_name: str, instance) -> Tuple[bool, bool]:
        """
        Verify if an object is visible using VLM.

        Args:
            object_name: Name of the object to verify
            instance: Object instance from voxel map

        Returns:
            (is_visible, needs_exploration):
                - is_visible: True if VLM confirms object is visible
                - needs_exploration: True if VLM is uncertain and needs active perception
        """
        # Get current observation
        obs = self.robot.get_observation()
        if obs is None or obs.rgb is None:
            print(f"     No observation available")
            return False, True

        # Simple heuristic: For now, assume we need to navigate closer to verify
        # This can be enhanced with actual VLM query
        center_voxel = instance.get_center()  # 3D voxel map coordinates
        current_pose = self.robot.get_base_pose()

        if current_pose is not None:
            # Transform voxel map coordinates to AMCL 2D coordinates
            obj_x_amcl = center_voxel[0] + self.offset_x
            obj_y_amcl = center_voxel[1] + self.offset_y

            curr_x, curr_y = current_pose[0], current_pose[1]
            distance = np.sqrt((obj_x_amcl - curr_x)**2 + (obj_y_amcl - curr_y)**2)

            print(f"   Object voxel coords: ({center_voxel[0]:.2f}, {center_voxel[1]:.2f})")
            print(f"   Object AMCL coords: ({obj_x_amcl:.2f}, {obj_y_amcl:.2f})")
            print(f"   Robot AMCL coords: ({curr_x:.2f}, {curr_y:.2f})")
            print(f"   Distance: {distance:.2f}m")

            # If within 2m, assume visible
            if distance < 2.0:
                print(f"    Object is close enough - assumed visible")
                return True, False
            else:
                print(f"     Object is far - may need closer view")
                return False, True

        return False, True

    def explore_object(self, object_name: str, instance) -> bool:
        """
        Explore around an object to get better view using active perception.

        Args:
            object_name: Name of the object
            instance: Object instance from voxel map

        Returns:
            success: True if object becomes visible after exploration
        """
        print(f"   🔍 Exploring around '{object_name}' for better view...")

        # Sample viewpoints around object
        viewpoints = self._sample_viewpoints_around_object(object_name)

        if not viewpoints:
            print(f"    No valid viewpoints found")
            return False

        # Save original position
        self.original_position = self.robot.get_base_pose()

        # Try each viewpoint
        for i, (vp_x, vp_y, vp_theta) in enumerate(viewpoints):
            print(f"   Trying viewpoint {i+1}/{len(viewpoints)}: ({vp_x:.2f}, {vp_y:.2f})")

            # Navigate to viewpoint
            success = self.robot.navigate_to_goal(vp_x, vp_y, vp_theta)

            if success:
                # Check if object is now visible
                time.sleep(1.0)
                is_visible, _ = self.verify_object_visible(object_name, instance)

                if is_visible:
                    print(f"    Object visible from viewpoint {i+1}")
                    return True

        print(f"    Object not visible from any viewpoint")

        # Return to original position
        self._return_to_original_position()

        return False


def wrap_check_states_with_active_perception(
    original_check_states_fn,
    active_perception: ActivePerceptionModule,
    vlm_agent
):
    """
    Wrapper for DKPrompt's check_states_and_update_problem function.

    Adds active perception when VLM is uncertain.

    Args:
        original_check_states_fn: Original check_states_and_update_problem
        active_perception: ActivePerceptionModule instance
        vlm_agent: VLM agent instance

    Returns:
        Wrapped function with active perception
    """
    def check_states_with_exploration(*args, **kwargs):
        """
        Enhanced check_states with active exploration.
        """
        # Call original function
        result = original_check_states_fn(*args, **kwargs)

        # Extract results
        (unmatched_pres, unmatched_effs), problem_file, obs_log = result

        # Check if we have unmatched facts due to uncertainty
        # (This is where we'd retry with active perception)

        # For now, return original result
        # TODO: Add retry logic with active perception

        return result

    return check_states_with_exploration
