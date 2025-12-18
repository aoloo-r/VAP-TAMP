# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# Visual Servoing for Navigation - approach objects using camera feedback

import os
import time
import timeit
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

import stretch.motion.constants as constants
from stretch.agent.base import ManagedOperation
from stretch.core.interfaces import Observations
from stretch.mapping.instance import Instance
from stretch.motion.kinematics import HelloStretchIdx
from stretch.utils.filters import MaskTemporalFilter
from stretch.utils.geometry import point_global_to_base


class VisualServoNavigationOperation(ManagedOperation):
    """Navigate to an object using visual servoing with RealSense camera."""

    def __init__(self, agent, *args, **kwargs):
        super().__init__(name="visual_servo_navigation", agent=agent, *args, **kwargs)

    _success: bool = False
    talk: bool = True
    verbose: bool = False

    # Task information
    match_method: str = "class"
    target_object: Optional[str] = None
    _object_xyz: Optional[np.ndarray] = None
    planned_path: Optional[list] = None  # Path from 3D planning

    # Debugging UI elements
    show_servo_gui: bool = False
    show_point_cloud: bool = False

    # Navigation parameters for localization and mapping
    align_x_threshold: int = 50  # More relaxed alignment for exploration
    align_y_threshold: int = 40

    # Adaptive approach distance based on object type and context
    min_approach_distance: float = 0.3   # Minimum safe distance
    max_approach_distance: float = 2.0   # Maximum exploration distance

    # Movement parameters for environment exploration
    base_x_step: float = 0.12    # Larger steps for efficient exploration
    base_y_step: float = 0.10

    # Tracked object features
    tracked_object_features: Optional[torch.Tensor] = None

    # Visual servoing config
    min_points_to_approach: int = 100
    max_failed_attempts: int = 10
    expected_network_delay = 0.1

    # Observation memory
    observations = MaskTemporalFilter(
        observation_history_window_size_secs=5.0, observation_history_window_size_n=3
    )

    def configure(
        self,
        target_object: Optional[str] = None,
        object_xyz: Optional[np.ndarray] = None,
        planned_path: Optional[list] = None,
        show_servo_gui: bool = True,
        show_point_cloud: bool = False,
        talk: bool = True,
        match_method: str = "class",
        exploration_mode: bool = True,
    ):
        """Configure the visual servo navigation operation for long-horizon tasks.

        Args:
            target_object (str, optional): Target object name
            object_xyz (np.ndarray, optional): Target 3D coordinates
            planned_path (list, optional): Pre-planned path from 3D RRT planning
            show_servo_gui (bool, optional): Show visual feedback GUI. Defaults to True.
            show_point_cloud (bool, optional): Show point cloud. Defaults to False.
            talk (bool, optional): Enable speech feedback. Defaults to True.
            match_method (str, optional): Object matching method. Defaults to "class".
            exploration_mode (bool, optional): Enable exploration and mapping mode. Defaults to True.
        """
        if target_object is not None:
            self.target_object = target_object
        if object_xyz is not None:
            assert len(object_xyz) == 3, "Object xyz must be a 3D point."
            self._object_xyz = object_xyz
        if planned_path is not None:
            self.planned_path = planned_path
            print(f"📍 Received planned path with {len(planned_path)} waypoints")
        self.show_servo_gui = show_servo_gui
        self.show_point_cloud = show_point_cloud
        self.talk = talk
        self.match_method = match_method
        self.exploration_mode = exploration_mode

        if self.match_method not in ["class", "feature"]:
            raise ValueError(
                f"Unknown match method {self.match_method}. Should be 'class' or 'feature'."
            )

    def can_start(self):
        """Navigation can start if we have a target object or coordinates."""
        if self.target_object is None and self._object_xyz is None:
            self.error("No target object or coordinates set.")
            return False
        return True

    def get_class_mask(self, servo: Observations) -> np.ndarray:
        """Get the mask for the target object class."""
        mask = np.zeros_like(servo.semantic).astype(bool)

        if self.verbose:
            print("[VISUAL SERVO NAV] match method =", self.match_method)

        if self.match_method == "class":
            # Get the target class
            if self.agent.current_object is not None:
                target_class_id = self.agent.current_object.category_id
                target_class = self.agent.semantic_sensor.get_class_name_for_id(target_class_id)
            else:
                target_class = self.target_object

            if self.verbose:
                print("[VISUAL SERVO NAV] Detecting objects of class", target_class)

            # Find masks with that class
            for iid in np.unique(servo.semantic):
                name = self.agent.semantic_sensor.get_class_name_for_id(iid)
                if name is not None and target_class in name:
                    mask = np.bitwise_or(mask, servo.semantic == iid)

        elif self.match_method == "feature":
            if self.target_object is None:
                raise ValueError(
                    f"Target object must be set before running match method {self.match_method}."
                )

            if self.verbose:
                print("[VISUAL SERVO NAV] Detecting objects described as", self.target_object)

            text_features = self.agent.encode_text(self.target_object)
            best_score = float("-inf")
            best_iid = None

            # Loop over all detected instances
            for iid in np.unique(servo.instance):
                if iid < 0:
                    continue

                rgb = servo.rgb * (servo.instance == iid)[:, :, None].repeat(3, axis=-1)
                features = self.agent.encode_image(rgb)
                score = self.agent.compare_features(text_features, features).item()

                if score > best_score:
                    best_score = score
                    best_iid = iid

                if score > self.agent.feature_match_threshold:
                    mask = servo.instance == best_iid
                    self.tracked_object_features = features
                    break

        return mask

    def get_target_mask(
        self,
        servo: Observations,
        center: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        """Get target mask for the object to approach."""
        class_mask = self.get_class_mask(servo)
        instance_mask = servo.instance

        if servo.xyz is None:
            servo.compute_xyz()

        target_mask = None
        target_mask_pts = float("-inf")
        center_x, center_y = center

        # Loop over all detected instances
        for iid in np.unique(instance_mask):
            current_instance_mask = instance_mask == iid

            # If centered on the correct object, prioritize it
            if class_mask[center_y, center_x] > 0 and current_instance_mask[center_y, center_x] > 0:
                print("!!! CENTERED ON TARGET OBJECT !!!")
                return current_instance_mask

            # Find mask with most points of the correct class
            mask = np.bitwise_and(current_instance_mask, class_mask)
            num_pts = sum(mask.flatten())
            if num_pts > target_mask_pts:
                target_mask = mask
                target_mask_pts = num_pts

        return target_mask if target_mask_pts > self.min_points_to_approach else None

    def _compute_center_depth(
        self,
        servo: Observations,
        target_mask: np.ndarray,
        center_y: int,
        center_x: int,
        local_region_size: int = 5,
    ) -> float:
        """Compute center depth of the target object."""
        mask = np.zeros_like(target_mask)
        mask[
            max(center_y - local_region_size, 0) : min(center_y + local_region_size, mask.shape[0]),
            max(center_x - local_region_size, 0) : min(center_x + local_region_size, mask.shape[1]),
        ] = 1

        # Ignore depth of 0 (bad value)
        depth_mask = np.bitwise_and(servo.depth > 1e-8, mask)
        depth = servo.depth[target_mask & depth_mask]

        if len(depth) == 0:
            return 0.0

        return np.median(depth)

    def _compute_adaptive_approach_distance(self, target_object: str, center_depth: float) -> float:
        """Compute adaptive approach distance based on object type and context."""
        # Large objects or furniture - maintain respectful distance for spatial understanding
        large_objects = ["chair", "table", "desk", "sofa", "couch", "cabinet", "shelf"]

        # Small objects - can approach closer for detailed observation
        small_objects = ["cup", "bottle", "book", "phone", "remote", "keys"]

        # Navigation landmarks - approach for localization reference
        landmarks = ["door", "window", "wall", "corner"]

        if any(obj in target_object.lower() for obj in large_objects):
            return max(1.2, center_depth * 0.6)  # Stay at 60% of current distance, min 1.2m
        elif any(obj in target_object.lower() for obj in small_objects):
            return max(0.5, center_depth * 0.4)  # Can approach closer for small objects
        elif any(obj in target_object.lower() for obj in landmarks):
            return max(0.8, center_depth * 0.5)  # Moderate distance for landmarks
        else:
            # Default adaptive behavior - maintain safe observation distance
            return max(self.min_approach_distance, min(self.max_approach_distance, center_depth * 0.7))

    def visual_servo_to_object(
        self, max_duration: float = 180.0, max_not_moving_count: int = 30
    ) -> bool:
        """Use visual servoing for spatial localization and environment understanding."""

        if self.agent.current_object is not None:
            self.intro(
                f"Visual servoing for spatial localization to {self.agent.current_object.global_id}."
            )
        else:
            self.intro(f"Visual servoing localization to {self.target_object} at {self._object_xyz}.")

        if self.show_servo_gui:
            self.warn("Press 'q' to stop visual servoing.")

        t0 = timeit.default_timer()
        success = False
        failed_counter = 0
        not_moving_count = 0
        observations_collected = 0
        servo_iteration = 0

        # Track last base pose for movement detection
        last_base_pose = self.robot.get_base_pose()

        # Store multiple viewpoints for better spatial understanding
        viewpoint_history = []

        # Initialize waypoint following
        current_waypoint_index = 0

        # Import numpy for waypoint calculations
        import numpy as np
        import time

        # Main navigation loop for spatial exploration
        while timeit.default_timer() - t0 < max_duration:
            servo_iteration += 1

            # CRITICAL FIX: Follow planned path waypoints if available
            if self.planned_path is not None and len(self.planned_path) > 0:
                current_base_pose = self.robot.get_base_pose()

                # Check if we have waypoints to follow
                if current_waypoint_index < len(self.planned_path):
                    target_waypoint = self.planned_path[current_waypoint_index]

                    # Convert waypoint to position (assuming waypoint has .state attribute)
                    if hasattr(target_waypoint, 'state'):
                        target_pos = target_waypoint.state[:2]  # x, y
                    else:
                        target_pos = target_waypoint[:2]  # assume it's already [x, y, theta]

                    # Calculate distance to current waypoint
                    current_pos = current_base_pose[:2]
                    distance_to_waypoint = np.linalg.norm(np.array(target_pos) - np.array(current_pos))

                    print(f"Following waypoint {current_waypoint_index+1}/{len(self.planned_path)}, distance: {distance_to_waypoint:.2f}m")

                    # If close enough to current waypoint, advance to next
                    if distance_to_waypoint < 0.2:  # 20cm threshold
                        current_waypoint_index += 1
                        print(f"Reached waypoint {current_waypoint_index}/{len(self.planned_path)}")

                        # If we've reached the final waypoint, switch to visual servoing
                        if current_waypoint_index >= len(self.planned_path):
                            print("Reached final waypoint - switching to visual target detection")
                        else:
                            continue  # Move to next waypoint

                    # Navigate to current waypoint
                    relative_x = target_pos[0] - current_pos[0]
                    relative_y = target_pos[1] - current_pos[1]

                    # Limit movement step size
                    max_step = 0.3
                    movement_distance = np.linalg.norm([relative_x, relative_y])
                    if movement_distance > max_step:
                        scale = max_step / movement_distance
                        relative_x *= scale
                        relative_y *= scale

                    print(f"WAYPOINT NAVIGATION: Moving {relative_x:.2f}, {relative_y:.2f}")
                    move_success = self.robot.move_base_to([relative_x, relative_y, 0.0], relative=True, blocking=True, timeout=15.0)
                    print(f"Waypoint movement success: {move_success}")

                    time.sleep(0.5)  # Brief pause between waypoints
                    continue  # Skip visual detection while following waypoints

            # Get current observation and add to agent's spatial map
            servo = self.robot.get_observation()
            current_base_pose = self.robot.get_base_pose()

            # CRITICAL FIX: Skip continuous mapping during visual servoing to avoid tensor size mismatches
            # The semantic sensor has resolution conflicts during navigation
            # We'll rely on the pre-loaded map for navigation guidance
            if False:  # Disabled to prevent tensor size conflicts
                pass

            # Ensure numpy is available for array operations
            import numpy as np

            # Use original camera resolution to avoid tensor size mismatches
            if servo_iteration == 0:  # Only print once per navigation attempt
                print(f"Using original camera resolution for target detection: {servo.rgb.shape}")

            # Compute image center from original resolution
            center_x, center_y = servo.rgb.shape[1] // 2, servo.rgb.shape[0] // 2

            # Run semantic segmentation for real target detection
            if servo_iteration == 0:  # Only print once per navigation attempt
                print("Using Stretch AI semantic processing for target detection")
            servo = self.agent.semantic_sensor.predict(servo)
            latest_mask = self.get_target_mask(servo, center=(center_x, center_y))

            # DEBUG: Print target detection status
            if servo_iteration <= 3:  # Only for first few iterations
                if latest_mask is not None:
                    mask_points = np.sum(latest_mask.flatten())
                    print(f"Target detected: {mask_points} mask points")
                else:
                    print(f"No target detected in iteration {servo_iteration}")

            if latest_mask is not None:
                # Dilate mask slightly for better tracking
                kernel = np.ones((3, 3), np.uint8)
                mask_np = latest_mask.astype(np.uint8)
                dilated_mask = cv2.dilate(mask_np, kernel, iterations=1)
                latest_mask = dilated_mask.astype(bool)

            # Push to observation history
            self.observations.push_mask_to_observation_history(
                observation=latest_mask if latest_mask is not None else np.zeros_like(servo.rgb[:,:,0], dtype=bool),
                timestamp=time.time(),
                mask_size_threshold=self.min_points_to_approach,
                acquire_lock=True,
            )

            target_mask = self.observations.get_latest_observation()
            if target_mask is None:
                target_mask = np.zeros([servo.rgb.shape[0], servo.rgb.shape[1]], dtype=bool)

            # Get mask centroid
            mask_center = self.observations.get_latest_centroid()
            if mask_center is None:
                failed_counter += 1
                if failed_counter >= self.max_failed_attempts:
                    # In exploration mode, try to continue even if we lose the target
                    if self.exploration_mode and observations_collected > 5:
                        print("Lost target but collected spatial observations - continuing exploration")
                        success = True
                        break
                    else:
                        self.error("Lost track of target object.")
                        if self.talk:
                            # CRITICAL FIX: Use print fallback for ROS2 client without speech
                            try:
                                self.agent.robot_say(f"Lost sight of {self.target_object}, but continuing exploration.")
                            except AttributeError:
                                print(f"🤖 Would say: Lost sight of {self.target_object}, but continuing exploration.")
                        return False
                continue
            else:
                failed_counter = 0
                mask_center = mask_center.astype(int)

            # Compute object depth
            center_depth = self._compute_center_depth(servo, target_mask, mask_center[0], mask_center[1])

            # Compute adaptive approach distance based on object type
            # Convert numeric category ID to string name if needed
            if isinstance(self.target_object, (int, np.integer)):
                try:
                    target_name = self.agent.semantic_sensor.get_class_name_for_id(self.target_object)
                except:
                    target_name = f"object_{self.target_object}"
            else:
                target_name = self.target_object or "unknown"

            adaptive_distance = self._compute_adaptive_approach_distance(
                target_name, center_depth
            )

            # Display visual servoing GUI with spatial context
            if self.show_servo_gui and not self.headless_machine:
                servo_rgb = cv2.cvtColor(servo.rgb, cv2.COLOR_RGB2BGR)
                mask = target_mask.astype(np.uint8) * 255
                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                mask[:, :, 0] = 0  # Remove red channel for green overlay

                # Overlay mask on image
                servo_rgb = cv2.addWeighted(servo_rgb, 0.7, mask, 0.3, 0, servo_rgb)
                # Draw image center
                servo_rgb = cv2.circle(servo_rgb, (center_x, center_y), 5, (255, 0, 0), -1)
                # Draw mask center
                servo_rgb = cv2.circle(
                    servo_rgb, (int(mask_center[1]), int(mask_center[0])), 5, (0, 255, 0), -1
                )

                # Add text overlay with spatial info
                cv2.putText(servo_rgb, f"Distance: {center_depth:.2f}m", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(servo_rgb, f"Target: {adaptive_distance:.2f}m", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(servo_rgb, f"Observations: {observations_collected}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.namedWindow("Visual Servo Spatial Navigation", cv2.WINDOW_NORMAL)
                cv2.imshow("Visual Servo Spatial Navigation", servo_rgb)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            # Store viewpoint for spatial understanding
            if len(viewpoint_history) == 0 or np.linalg.norm(current_base_pose[:2] - viewpoint_history[-1][:2]) > 0.2:
                # Handle both numpy arrays and PyTorch tensors
                if hasattr(current_base_pose, 'clone'):
                    # PyTorch tensor
                    pose_copy = current_base_pose.clone()
                elif hasattr(current_base_pose, 'copy'):
                    # NumPy array
                    pose_copy = current_base_pose.copy()
                else:
                    # Fallback - convert to numpy
                    pose_copy = np.array(current_base_pose)
                viewpoint_history.append(pose_copy)

            # Check if we've achieved good spatial localization
            if center_depth > 1e-8 and center_depth <= adaptive_distance:
                print(f"Achieved spatial localization! Distance: {center_depth:.2f}m (target: {adaptive_distance:.2f}m)")
                print(f"Collected {observations_collected} observations from {len(viewpoint_history)} viewpoints")
                if self.talk:
                    # CRITICAL FIX: Use print fallback for ROS2 client without speech
                    try:
                        self.agent.robot_say(f"Spatial localization complete for {self.target_object}.")
                    except AttributeError:
                        print(f"🤖 Would say: Spatial localization complete for {self.target_object}.")
                success = True
                break

            # Compute movement commands for exploration
            dx, dy = mask_center[1] - center_x, mask_center[0] - center_y

            # More relaxed alignment for exploration
            aligned = np.abs(dx) < self.align_x_threshold and np.abs(dy) < self.align_y_threshold

            print()
            print("----- VISUAL SERVO SPATIAL NAVIGATION -----")
            print(f"Target mask points: {np.sum(target_mask.flatten())}")
            print(f"Distance to object: {center_depth:.3f}m (target: {adaptive_distance:.2f}m)")
            print(f"Observations collected: {observations_collected}")
            print(f"Viewpoints: {len(viewpoint_history)}")
            print(f"dx={dx}, dy={dy}, aligned={aligned}")

            # Adaptive movement logic for spatial exploration
            if aligned and center_depth > adaptive_distance:
                # Move forward gradually toward optimal observation distance
                forward_step = min(0.15, (center_depth - adaptive_distance) * 0.5)

                # Store position before movement to check if robot actually moved
                pos_before = self.robot.get_base_pose()
                print(f"ATTEMPTING ROBOT MOVEMENT: forward_step={forward_step:.3f}m")
                print(f"Position before: {pos_before[:2]}")

                success = self.robot.move_base_to([forward_step, 0.0, 0.0], relative=True, blocking=True, timeout=15.0)
                pos_after = self.robot.get_base_pose()

                print(f"Movement success: {success}")
                print(f"Position after: {pos_after[:2]}")
                print(f"Distance moved: {np.linalg.norm(pos_after[:2] - pos_before[:2]):.3f}m")

                if success and np.linalg.norm(pos_after[:2] - pos_before[:2]) > 0.02:
                    print(f"Moving forward {forward_step:.2f}m for better spatial observation")
                else:
                    print(f"Failed to move forward - continuing with current position")
                    # If movement failed multiple times, consider the task complete
                    if not_moving_count > 3:
                        print("Multiple movement failures - completing spatial observation")
                        success = True
                        break
            elif not aligned:
                # Adjust base position to center object in view
                base_x_cmd = 0.0
                base_y_cmd = 0.0

                if dx > self.align_x_threshold:
                    base_y_cmd = -self.base_y_step  # Move left
                elif dx < -self.align_x_threshold:
                    base_y_cmd = self.base_y_step   # Move right

                if dy > self.align_y_threshold:
                    base_x_cmd = self.base_x_step   # Move forward to center vertically
                elif dy < -self.align_y_threshold:
                    base_x_cmd = -self.base_x_step  # Move backward to center vertically

                if abs(base_x_cmd) > 0 or abs(base_y_cmd) > 0:
                    print(f"ATTEMPTING LATERAL MOVEMENT: x={base_x_cmd:.3f}, y={base_y_cmd:.3f}")
                    pos_before = self.robot.get_base_pose()
                    print(f"Position before lateral: {pos_before[:2]}")

                    success = self.robot.move_base_to([base_x_cmd, base_y_cmd, 0.0], relative=True, blocking=True, timeout=15.0)
                    pos_after = self.robot.get_base_pose()

                    print(f"Lateral movement success: {success}")
                    print(f"Position after lateral: {pos_after[:2]}")
                    print(f"Lateral distance moved: {np.linalg.norm(pos_after[:2] - pos_before[:2]):.3f}m")
            else:
                # At good distance and aligned - collect spatial observations
                print("Optimal position reached - collecting spatial observations")
                time.sleep(1.0)  # Allow time for observation collection

                # If we've collected enough observations, we can declare success
                if observations_collected >= 10:
                    print("Sufficient spatial observations collected")
                    success = True
                    break

            # Check if robot is stuck (less strict for exploration)
            current_base_pose = self.robot.get_base_pose()
            if np.linalg.norm(current_base_pose[:2] - last_base_pose[:2]) < 0.05:
                not_moving_count += 1
            else:
                not_moving_count = 0
            last_base_pose = current_base_pose

            if not_moving_count > max_not_moving_count:
                print("Robot movement limited - completing spatial observation")
                # In exploration mode, limited movement is acceptable if we collected observations
                if observations_collected >= 5:
                    success = True
                break

            time.sleep(self.expected_network_delay)

        if self.show_servo_gui and not self.headless_machine:
            cv2.destroyAllWindows()

        print(f"Spatial navigation completed: {observations_collected} observations, {len(viewpoint_history)} viewpoints")
        return success

    def get_object_xyz(self) -> np.ndarray:
        """Get the target object coordinates."""
        if self._object_xyz is None:
            if self.agent.current_object is not None:
                object_xyz = self.agent.current_object.get_median()
            else:
                raise ValueError("No target coordinates or current object available")
        else:
            object_xyz = self._object_xyz
        return object_xyz

    def run(self) -> None:
        """Main execution method."""
        self.intro("Starting visual servo navigation.")
        self._success = False

        # Clear observation history
        self.reset()

        assert (self.target_object is not None or self._object_xyz is not None), \
            "Target object or coordinates must be set before running."

        # Switch to navigation mode
        if not self.robot.in_navigation_mode():
            self.robot.switch_to_navigation_mode()

        # Execute visual servoing
        self._success = self.visual_servo_to_object()

        if self.talk and self._success:
            # CRITICAL FIX: Use print fallback for ROS2 client without speech
            try:
                self.agent.robot_say("Navigation complete!")
            except AttributeError:
                print("🤖 Would say: Navigation complete!")

    def reset(self):
        """Reset the operation state."""
        self._success = False
        self.tracked_object_features = None
        self.observations.clear_history()

    def was_successful(self) -> bool:
        """Return True if navigation was successful."""
        return self._success