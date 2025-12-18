#!/usr/bin/env python3
# Copyright (c) Hello Robot, Inc.
# Hybrid Navigation: 3D Map → Rough Navigation → Visual Servoing

"""
Two-phase navigation for Segway with misaligned maps:

Phase 1: Rough Navigation (±50cm accuracy)
- Query 3D voxel map for object position
- Apply manual transformation to robot's coordinate frame
- Navigate using move_base

Phase 2: Visual Servoing (±5cm accuracy)
- Use camera to detect object
- Fine-tune position with visual feedback
- Center object in view
"""

import time
import numpy as np
import click
from pathlib import Path

from stretch.agent.zmq_client import HomeRobotZmqClient
from stretch.core.parameters import get_parameters
from stretch.mapping.voxel import SparseVoxelMap
from stretch.perception import create_semantic_sensor
from stretch.utils.logger import Logger

logger = Logger(__name__)


class MapTransform:
    """Handles coordinate transformation between 3D map and robot 2D map."""

    def __init__(self, offset_x=0.0, offset_y=0.0, rotation=0.0):
        """
        Initialize transform with manual calibration.

        Args:
            offset_x: X offset from 3D map to robot map (meters)
            offset_y: Y offset from 3D map to robot map (meters)
            rotation: Rotation from 3D map to robot map (radians)
        """
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.rotation = rotation

    def transform_3d_to_2d(self, pos_3d):
        """Convert 3D map position to 2D robot map position.

        Args:
            pos_3d: [x, y, z] position in 3D map

        Returns:
            [x, y] position in robot's 2D map
        """
        x_3d, y_3d = pos_3d[0], pos_3d[1]

        # Apply rotation
        cos_r = np.cos(self.rotation)
        sin_r = np.sin(self.rotation)
        x_rot = x_3d * cos_r - y_3d * sin_r
        y_rot = x_3d * sin_r + y_3d * cos_r

        # Apply translation
        x_2d = x_rot + self.offset_x
        y_2d = y_rot + self.offset_y

        return [x_2d, y_2d]

    @classmethod
    def calibrate_from_landmark(cls, landmark_3d, landmark_2d):
        """Simple calibration from one landmark point.

        Args:
            landmark_3d: [x, y] position of landmark in 3D map
            landmark_2d: [x, y] position of same landmark in robot map

        Returns:
            MapTransform object
        """
        offset_x = landmark_2d[0] - landmark_3d[0]
        offset_y = landmark_2d[1] - landmark_3d[1]

        logger.info(f"Calibrated transform: offset=({offset_x:.2f}, {offset_y:.2f})")
        return cls(offset_x=offset_x, offset_y=offset_y, rotation=0.0)


def visual_servo_to_object(robot, semantic_sensor, target_object, timeout=60.0):
    """Use visual servoing to precisely locate object.

    Args:
        robot: Robot client
        semantic_sensor: Semantic segmentation sensor
        target_object: Name of object to find
        timeout: Maximum time for visual servoing

    Returns:
        bool: True if object was successfully centered
    """
    logger.info(f"Starting visual servoing for '{target_object}'")

    # Visual servoing parameters
    align_x_threshold = 80  # pixels - relaxed for initial detection
    align_y_threshold = 60  # pixels
    min_points_to_approach = 100
    max_iterations = 50
    base_x_step = 0.10  # meters
    base_y_step = 0.08  # meters

    t0 = time.time()
    iteration = 0

    while time.time() - t0 < timeout and iteration < max_iterations:
        iteration += 1

        # Get camera observation
        obs = robot.get_observation()
        if obs is None:
            logger.warning("No observation available")
            time.sleep(0.5)
            continue

        # Run semantic segmentation
        obs = semantic_sensor.predict(obs)

        # Get image center
        center_x = obs.rgb.shape[1] // 2
        center_y = obs.rgb.shape[0] // 2

        # Find target object mask
        target_mask = None
        for instance_id in np.unique(obs.semantic):
            class_name = semantic_sensor.get_class_name_for_id(instance_id)
            if class_name and target_object.lower() in class_name.lower():
                target_mask = (obs.semantic == instance_id)
                break

        if target_mask is None or np.sum(target_mask) < min_points_to_approach:
            logger.warning(f"Object '{target_object}' not visible (iteration {iteration})")
            time.sleep(0.3)
            continue

        # Calculate mask centroid
        y_indices, x_indices = np.where(target_mask)
        if len(x_indices) == 0:
            continue

        mask_center_x = int(np.mean(x_indices))
        mask_center_y = int(np.mean(y_indices))

        # Calculate offset from image center
        dx = mask_center_x - center_x
        dy = mask_center_y - center_y

        logger.info(f"[Iter {iteration}] Object at ({mask_center_x}, {mask_center_y}), "
                   f"center at ({center_x}, {center_y}), offset=({dx}, {dy})")

        # Check if aligned
        if abs(dx) < align_x_threshold and abs(dy) < align_y_threshold:
            logger.info(f"✅ Object centered! Final offset: ({dx}, {dy})")
            return True

        # Calculate movement commands
        base_x_cmd = 0.0
        base_y_cmd = 0.0

        if dx > align_x_threshold:
            base_y_cmd = -base_y_step  # Move left
        elif dx < -align_x_threshold:
            base_y_cmd = base_y_step   # Move right

        if dy > align_y_threshold:
            base_x_cmd = base_x_step   # Move forward
        elif dy < -align_y_threshold:
            base_x_cmd = -base_x_step  # Move backward

        # Execute movement
        if abs(base_x_cmd) > 0 or abs(base_y_cmd) > 0:
            logger.info(f"Moving: x={base_x_cmd:.2f}, y={base_y_cmd:.2f}")
            robot.move_base_to([base_x_cmd, base_y_cmd, 0.0], relative=True, blocking=True, timeout=10.0)
            time.sleep(0.3)

    logger.error(f"Visual servoing timeout after {iteration} iterations")
    return False


@click.command()
@click.option("--robot_ip", default="", help="Robot IP address")
@click.option("--map_file", required=True, help="Path to 3D voxel map (.pkl)")
@click.option("--target_object", default=None, help="Object to find (e.g., 'cup')")
@click.option("--task", default=None, help="Natural language task for VLM (e.g., 'find the trash can')")
@click.option("--api_key", default=None, help="VLM API key (Gemini or OpenAI)")
@click.option("--landmark_3d", type=str, default=None,
              help="Landmark position in 3D map as 'x,y' (e.g., '3.0,2.0')")
@click.option("--landmark_2d", type=str, default=None,
              help="Same landmark position in robot map as 'x,y' (e.g., '5.0,-1.0')")
@click.option("--offset_x", type=float, default=0.0, help="Manual X offset (meters)")
@click.option("--offset_y", type=float, default=0.0, help="Manual Y offset (meters)")
@click.option("--use_visual_servo", is_flag=True, default=True, help="Enable visual servoing")
def main(robot_ip, map_file, target_object, task, api_key, landmark_3d, landmark_2d,
         offset_x, offset_y, use_visual_servo):
    """Hybrid navigation: 3D map → rough goal → visual servoing."""

    logger.info("=" * 70)
    logger.info("SEGWAY HYBRID NAVIGATION")
    logger.info("=" * 70)

    # 1. Setup transform
    if landmark_3d and landmark_2d:
        # Calibrate from landmarks
        l3d = [float(x) for x in landmark_3d.split(',')]
        l2d = [float(x) for x in landmark_2d.split(',')]
        transform = MapTransform.calibrate_from_landmark(l3d, l2d)
    else:
        # Use manual offsets
        transform = MapTransform(offset_x=offset_x, offset_y=offset_y, rotation=0.0)
        logger.info(f"Using manual transform: offset=({offset_x}, {offset_y})")

    # 2. Load parameters with base config merge (same as vlm_planning)
    logger.info("Loading configuration...")
    parameters = get_parameters("rosbridge_robot_config.yaml")

    # Merge with base config if specified (same as vlm_planning does)
    if parameters.get("vlm_base_config"):
        base_config_file = parameters.get("vlm_base_config")
        base_parameters = get_parameters(base_config_file)
        base_parameters.data.update(parameters.data)
        parameters.data = base_parameters.data

    # 3. Initialize semantic sensor (needed for map loading and visual servoing)
    logger.info("Initializing semantic sensor...")
    semantic_sensor = create_semantic_sensor(parameters=parameters)
    logger.info("✅ Semantic sensor ready")

    # 4. Connect to robot
    logger.info("Connecting to robot...")
    robot = HomeRobotZmqClient(
        robot_ip=robot_ip if robot_ip else "",
        parameters=parameters,
        use_remote_computer=False,  # Local connection
        enable_rerun_server=False,
    )
    logger.info("✅ Connected to robot")

    # 5. Load 3D map (same as calibrate_map_transform.py)
    logger.info(f"Loading 3D map from {map_file}...")
    from stretch.agent.robot_agent import RobotAgent
    from stretch.utils.dummy_stretch_client import DummyStretchClient

    # Use RobotAgent to get voxel map (same pattern as vlm_planning)
    dummy_robot = DummyStretchClient()
    agent = RobotAgent(dummy_robot, parameters, semantic_sensor=semantic_sensor)
    voxel_map = agent.get_voxel_map()

    # Set matplotlib to non-GUI backend before loading
    import matplotlib
    matplotlib.use('Agg')

    # Load from pickle
    voxel_map.read_from_pickle(str(Path(map_file)), num_frames=-1, perception=semantic_sensor)
    logger.info(f"✅ Loaded map with {len(voxel_map.instances)} instances")

    # 6. Determine target: use VLM task or direct object name
    target_instance = None

    if task:
        # Use VLM planning to determine target
        logger.info(f"🎯 Using VLM planner for task: '{task}'")
        from stretch.agent.vlm_planner import VLMPlanner

        vlm_planner = VLMPlanner(agent, api_key=api_key)
        plan = vlm_planner.plan(query=task, show_plan=False, show_prompts=False)

        logger.info(f"Generated plan: {plan}")

        # Extract instance from plan
        import re
        match = re.search(r'img_(\d+)', str(plan))
        if match:
            # Get world representation to map crop_id to instance_id
            world_repr = agent.get_object_centric_observations(task=task, show_prompts=False)
            crop_id = int(match.group(1))

            if crop_id < len(world_repr.object_images):
                instance_id = world_repr.object_images[crop_id].instance_id
                target_instance = voxel_map.get_instances()[instance_id]

                # Get category name
                category_name = semantic_sensor.get_class_name_for_id(target_instance.category_id)
                logger.info(f"✅ VLM selected: instance {instance_id} ({category_name})")
            else:
                logger.error(f"❌ Invalid crop ID {crop_id}")
        else:
            logger.error(f"❌ Could not parse instance from plan: {plan}")

    elif target_object:
        # Direct object search (original behavior)
        logger.info(f"Searching for '{target_object}' in 3D map...")

        all_instances = voxel_map.get_instances()
        for instance in all_instances:
            # Get human-readable category name
            category_name = None
            if hasattr(instance, 'category_id'):
                try:
                    category_name = semantic_sensor.get_class_name_for_id(instance.category_id)
                except:
                    category_name = str(instance.category_id)

            # Match against target
            if category_name and target_object.lower() in category_name.lower():
                target_instance = instance
                logger.info(f"✅ Found '{target_object}' as '{category_name}'")
                break
    else:
        logger.error("❌ Must specify either --task or --target_object")
        return

    if target_instance is None:
        logger.error(f"❌ Target not found in map")
        logger.info("Available objects (first 10):")
        all_instances = voxel_map.get_instances()
        for i, inst in enumerate(all_instances[:10]):
            cat_name = semantic_sensor.get_class_name_for_id(inst.category_id) if hasattr(inst, 'category_id') else 'unknown'
            logger.info(f"  - {cat_name}")
        return

    # 6. Get 3D position and transform to 2D
    pos_3d = target_instance.get_center()
    logger.info(f"3D position: ({pos_3d[0]:.2f}, {pos_3d[1]:.2f}, {pos_3d[2]:.2f})")

    pos_2d = transform.transform_3d_to_2d(pos_3d)
    logger.info(f"2D position (robot frame): ({pos_2d[0]:.2f}, {pos_2d[1]:.2f})")

    # 7. Phase 1: Rough navigation with move_base
    logger.info("")
    logger.info("PHASE 1: ROUGH NAVIGATION")
    logger.info("-" * 40)

    current_pose = robot.get_base_pose()
    logger.info(f"Current position: ({current_pose[0]:.2f}, {current_pose[1]:.2f})")

    # Calculate goal orientation (face object)
    goal_theta = np.arctan2(pos_2d[1] - current_pose[1],
                           pos_2d[0] - current_pose[0])

    logger.info(f"Sending rough navigation goal:")
    logger.info(f"  Position: ({pos_2d[0]:.2f}, {pos_2d[1]:.2f})")
    logger.info(f"  Orientation: {goal_theta:.2f} rad ({np.degrees(goal_theta):.1f}°)")

    success = robot.navigate_to_goal(pos_2d[0], pos_2d[1], goal_theta)

    if not success:
        logger.error("❌ Failed to send navigation goal")
        return

    logger.info("✅ Navigation goal sent successfully")
    logger.info("⏳ Monitoring robot movement...")

    # Monitor robot movement with position updates
    for i in range(30):
        time.sleep(1)
        current = robot.get_base_pose()
        dist = np.linalg.norm(np.array(current[:2]) - np.array(pos_2d))

        if i % 5 == 0:  # Log every 5 seconds
            logger.info(f"  [{i}s] Position: ({current[0]:.2f}, {current[1]:.2f}), Distance: {dist:.2f}m")

        if dist < 0.5:  # Reached goal
            logger.info(f"✅ Reached goal in {i} seconds!")
            break

    final_pose = robot.get_base_pose()
    distance = np.linalg.norm(np.array(final_pose[:2]) - np.array(pos_2d))
    logger.info(f"Arrived at: ({final_pose[0]:.2f}, {final_pose[1]:.2f})")
    logger.info(f"Distance from goal: {distance:.2f}m")

    # 8. Phase 2: Visual servoing for precision
    if use_visual_servo:
        logger.info("")
        logger.info("PHASE 2: VISUAL SERVOING")
        logger.info("-" * 40)

        success = visual_servo_to_object(robot, semantic_sensor, target_object, timeout=60.0)

        if success:
            logger.info("✅ NAVIGATION COMPLETE - Object centered!")
        else:
            logger.warning("⚠️  Visual servoing incomplete - object may not be perfectly centered")

    logger.info("=" * 70)
    robot.stop()


if __name__ == "__main__":
    main()
