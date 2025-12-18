#!/usr/bin/env python3
# Complete live navigation system for Segway with 3D map + live camera

"""
Complete workflow:
1. Load 3D voxel map (pkl) - for semantic object positions
2. Connect to live robot via ZMQ - for camera feed and control
3. Calibrate transform between map and robot frames
4. Navigate to semantic objects using:
   - 3D map for object position
   - Transform to robot coordinates
   - move_base for navigation
   - Live camera for visual servoing
"""

import time
import numpy as np
import click
from pathlib import Path
import cv2

from stretch.agent.zmq_client import HomeRobotZmqClient
from stretch.core.parameters import get_parameters
from stretch.mapping.voxel import SparseVoxelMap
from stretch.perception.wrapper import OvmmPerception
from stretch.utils.logger import Logger

logger = Logger(__name__)


class MapTransform:
    """Transform between 3D map and robot 2D coordinates."""

    def __init__(self, offset_x=0.0, offset_y=0.0, rotation=0.0):
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.rotation = rotation

    def transform_3d_to_2d(self, pos_3d):
        """Convert 3D map position [x,y,z] to 2D robot position [x,y]"""
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


def show_camera_feed(robot, window_name="Robot Camera"):
    """Display live camera feed in a window."""
    obs = robot.get_observation()
    if obs is None:
        return False

    # Get RGB and depth
    rgb = obs.rgb
    depth = obs.depth

    # Normalize depth for visualization
    depth_vis = np.clip(depth / 5.0 * 255, 0, 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

    # Combine RGB and depth side by side
    combined = np.hstack([rgb, depth_color])

    # Show in window
    cv2.imshow(window_name, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)
    return True


def visual_servo_to_object(robot, semantic_sensor, target_object, timeout=60.0):
    """Use visual servoing with live camera to center object."""
    logger.info(f"🎯 Visual servoing to '{target_object}'")

    # Parameters
    center_x_threshold = 80
    center_y_threshold = 60
    min_pixels = 100
    max_iterations = 50
    base_step = 0.08

    t0 = time.time()
    iteration = 0

    while time.time() - t0 < timeout and iteration < max_iterations:
        iteration += 1

        # Get live observation
        obs = robot.get_observation()
        if obs is None:
            time.sleep(0.5)
            continue

        # Show live feed
        show_camera_feed(robot, f"Visual Servoing - {target_object}")

        # Run semantic segmentation
        obs = semantic_sensor.predict(obs)

        # Find target object
        img_h, img_w = obs.rgb.shape[:2]
        center_x = img_w // 2
        center_y = img_h // 2

        target_mask = None
        for instance_id in np.unique(obs.semantic):
            class_name = semantic_sensor.get_class_name_for_id(instance_id)
            if class_name and target_object.lower() in class_name.lower():
                target_mask = (obs.semantic == instance_id)
                break

        if target_mask is None or np.sum(target_mask) < min_pixels:
            logger.warning(f"'{target_object}' not visible (iter {iteration})")
            time.sleep(0.3)
            continue

        # Calculate object center in image
        y_indices, x_indices = np.where(target_mask)
        mask_center_x = int(np.mean(x_indices))
        mask_center_y = int(np.mean(y_indices))

        # Calculate offset
        dx = mask_center_x - center_x
        dy = mask_center_y - center_y

        logger.info(f"[{iteration}] Object at ({mask_center_x}, {mask_center_y}), "
                   f"offset=({dx}, {dy})")

        # Check if centered
        if abs(dx) < center_x_threshold and abs(dy) < center_y_threshold:
            logger.info(f"✅ Object centered! Offset: ({dx}, {dy})")
            cv2.destroyAllWindows()
            return True

        # Calculate movement
        move_x = 0.0
        move_y = 0.0

        if dx > center_x_threshold:
            move_y = -base_step  # Move left
        elif dx < -center_x_threshold:
            move_y = base_step   # Move right

        if dy > center_y_threshold:
            move_x = base_step   # Move forward
        elif dy < -center_y_threshold:
            move_x = -base_step  # Move backward

        # Execute movement
        if abs(move_x) > 0 or abs(move_y) > 0:
            logger.info(f"Moving: x={move_x:.2f}, y={move_y:.2f}")
            robot.move_base_to([move_x, move_y, 0.0], relative=True, blocking=True, timeout=5.0)
            time.sleep(0.3)

    cv2.destroyAllWindows()
    logger.error(f"Visual servoing timeout after {iteration} iterations")
    return False


@click.command()
@click.option("--map_file", required=True, help="3D voxel map (.pkl)")
@click.option("--target_object", required=True, help="Object to navigate to (e.g., 'cup')")
@click.option("--offset_x", type=float, default=0.0, help="Map transform X offset")
@click.option("--offset_y", type=float, default=0.0, help="Map transform Y offset")
@click.option("--show_camera", is_flag=True, default=True, help="Show live camera feed")
@click.option("--use_visual_servo", is_flag=True, default=True, help="Use visual servoing")
def main(map_file, target_object, offset_x, offset_y, show_camera, use_visual_servo):
    """Live navigation: 3D map + live camera + move_base + visual servoing."""

    logger.info("=" * 70)
    logger.info("SEGWAY LIVE NAVIGATION SYSTEM")
    logger.info("=" * 70)

    # 1. Setup transform
    transform = MapTransform(offset_x=offset_x, offset_y=offset_y, rotation=0.0)
    logger.info(f"Map transform: offset=({offset_x:.2f}, {offset_y:.2f})")

    # 2. Connect to robot (ZMQ on localhost)
    logger.info("Connecting to robot via ZMQ...")
    parameters = get_parameters("rosbridge_robot_config.yaml")

    robot = HomeRobotZmqClient(
        robot_ip="",  # Uses localhost
        parameters=parameters,
        use_remote_computer=False,  # Local connection
        enable_rerun_server=False,
    )

    logger.info("✅ Connected to robot")

    # Wait for robot to be ready
    logger.info("Waiting for robot data...")
    time.sleep(2)

    # 3. Test camera feed
    logger.info("Testing camera feed...")
    for i in range(10):
        if show_camera_feed(robot, "Live Camera Test"):
            logger.info("✅ Camera feed working")
            break
        time.sleep(0.5)
    else:
        logger.error("❌ No camera feed available")
        return

    time.sleep(2)
    cv2.destroyAllWindows()

    # 4. Load 3D map
    logger.info(f"Loading 3D map from {map_file}...")
    voxel_map = SparseVoxelMap(
        resolution=0.01,
        grid_resolution=0.05,
        use_instance_memory=True
    )
    voxel_map.read_from_pickle(str(Path(map_file)))
    logger.info(f"✅ Loaded map with {len(voxel_map.instances)} instances")

    # 5. Initialize semantic sensor
    logger.info("Initializing semantic sensor...")
    semantic_sensor = OvmmPerception(parameters=parameters)
    logger.info("✅ Semantic sensor ready")

    # 6. Find target in 3D map
    logger.info(f"Searching for '{target_object}' in 3D map...")
    target_instance = None

    for instance_id, instance in voxel_map.instances.items():
        category = str(instance.category_id).lower()
        if target_object.lower() in category:
            target_instance = instance
            logger.info(f"✅ Found '{target_object}' as instance {instance_id}")
            break

    if target_instance is None:
        logger.error(f"❌ '{target_object}' not found in map")
        logger.info("Available objects:")
        for i, (iid, inst) in enumerate(list(voxel_map.instances.items())[:10]):
            logger.info(f"  - {inst.category_id} (ID: {iid})")
        return

    # 7. Get 3D position and transform
    pos_3d = target_instance.get_center()
    logger.info(f"3D position: ({pos_3d[0]:.2f}, {pos_3d[1]:.2f}, {pos_3d[2]:.2f})")

    pos_2d = transform.transform_3d_to_2d(pos_3d)
    logger.info(f"2D position (robot frame): ({pos_2d[0]:.2f}, {pos_2d[1]:.2f})")

    # 8. PHASE 1: Navigate with move_base
    logger.info("")
    logger.info("PHASE 1: ROUGH NAVIGATION")
    logger.info("-" * 40)

    current_pose = robot.get_base_pose()
    logger.info(f"Current: ({current_pose[0]:.2f}, {current_pose[1]:.2f})")

    # Calculate goal orientation (face object)
    goal_theta = np.arctan2(pos_2d[1] - current_pose[1],
                           pos_2d[0] - current_pose[0])

    logger.info(f"Sending navigation goal: ({pos_2d[0]:.2f}, {pos_2d[1]:.2f}, {goal_theta:.2f})")

    # Send goal
    success = robot.navigate_to_goal(pos_2d[0], pos_2d[1], goal_theta)

    if not success:
        logger.error("❌ Failed to send navigation goal")
        return

    # Monitor navigation with live camera
    logger.info("⏳ Navigating... (showing live camera)")
    nav_timeout = 30.0
    t_nav_start = time.time()

    while time.time() - t_nav_start < nav_timeout:
        if show_camera:
            show_camera_feed(robot, "Navigating to Object")

        # Check if arrived (simple distance check)
        current = robot.get_base_pose()
        dist = np.linalg.norm(np.array(current[:2]) - np.array(pos_2d))

        logger.info(f"Distance to goal: {dist:.2f}m")

        if dist < 0.5:  # Within 50cm
            logger.info("✅ Arrived at rough position")
            break

        time.sleep(1.0)

    cv2.destroyAllWindows()

    final_pose = robot.get_base_pose()
    distance = np.linalg.norm(np.array(final_pose[:2]) - np.array(pos_2d))
    logger.info(f"Final position: ({final_pose[0]:.2f}, {final_pose[1]:.2f})")
    logger.info(f"Distance from goal: {distance:.2f}m")

    # 9. PHASE 2: Visual servoing
    if use_visual_servo:
        logger.info("")
        logger.info("PHASE 2: VISUAL SERVOING")
        logger.info("-" * 40)

        success = visual_servo_to_object(robot, semantic_sensor, target_object, timeout=60.0)

        if success:
            logger.info("✅ NAVIGATION COMPLETE - Object centered!")
        else:
            logger.warning("⚠️  Visual servoing incomplete")

    logger.info("=" * 70)
    cv2.destroyAllWindows()
    robot.stop()


if __name__ == "__main__":
    main()
