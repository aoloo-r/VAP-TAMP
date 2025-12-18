#!/usr/bin/env python3
"""
Multi-point calibration tool for transforming 3D voxel map coordinates to 2D AMCL coordinates.

This tool:
1. Lists object positions from the 3D voxel map
2. Guides you to navigate the robot to those positions
3. Records the actual AMCL positions
4. Calculates the transformation (offset + optional rotation)

Usage:
    python3 calibrate_multi_point.py --map-file maps/your_map.pkl --robot-ip 172.20.10.3
"""

import argparse
import time
import yaml
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Tuple

# Disable Rerun before any imports that might initialize it
os.environ['DISABLE_RERUN'] = '1'

from stretch.agent.robot_agent import RobotAgent
from stretch.agent.zmq_client import HomeRobotZmqClient
from stretch.core.parameters import get_parameters
from stretch.perception import create_semantic_sensor
from stretch.utils.logger import Logger

logger = Logger(__name__)


def calculate_transformation(voxel_points: np.ndarray, amcl_points: np.ndarray) -> Dict:
    """
    Calculate transformation from voxel coordinates to AMCL coordinates.

    Uses least squares to find best-fit offset (and optionally rotation).

    Args:
        voxel_points: Nx2 array of (x, y) in voxel map frame
        amcl_points: Nx2 array of (x, y) in AMCL frame

    Returns:
        Dictionary with transformation parameters
    """
    # Simple translation-only model: AMCL = Voxel + Offset
    offset = np.mean(amcl_points - voxel_points, axis=0)

    # Calculate residuals
    predicted = voxel_points + offset
    residuals = amcl_points - predicted
    rmse = np.sqrt(np.mean(np.sum(residuals**2, axis=1)))

    print(f"\n{'='*70}")
    print(f"TRANSFORMATION RESULTS (Translation-Only Model)")
    print(f"{'='*70}")
    print(f"Offset X: {offset[0]:.3f} m")
    print(f"Offset Y: {offset[1]:.3f} m")
    print(f"RMSE: {rmse:.3f} m")
    print(f"\nPer-point errors:")
    for i, (vox, amcl, res) in enumerate(zip(voxel_points, amcl_points, residuals)):
        error = np.linalg.norm(res)
        print(f"  Point {i+1}: Voxel ({vox[0]:.2f}, {vox[1]:.2f}) "
              f"→ AMCL ({amcl[0]:.2f}, {amcl[1]:.2f}) "
              f"[Error: {error:.3f}m]")
    print(f"{'='*70}\n")

    return {
        'offset_x': float(offset[0]),
        'offset_y': float(offset[1]),
        'rmse': float(rmse),
        'num_points': len(voxel_points)
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-point calibration tool")
    parser.add_argument("--map-file", required=True, help="Path to 3D voxel map (.pkl)")
    parser.add_argument("--config", default="rosbridge_robot_config.yaml", help="Robot config file")
    parser.add_argument("--output", default="multi_point_calibration.yaml", help="Output calibration file")
    parser.add_argument("--num-points", type=int, default=5, help="Number of calibration points (default: 5)")

    args = parser.parse_args()

    print("="*70)
    print("MULTI-POINT CALIBRATION TOOL")
    print("="*70)
    print(f"Map file: {args.map_file}")
    print(f"Calibration points: {args.num_points}")
    print("="*70)

    # 1. Load parameters (no robot connection needed)
    logger.info(f"Loading configuration from {args.config}...")
    parameters = get_parameters(args.config)

    # Disable Rerun visualization
    parameters.data['enable_rerun_server'] = False
    parameters.data['use_rerun_visualizer'] = False

    # Merge with base config if specified
    if parameters.get("vlm_base_config"):
        base_config_file = parameters.get("vlm_base_config")
        base_parameters = get_parameters(base_config_file)
        base_parameters.data.update(parameters.data)
        parameters.data = base_parameters.data
        # Ensure Rerun is still disabled after merge
        parameters.data['enable_rerun_server'] = False
        parameters.data['use_rerun_visualizer'] = False

    # 2. Initialize semantic sensor and load map (using DummyStretchClient - no real robot)
    logger.info("Initializing semantic sensor...")
    semantic_sensor = create_semantic_sensor(parameters=parameters)

    logger.info(f"Loading 3D map from {args.map_file}...")
    # Use DummyStretchClient - no robot connection needed
    from stretch.utils.dummy_stretch_client import DummyStretchClient
    dummy_robot = DummyStretchClient()
    agent = RobotAgent(dummy_robot, parameters, semantic_sensor=semantic_sensor)
    voxel_map = agent.get_voxel_map()

    import matplotlib
    matplotlib.use('Agg')

    voxel_map.read_from_pickle(str(Path(args.map_file)), num_frames=-1, perception=semantic_sensor)
    logger.info(f"✅ Loaded map with {len(voxel_map.instances)} instances")

    # 4. Get instances and select calibration points
    print(f"\n{'='*70}")
    print("AVAILABLE OBJECTS IN 3D MAP:")
    print(f"{'='*70}")

    instances = voxel_map.get_instances()
    instance_list = []

    for i, inst in enumerate(instances):
        if hasattr(inst, 'category_id'):
            try:
                category_name = semantic_sensor.get_class_name_for_id(inst.category_id)
            except:
                category_name = str(inst.category_id)
        else:
            category_name = 'unknown'

        center = inst.get_center()
        if center is not None:
            instance_list.append({
                'id': i,
                'name': category_name,
                'center': center
            })
            print(f"  {i:3d}. {category_name:20s} at ({center[0]:6.2f}, {center[1]:6.2f}, {center[2]:6.2f})")

    # 5. Collect calibration data (manual input - no robot connection)
    print(f"\n{'='*70}")
    print(f"CALIBRATION DATA COLLECTION")
    print(f"{'='*70}")
    print(f"We will collect {args.num_points} calibration points.")
    print(f"For each point:")
    print(f"  1. Select an object from the 3D voxel map above")
    print(f"  2. Navigate the robot to that object's location")
    print(f"  3. On the robot, run: rostopic echo /amcl_pose -n 1")
    print(f"  4. Enter the X and Y coordinates from AMCL")
    print(f"{'='*70}\n")

    voxel_points = []
    amcl_points = []

    for point_num in range(args.num_points):
        print(f"\n{'='*70}")
        print(f"CALIBRATION POINT {point_num + 1}/{args.num_points}")
        print(f"{'='*70}")

        while True:
            try:
                # Get object selection
                choice = input(f"Select object ID (or 'q' to quit): ").strip()

                if choice.lower() == 'q':
                    if point_num < 3:
                        print("❌ Need at least 3 points for calibration")
                        continue
                    else:
                        print(f"Proceeding with {point_num} points...")
                        break

                try:
                    obj_id = int(choice)
                    if obj_id < 0 or obj_id >= len(instance_list):
                        print(f"❌ Invalid ID. Choose 0-{len(instance_list)-1}")
                        continue
                except ValueError:
                    print("❌ Please enter a valid number")
                    continue

                selected = instance_list[obj_id]
                vox_x, vox_y, vox_z = selected['center']

                print(f"\n✅ Selected: {selected['name']}")
                print(f"   Voxel map position: ({vox_x:.3f}, {vox_y:.3f}, {vox_z:.3f})")
                print(f"\n📍 Navigate the robot to this object location")
                print(f"   Then run: rostopic echo /amcl_pose -n 1")
                print(f"   Look for: pose.pose.position.x and pose.pose.position.y")

                # Get manual AMCL input
                amcl_x_str = input(f"\nEnter AMCL X coordinate: ").strip()
                amcl_y_str = input(f"Enter AMCL Y coordinate: ").strip()

                try:
                    amcl_x = float(amcl_x_str)
                    amcl_y = float(amcl_y_str)
                except ValueError:
                    print("❌ Invalid coordinates. Please enter numbers.")
                    continue

                print(f"\n📊 Summary:")
                print(f"   Voxel map: ({vox_x:.3f}, {vox_y:.3f})")
                print(f"   AMCL:      ({amcl_x:.3f}, {amcl_y:.3f})")

                confirm = input("\nIs this correct? [Y/n]: ").strip().lower()
                if confirm in ['', 'y', 'yes']:
                    voxel_points.append([vox_x, vox_y])
                    amcl_points.append([amcl_x, amcl_y])
                    print(f"✅ Point {point_num + 1} recorded")
                    break
                else:
                    print("Let's try this point again...")
                    continue

            except KeyboardInterrupt:
                print("\n\nCalibration cancelled")
                return

        if choice.lower() == 'q':
            break

    if len(voxel_points) < 3:
        print("❌ Need at least 3 points for calibration. Exiting.")
        return

    # 6. Calculate transformation
    voxel_array = np.array(voxel_points)
    amcl_array = np.array(amcl_points)

    transformation = calculate_transformation(voxel_array, amcl_array)

    # 7. Save calibration
    calibration_data = {
        'transformation': transformation,
        'calibration_points': {
            'voxel': [[float(x), float(y)] for x, y in voxel_points],
            'amcl': [[float(x), float(y)] for x, y in amcl_points]
        },
        'map_file': str(args.map_file),
        'config_file': str(args.config)
    }

    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        yaml.dump(calibration_data, f, default_flow_style=False)

    logger.info(f"\n✅ Calibration saved to {output_path}")

    print(f"\n{'='*70}")
    print("USAGE:")
    print(f"{'='*70}")
    print("Use these values in your code:")
    print(f"\noffset_x = {transformation['offset_x']:.3f}")
    print(f"offset_y = {transformation['offset_y']:.3f}")
    print(f"\nAMCL_x = Voxel_x + {transformation['offset_x']:.3f}")
    print(f"AMCL_y = Voxel_y + {transformation['offset_y']:.3f}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
