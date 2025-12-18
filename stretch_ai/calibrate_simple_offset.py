#!/usr/bin/env python3
"""
Simple offset calibration tool for aligning 3D voxel map with 2D robot map.

This tool helps you find the offset between your 3D voxel map coordinate frame
and your robot's 2D map coordinate frame using a single landmark.

Usage:
    python3 calibrate_simple_offset.py --map-file maps/your_map.pkl
"""

import click
import yaml
import numpy as np
from pathlib import Path

from stretch.agent.robot_agent import RobotAgent
from stretch.core.parameters import get_parameters
from stretch.perception import create_semantic_sensor
from stretch.utils.dummy_stretch_client import DummyStretchClient
from stretch.utils.logger import Logger

logger = Logger(__name__)


@click.command()
@click.option("--map-file", required=True, help="Path to 3D voxel map (.pkl)")
@click.option("--config", default="rosbridge_robot_config.yaml", help="Robot config file")
@click.option("--output", default="simple_offset_calibration.yaml", help="Output calibration file")
def main(map_file, config, output):
    """Interactive tool to calibrate simple offset transformation."""

    logger.info("="*70)
    logger.info("SIMPLE OFFSET CALIBRATION TOOL")
    logger.info("="*70)

    # 1. Load parameters
    logger.info(f"Loading configuration from {config}...")
    parameters = get_parameters(config)

    # Merge with base config if specified
    if parameters.get("vlm_base_config"):
        base_config_file = parameters.get("vlm_base_config")
        base_parameters = get_parameters(base_config_file)
        base_parameters.data.update(parameters.data)
        parameters.data = base_parameters.data

    # 2. Initialize semantic sensor
    logger.info("Initializing semantic sensor...")
    semantic_sensor = create_semantic_sensor(parameters=parameters)

    # 3. Load 3D voxel map
    logger.info(f"Loading 3D map from {map_file}...")
    dummy_robot = DummyStretchClient()
    agent = RobotAgent(dummy_robot, parameters, semantic_sensor=semantic_sensor)
    voxel_map = agent.get_voxel_map()

    # Set matplotlib to non-GUI backend
    import matplotlib
    matplotlib.use('Agg')

    voxel_map.read_from_pickle(str(Path(map_file)), num_frames=-1, perception=semantic_sensor)
    logger.info(f"✅ Loaded map with {len(voxel_map.instances)} instances")

    # 4. Display available objects
    print("\n" + "="*70)
    print("AVAILABLE OBJECTS IN 3D MAP:")
    print("="*70)

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

    # 5. Interactive calibration
    print("\n" + "="*70)
    print("CALIBRATION PROCESS:")
    print("="*70)
    print("1. Look at the 3D object positions above")
    print("2. Find a distinctive landmark you can locate in your robot's map")
    print("3. Get the robot's 2D AMCL position of that same landmark")
    print("   - Use: rostopic echo /amcl_pose -n 1  (on ROS1 robot)")
    print("   - Or: ros2 topic echo /amcl_pose -n 1  (if bridged to ROS2)")
    print("   - Navigate robot to the landmark and note the AMCL position")
    print("="*70)

    while True:
        try:
            # Get 3D landmark selection
            print("\nSelect a landmark from the 3D map:")
            instance_choice = input("Enter instance number (or 'q' to quit): ").strip()

            if instance_choice.lower() == 'q':
                logger.info("Calibration cancelled")
                return

            try:
                instance_idx = int(instance_choice)
                if instance_idx < 0 or instance_idx >= len(instance_list):
                    print(f"❌ Invalid instance number. Choose 0-{len(instance_list)-1}")
                    continue
            except ValueError:
                print("❌ Please enter a valid number")
                continue

            selected = instance_list[instance_idx]
            pos_3d = selected['center']

            print(f"\n✅ Selected: {selected['name']}")
            print(f"   3D position: ({pos_3d[0]:.3f}, {pos_3d[1]:.3f}, {pos_3d[2]:.3f})")

            # Get 2D landmark position from AMCL
            print("\nNow, what is this landmark's position in the robot's AMCL map?")
            print("You can get this by:")
            print("  - Running: rostopic echo /amcl_pose -n 1  (on ROS1 robot)")
            print("  - Or: ros2 topic echo /amcl_pose -n 1  (if bridged)")
            print("  - Navigate the robot to the landmark and note position.pose.position")

            pos_2d_x = input("Enter X coordinate from AMCL: ").strip()
            pos_2d_y = input("Enter Y coordinate from AMCL: ").strip()

            try:
                pos_2d_x = float(pos_2d_x)
                pos_2d_y = float(pos_2d_y)
            except ValueError:
                print("❌ Invalid coordinates. Please enter numbers.")
                continue

            # Calculate offset
            offset_x = pos_2d_x - pos_3d[0]
            offset_y = pos_2d_y - pos_3d[1]

            print("\n" + "="*70)
            print("CALCULATED OFFSET:")
            print("="*70)
            print(f"  offset_x = {offset_x:.3f} meters")
            print(f"  offset_y = {offset_y:.3f} meters")
            print("="*70)

            # Verify with user
            verify = input("\nDoes this look correct? [y/N]: ").strip().lower()
            if verify in ['y', 'yes']:
                break
            else:
                print("Let's try again...")
                continue

        except KeyboardInterrupt:
            print("\n\nCalibration cancelled")
            return

    # 6. Save calibration
    calibration_data = {
        'offset_x': float(offset_x),
        'offset_y': float(offset_y),
        'landmark': {
            '3d_position': [float(pos_3d[0]), float(pos_3d[1]), float(pos_3d[2])],
            '2d_position': [float(pos_2d_x), float(pos_2d_y)],
            'object_name': selected['name'],
            'instance_id': instance_idx
        },
        'map_file': str(map_file),
        'config_file': str(config)
    }

    output_path = Path(output)
    with open(output_path, 'w') as f:
        yaml.dump(calibration_data, f, default_flow_style=False)

    logger.info(f"\n✅ Calibration saved to {output_path}")

    print("\n" + "="*70)
    print("USAGE:")
    print("="*70)
    print("Now you can use this calibration with vlm_planning.py:")
    print(f"\npython3 src/stretch/app/vlm_planning.py \\")
    print(f"  --input-path {map_file} \\")
    print(f"  --offset_x {offset_x:.3f} \\")
    print(f"  --offset_y {offset_y:.3f} \\")
    print(f"  --task 'your task here' \\")
    print(f"  --local")
    print("="*70)

    # Optional: Test the calibration
    print("\nWould you like to test this calibration on another object?")
    test = input("Enter another instance number to test, or press Enter to skip: ").strip()

    if test:
        try:
            test_idx = int(test)
            if 0 <= test_idx < len(instance_list):
                test_obj = instance_list[test_idx]
                test_3d = test_obj['center']
                predicted_2d_x = test_3d[0] + offset_x
                predicted_2d_y = test_3d[1] + offset_y

                print(f"\n🧪 TEST PREDICTION:")
                print(f"   Object: {test_obj['name']}")
                print(f"   3D position: ({test_3d[0]:.3f}, {test_3d[1]:.3f}, {test_3d[2]:.3f})")
                print(f"   Predicted 2D position: ({predicted_2d_x:.3f}, {predicted_2d_y:.3f})")
                print(f"\n   Navigate your robot to verify this matches the actual location!")
        except:
            pass


if __name__ == "__main__":
    main()
