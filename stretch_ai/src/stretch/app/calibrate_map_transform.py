#!/usr/bin/env python3
# Quick calibration tool for map alignment

"""
Simple tool to calibrate the transform between 3D and 2D maps.

Usage:
1. Find a landmark visible in BOTH maps (door, table corner, etc.)
2. Note its position in 3D map (.pkl file)
3. Drive robot to that landmark
4. Record robot's position from /odom
5. Run this script to calculate the offset
"""

import click
import numpy as np
from pathlib import Path
from stretch.mapping.voxel import SparseVoxelMap
from stretch.core.parameters import get_parameters
from stretch.perception import create_semantic_sensor
from stretch.agent.robot_agent import RobotAgent
from stretch.utils.dummy_stretch_client import DummyStretchClient


@click.command()
@click.option("--map_file", required=True, help="Path to 3D voxel map (.pkl)")
@click.option("--show_instances", is_flag=True, help="Show all instances in map")
def main(map_file, show_instances):
    """Interactive calibration helper."""

    print("=" * 60)
    print("MAP TRANSFORM CALIBRATION HELPER")
    print("=" * 60)
    print()

    # Load parameters (same as vlm_planning - merges base config)
    print(f"Loading 3D map from {map_file}...")
    parameters = get_parameters("rosbridge_robot_config.yaml")

    # Merge with base config if specified (same as vlm_planning does)
    if parameters.get("vlm_base_config"):
        base_config_file = parameters.get("vlm_base_config")
        base_parameters = get_parameters(base_config_file)
        base_parameters.data.update(parameters.data)
        parameters.data = base_parameters.data

    # Create semantic sensor (same as vlm_planning)
    print("Initializing semantic sensor...")
    semantic_sensor = create_semantic_sensor(parameters=parameters)

    # Create dummy robot and agent (same as vlm_planning)
    robot = DummyStretchClient()
    agent = RobotAgent(robot, parameters, semantic_sensor=semantic_sensor)

    # Get voxel map from agent
    voxel_map = agent.get_voxel_map()

    # Set matplotlib to non-GUI backend to avoid Qt errors
    import matplotlib
    matplotlib.use('Agg')  # Non-GUI backend

    # Load from pickle (same as vlm_planning, -1 means load all frames)
    voxel_map.read_from_pickle(str(Path(map_file)), num_frames=-1, perception=semantic_sensor)
    print(f"✅ Loaded map with {len(voxel_map.instances)} instances")
    print()

    # Show instances if requested
    if show_instances:
        print("Instances in 3D map:")
        print("-" * 40)

        # Get instances using the correct method
        all_instances = voxel_map.get_instances()

        for i, instance in enumerate(all_instances):
            center = instance.get_center()
            instance_id = instance.global_id if hasattr(instance, 'global_id') else i

            # Get human-readable category name
            category_name = 'unknown'
            if hasattr(instance, 'category_id'):
                # Try to get the text name from semantic sensor
                try:
                    category_name = semantic_sensor.get_class_name_for_id(instance.category_id)
                    if category_name is None:
                        category_name = f"ID_{instance.category_id}"
                except:
                    category_name = f"ID_{instance.category_id}"

            print(f"{i+1}. {category_name:20s} Position=({center[0]:6.2f}, {center[1]:6.2f}, {center[2]:6.2f})")

            if i >= 19:  # Show first 20
                remaining = len(all_instances) - 20
                if remaining > 0:
                    print(f"... and {remaining} more")
                break
        print()

        # Visualize with Open3D (same as vlm_planning)
        print("Opening 3D visualization with Open3D...")
        print("(Close the visualization window when done to continue)")
        print()
        import open3d

        try:
            # Get geometries (without footprint for simple view)
            geoms = voxel_map._get_open3d_geometries(
                instances=True,
                orig=np.zeros(3),
                xyt=np.zeros(3),
                footprint=None,
                add_planner_visuals=False
            )

            # Create visualizer
            vis = open3d.visualization.Visualizer()
            vis.create_window(window_name="3D Voxel Map - Instances (Close to Continue)", width=800, height=600)

            for geom in geoms:
                vis.add_geometry(geom)

            # Set view
            ctr = vis.get_view_control()
            ctr.set_zoom(0.5)
            ctr.set_front([0.0, 0.0, -1.0])
            ctr.set_up([1.0, 0.0, 0.0])

            # Run visualizer - blocks until window is closed
            vis.run()
            vis.destroy_window()
        except Exception as e:
            print(f"Visualization error: {e}")

        print("Visualization closed. Continuing to calibration...")
        print()

    # Interactive calibration
    print("CALIBRATION STEPS:")
    print("-" * 40)
    print("1. Pick a landmark (door, table corner, large object)")
    print("2. Find its position in the 3D map above")
    print("3. Drive the robot to that same landmark")
    print("4. Check robot position with: rostopic echo /odom -n 1")
    print()

    # Get 3D position
    print("Enter landmark position in 3D map:")
    x_3d = float(input("  X coordinate (3D map): "))
    y_3d = float(input("  Y coordinate (3D map): "))
    print()

    # Get 2D position
    print("Enter landmark position in robot's map (/odom):")
    x_2d = float(input("  X coordinate (robot /odom): "))
    y_2d = float(input("  Y coordinate (robot /odom): "))
    print()

    # Calculate offset
    offset_x = x_2d - x_3d
    offset_y = y_2d - y_3d

    print("=" * 60)
    print("CALIBRATION RESULT:")
    print("-" * 60)
    print(f"Offset X: {offset_x:.3f} meters")
    print(f"Offset Y: {offset_y:.3f} meters")
    print()
    print("Use these values with segway_hybrid_navigation.py:")
    print()
    print(f"  python app/segway_hybrid_navigation.py \\")
    print(f"    --map_file {map_file} \\")
    print(f"    --target_object cup \\")
    print(f"    --offset_x {offset_x:.3f} \\")
    print(f"    --offset_y {offset_y:.3f}")
    print()
    print("OR use landmark-based calibration:")
    print()
    print(f"  python app/segway_hybrid_navigation.py \\")
    print(f"    --map_file {map_file} \\")
    print(f"    --target_object cup \\")
    print(f"    --landmark_3d '{x_3d},{y_3d}' \\")
    print(f"    --landmark_2d '{x_2d},{y_2d}'")
    print()
    print("=" * 60)

    # Test conversion
    print()
    print("TEST: Convert some positions from 3D to 2D:")
    print("-" * 60)

    test_points = [
        [0, 0],
        [1, 0],
        [0, 1],
        [x_3d, y_3d],  # The landmark itself
    ]

    for px, py in test_points:
        rx = px + offset_x
        ry = py + offset_y
        print(f"  3D map [{px:.1f}, {py:.1f}] → Robot map [{rx:.2f}, {ry:.2f}]")

    print("=" * 60)


if __name__ == "__main__":
    main()
