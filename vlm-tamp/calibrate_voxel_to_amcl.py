#!/usr/bin/env python3
"""
Voxel Map to AMCL Calibration Tool

This tool helps you calibrate the transformation between:
- Voxel map coordinates (from Record3D or other 3D mapping)
- AMCL map coordinates (from LIDAR-based localization)

Usage:
    python calibrate_voxel_to_amcl.py --map-file your_map.pkl

How it works:
1. Load the voxel map
2. Display object instances from the map
3. For each landmark object:
   - You provide the voxel map coordinates (from the instance)
   - You provide the AMCL coordinates (from the robot's LIDAR map)
4. Calculate the offset transformation (offset_x, offset_y)
5. Save to simple_offset_calibration.yaml

Recommended: Use 3-5 landmark objects spread across the environment
"""

import argparse
import numpy as np
import yaml
import sys
from pathlib import Path

# Add Stretch AI to path
stretch_ai_path = Path(__file__).parent.parent / "stretch_ai" / "src"
if str(stretch_ai_path) not in sys.path:
    sys.path.insert(0, str(stretch_ai_path))

from stretch.agent.robot_agent import RobotAgent
from stretch.agent.zmq_client import HomeRobotZmqClient
from stretch.perception import create_semantic_sensor
from stretch.core.parameters import get_parameters


def load_voxel_map(map_file: str, config_file: str = "rosbridge_robot_config.yaml"):
    """Load voxel map and list available object instances."""
    print(f"📦 Loading voxel map from {map_file}...")

    # Load parameters
    parameters = get_parameters(config_file)
    if parameters.get("vlm_base_config"):
        base_config = get_parameters(parameters.get("vlm_base_config"))
        base_config.data.update(parameters.data)
        parameters.data = base_config.data

    # Create semantic sensor
    semantic_sensor = create_semantic_sensor(parameters=parameters)

    # Create dummy robot client (not connected)
    robot = HomeRobotZmqClient(
        robot_ip="",
        parameters=parameters,
        use_remote_computer=False,
        enable_rerun_server=False,
    )

    # Create agent
    agent = RobotAgent(
        robot,
        parameters,
        semantic_sensor=semantic_sensor
    )

    # Load map
    voxel_map = agent.get_voxel_map()

    import matplotlib
    matplotlib.use('Agg')

    voxel_map.read_from_pickle(map_file, num_frames=-1, perception=None)

    print(f"✅ Loaded voxel map with {len(voxel_map.instances)} instances")

    return voxel_map


def display_instances(voxel_map):
    """Display all object instances in the voxel map."""
    print(f"\n{'='*80}")
    print("OBJECT INSTANCES IN VOXEL MAP")
    print(f"{'='*80}\n")

    instances = []

    for idx, (instance_id, instance) in enumerate(voxel_map.instances.items()):
        center = instance.get_center()
        category_id = instance.category_id

        # Get category name
        category_name = "unknown"
        if hasattr(instance, 'category_name'):
            category_name = instance.category_name
        elif hasattr(voxel_map, 'semantic_sensor') and voxel_map.semantic_sensor:
            try:
                category_name = voxel_map.semantic_sensor.get_class_name_for_id(category_id)
            except:
                pass

        print(f"[{idx}] Instance {instance_id} - {category_name}")
        print(f"    Voxel map center: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
        print(f"    Points: {len(instance.point_cloud)}")
        print()

        instances.append({
            'id': instance_id,
            'idx': idx,
            'category': category_name,
            'center': center
        })

    return instances


def collect_calibration_points():
    """Interactively collect calibration point pairs."""
    print(f"\n{'='*80}")
    print("CALIBRATION POINT COLLECTION")
    print(f"{'='*80}\n")

    print("For accurate calibration, select 3-5 landmark objects that:")
    print("  1. Are spread across different areas of the environment")
    print("  2. Are easily identifiable in both maps")
    print("  3. Have well-defined positions (corners, furniture, etc.)")
    print()

    calibration_points = []

    while True:
        print(f"\n--- Calibration Point {len(calibration_points) + 1} ---\n")

        # Get voxel map coordinates
        print("Enter voxel map coordinates (from the instance list above):")
        try:
            voxel_x = float(input("  Voxel X: "))
            voxel_y = float(input("  Voxel Y: "))
        except (ValueError, EOFError):
            print("\n❌ Invalid input")
            continue

        # Get AMCL coordinates
        print("\nEnter AMCL coordinates (from your LIDAR map):")
        try:
            amcl_x = float(input("  AMCL X: "))
            amcl_y = float(input("  AMCL Y: "))
        except (ValueError, EOFError):
            print("\n❌ Invalid input")
            continue

        calibration_points.append({
            'voxel': (voxel_x, voxel_y),
            'amcl': (amcl_x, amcl_y)
        })

        print(f"\n✅ Added calibration point:")
        print(f"   Voxel: ({voxel_x:.3f}, {voxel_y:.3f}) → AMCL: ({amcl_x:.3f}, {amcl_y:.3f})")

        if len(calibration_points) >= 3:
            choice = input(f"\nYou have {len(calibration_points)} points. Add more? (y/n): ").strip().lower()
            if choice != 'y':
                break
        elif len(calibration_points) >= 2:
            choice = input(f"\nYou have {len(calibration_points)} points (min 2). Add more? (y/n): ").strip().lower()
            if choice != 'y':
                break

    return calibration_points


def calculate_offset(calibration_points):
    """Calculate offset transformation using least squares."""
    print(f"\n{'='*80}")
    print("CALCULATING TRANSFORMATION")
    print(f"{'='*80}\n")

    # Extract coordinates
    voxel_coords = np.array([p['voxel'] for p in calibration_points])
    amcl_coords = np.array([p['amcl'] for p in calibration_points])

    # Calculate offset: AMCL = Voxel + Offset
    # offset = AMCL - Voxel
    offsets = amcl_coords - voxel_coords

    # Use mean offset
    offset_x = float(np.mean(offsets[:, 0]))
    offset_y = float(np.mean(offsets[:, 1]))

    # Calculate error statistics
    std_x = float(np.std(offsets[:, 0]))
    std_y = float(np.std(offsets[:, 1]))

    print(f"Calculated offset:")
    print(f"  offset_x = {offset_x:.6f} ± {std_x:.6f} meters")
    print(f"  offset_y = {offset_y:.6f} ± {std_y:.6f} meters")
    print()

    # Verify transformation
    print("Verification (applying offset to voxel coordinates):")
    print(f"{'Point':<8} {'Voxel X':<12} {'Voxel Y':<12} {'→ AMCL X':<12} {'→ AMCL Y':<12} {'Error (m)':<12}")
    print("-" * 80)

    errors = []
    for i, point in enumerate(calibration_points):
        voxel_x, voxel_y = point['voxel']
        amcl_x_true, amcl_y_true = point['amcl']

        amcl_x_calc = voxel_x + offset_x
        amcl_y_calc = voxel_y + offset_y

        error = np.sqrt((amcl_x_calc - amcl_x_true)**2 + (amcl_y_calc - amcl_y_true)**2)
        errors.append(error)

        print(f"{i+1:<8} {voxel_x:<12.3f} {voxel_y:<12.3f} {amcl_x_calc:<12.3f} {amcl_y_calc:<12.3f} {error:<12.4f}")

    print(f"\nMean error: {np.mean(errors):.4f} meters")
    print(f"Max error:  {np.max(errors):.4f} meters")

    if np.mean(errors) > 0.5:
        print("\n⚠️  WARNING: High calibration error! Check your coordinate inputs.")
    elif np.mean(errors) > 0.2:
        print("\n⚠️  Calibration error is moderate. Consider adding more points.")
    else:
        print("\n✅ Good calibration quality!")

    return offset_x, offset_y


def save_calibration(offset_x, offset_y, output_file="simple_offset_calibration.yaml"):
    """Save calibration to YAML file."""
    calibration_data = {
        'offset_x': float(offset_x),
        'offset_y': float(offset_y),
        'description': 'Transformation from voxel map coordinates to AMCL map coordinates',
        'formula': 'amcl_coord = voxel_coord + offset'
    }

    with open(output_file, 'w') as f:
        yaml.dump(calibration_data, f, default_flow_style=False)

    print(f"\n💾 Saved calibration to: {output_file}")
    print(f"\n{'='*80}")
    print("CALIBRATION COMPLETE")
    print(f"{'='*80}\n")
    print(f"Transformation: AMCL_coord = Voxel_coord + Offset")
    print(f"  offset_x = {offset_x:.6f}")
    print(f"  offset_y = {offset_y:.6f}")
    print(f"\nUse this calibration file with eval_real_robot.py:")
    print(f"  --calibration {output_file}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Calibrate voxel map to AMCL coordinates")
    parser.add_argument("--map-file", required=True, help="Path to voxel map (.pkl)")
    parser.add_argument("--config", default="rosbridge_robot_config.yaml", help="Robot config")
    parser.add_argument("--output", default="simple_offset_calibration.yaml", help="Output calibration file")

    args = parser.parse_args()

    print("""
================================================================================
VOXEL MAP TO AMCL CALIBRATION TOOL
================================================================================

This tool helps you calibrate the coordinate transformation between:
  - Voxel map coordinates (from 3D mapping)
  - AMCL coordinates (from LIDAR-based localization)

You'll need to identify 3-5 landmark objects that appear in both maps.

================================================================================
""")

    # Load voxel map
    voxel_map = load_voxel_map(args.map_file, args.config)

    # Display instances
    instances = display_instances(voxel_map)

    if not instances:
        print("❌ No instances found in voxel map!")
        return

    # Collect calibration points
    calibration_points = collect_calibration_points()

    if len(calibration_points) < 2:
        print("\n❌ Need at least 2 calibration points!")
        return

    # Calculate offset
    offset_x, offset_y = calculate_offset(calibration_points)

    # Save calibration
    save_calibration(offset_x, offset_y, args.output)


if __name__ == "__main__":
    main()
