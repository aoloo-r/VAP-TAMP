#!/usr/bin/env python3
"""
Visualization tool to help identify landmarks for calibration.
Shows both Record3D and LIDAR maps side-by-side with coordinate grids.
"""

import numpy as np
import pickle
import cv2
import yaml
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pathlib import Path


def load_pkl_map_for_viz(pkl_path):
    """Load PKL and create top-down 2D projection"""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # Extract camera poses
    camera_poses = [np.array(pose) if not hasattr(pose, 'numpy') else pose.numpy()
                    for pose in data['camera_poses']]
    camera_positions = np.array([pose[:3, 3] for pose in camera_poses])

    # Extract RGB and world XYZ if available
    rgb_frames = data.get('rgb', [])

    print(f"✅ PKL loaded: {len(camera_poses)} frames")
    print(f"   Camera range: X[{camera_positions[:,0].min():.2f}, {camera_positions[:,0].max():.2f}], "
          f"Y[{camera_positions[:,1].min():.2f}, {camera_positions[:,1].max():.2f}]")

    return data, camera_positions


def create_pkl_topdown_view(data, camera_positions, resolution=0.05):
    """
    Create top-down 2D view of Record3D data.
    Projects 3D points onto XY plane.
    """
    # Get bounds
    min_x, max_x = camera_positions[:,0].min(), camera_positions[:,0].max()
    min_y, max_y = camera_positions[:,1].min(), camera_positions[:,1].max()

    # Add margin
    margin = 2.0  # meters
    min_x -= margin
    max_x += margin
    min_y -= margin
    max_y += margin

    # Create grid
    width = int((max_x - min_x) / resolution)
    height = int((max_y - min_y) / resolution)

    grid = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background

    # Plot camera trajectory
    for i in range(len(camera_positions) - 1):
        x1, y1 = camera_positions[i, 0], camera_positions[i, 1]
        x2, y2 = camera_positions[i+1, 0], camera_positions[i+1, 1]

        # Convert to pixel coordinates
        px1 = int((x1 - min_x) / resolution)
        py1 = int((max_y - y1) / resolution)  # Flip Y
        px2 = int((x2 - min_x) / resolution)
        py2 = int((max_y - y2) / resolution)

        # Draw line
        cv2.line(grid, (px1, py1), (px2, py2), (0, 0, 255), 2)

    # Mark start (green) and end (red)
    start_x = int((camera_positions[0, 0] - min_x) / resolution)
    start_y = int((max_y - camera_positions[0, 1]) / resolution)
    end_x = int((camera_positions[-1, 0] - min_x) / resolution)
    end_y = int((max_y - camera_positions[-1, 1]) / resolution)

    cv2.circle(grid, (start_x, start_y), 10, (0, 255, 0), -1)  # Green start
    cv2.circle(grid, (end_x, end_y), 10, (0, 0, 255), -1)      # Red end

    # Create metadata
    metadata = {
        'origin': [min_x, min_y],
        'resolution': resolution,
        'size': [width, height]
    }

    return grid, metadata


def load_lidar_map(map_path, yaml_path):
    """Load LIDAR map and metadata"""
    img = cv2.imread(str(map_path), cv2.IMREAD_GRAYSCALE)

    # Convert to color for visualization
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Invert colors: black=free, white=obstacle, gray=unknown
    # PGM: 255=free, 0=obstacle, 205=unknown
    # We want: white=free, black=obstacle for better visibility
    img_color = 255 - img_color

    with open(yaml_path, 'r') as f:
        metadata = yaml.safe_load(f)

    print(f"✅ LIDAR map loaded: {img.shape[0]}x{img.shape[1]}")
    print(f"   Resolution: {metadata['resolution']}m/pixel")
    print(f"   Origin: {metadata['origin']}")

    return img_color, metadata


def add_coordinate_grid(img, metadata, grid_spacing=5.0, color=(0, 255, 0)):
    """Add coordinate grid overlay to map"""
    img_with_grid = img.copy()

    origin = metadata['origin']
    resolution = metadata['resolution']
    height, width = img.shape[:2]

    # Calculate world bounds
    max_x = origin[0] + width * resolution
    max_y = origin[1] + height * resolution

    # Draw vertical grid lines (X axis)
    x = origin[0]
    while x <= max_x:
        if x >= origin[0]:
            px = int((x - origin[0]) / resolution)
            if 0 <= px < width:
                cv2.line(img_with_grid, (px, 0), (px, height-1), color, 1)
                # Add label
                label = f"{x:.0f}"
                cv2.putText(img_with_grid, label, (px+5, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        x += grid_spacing

    # Draw horizontal grid lines (Y axis)
    y = origin[1]
    while y <= max_y:
        if y >= origin[1]:
            py = int((y - origin[1]) / resolution)
            # Flip Y for image coordinates
            py = height - py
            if 0 <= py < height:
                cv2.line(img_with_grid, (0, py), (width-1, py), color, 1)
                # Add label
                label = f"{y:.0f}"
                cv2.putText(img_with_grid, label, (5, py-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y += grid_spacing

    return img_with_grid


def visualize_maps_side_by_side(pkl_path, lidar_map_path, lidar_yaml_path):
    """Show both maps side by side for landmark identification"""

    print("\n" + "="*80)
    print("MAP VISUALIZATION FOR CALIBRATION")
    print("="*80)

    # Load Record3D data
    pkl_data, camera_positions = load_pkl_map_for_viz(pkl_path)
    r3d_img, r3d_metadata = create_pkl_topdown_view(pkl_data, camera_positions)

    # Load LIDAR map
    lidar_img, lidar_metadata = load_lidar_map(lidar_map_path, lidar_yaml_path)

    # Add coordinate grids
    r3d_with_grid = add_coordinate_grid(r3d_img, r3d_metadata, grid_spacing=5.0, color=(0, 255, 0))
    lidar_with_grid = add_coordinate_grid(lidar_img, lidar_metadata, grid_spacing=5.0, color=(255, 0, 0))

    # Resize to same height for side-by-side display
    target_height = 800
    r3d_aspect = r3d_with_grid.shape[1] / r3d_with_grid.shape[0]
    lidar_aspect = lidar_with_grid.shape[1] / lidar_with_grid.shape[0]

    r3d_resized = cv2.resize(r3d_with_grid, (int(target_height * r3d_aspect), target_height))
    lidar_resized = cv2.resize(lidar_with_grid, (int(target_height * lidar_aspect), target_height))

    # Combine side by side
    combined = np.hstack([r3d_resized, lidar_resized])

    # Add labels
    cv2.putText(combined, "RECORD3D (iPhone) - Green Grid", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.putText(combined, "LIDAR (Robot) - Red Grid", (r3d_resized.shape[1] + 20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)

    # Add instructions
    instructions = [
        "INSTRUCTIONS:",
        "1. Identify the SAME landmarks in BOTH maps",
        "2. Note their coordinates using the grid lines",
        "3. Common landmarks: room corners, doorways, furniture",
        "4. You need at least 3 landmarks for good calibration",
        "",
        "Press 'q' to close and start calibration"
    ]

    y_offset = target_height + 40
    for i, line in enumerate(instructions):
        cv2.putText(combined, line, (20, y_offset + i*30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Display
    window_name = "Map Calibration Helper - Press 'q' to continue"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, combined)

    print("\n📍 LANDMARK IDENTIFICATION TIPS:")
    print("-"*80)
    print("Good landmarks:")
    print("  ✅ Room corners (easy to identify in both maps)")
    print("  ✅ Doorways (distinctive features)")
    print("  ✅ Large furniture (tables, beds, couches)")
    print("  ✅ Wall intersections")
    print("")
    print("Avoid:")
    print("  ❌ Small objects (hard to locate precisely)")
    print("  ❌ Areas with no distinctive features")
    print("  ❌ Objects that might have moved between recordings")
    print("-"*80)
    print("\nRecord3D coordinate system:")
    print(f"  Origin: ({r3d_metadata['origin'][0]:.2f}, {r3d_metadata['origin'][1]:.2f})")
    print(f"  Bounds: X[{r3d_metadata['origin'][0]:.1f}, {r3d_metadata['origin'][0] + r3d_metadata['size'][0]*r3d_metadata['resolution']:.1f}], "
          f"Y[{r3d_metadata['origin'][1]:.1f}, {r3d_metadata['origin'][1] + r3d_metadata['size'][1]*r3d_metadata['resolution']:.1f}]")
    print("\nLIDAR map coordinate system:")
    print(f"  Origin: ({lidar_metadata['origin'][0]:.2f}, {lidar_metadata['origin'][1]:.2f})")
    max_x = lidar_metadata['origin'][0] + lidar_img.shape[1] * lidar_metadata['resolution']
    max_y = lidar_metadata['origin'][1] + lidar_img.shape[0] * lidar_metadata['resolution']
    print(f"  Bounds: X[{lidar_metadata['origin'][0]:.1f}, {max_x:.1f}], "
          f"Y[{lidar_metadata['origin'][1]:.1f}, {max_y:.1f}]")
    print("\n✨ Window opened! Examine both maps and identify landmarks.")
    print("   Press 'q' when ready to enter landmark coordinates.\n")

    # Wait for user
    while True:
        key = cv2.waitKey(100)
        if key == ord('q') or key == 27:  # q or ESC
            break

    cv2.destroyAllWindows()

    return r3d_metadata, lidar_metadata


def main():
    import sys

    # Default paths
    pkl_path = "scripts/visual_grounding_benchmark/sample9_unaligned.pkl"
    lidar_map_path = "maps/multi_room.pgm"
    lidar_yaml_path = "maps/multi_room.yaml"

    if len(sys.argv) > 1:
        pkl_path = sys.argv[1]
    if len(sys.argv) > 2:
        lidar_map_path = sys.argv[2]
    if len(sys.argv) > 3:
        lidar_yaml_path = sys.argv[3]

    # Check files exist
    if not Path(pkl_path).exists():
        print(f"❌ PKL file not found: {pkl_path}")
        return
    if not Path(lidar_map_path).exists():
        print(f"❌ LIDAR map not found: {lidar_map_path}")
        return
    if not Path(lidar_yaml_path).exists():
        print(f"❌ YAML file not found: {lidar_yaml_path}")
        return

    visualize_maps_side_by_side(pkl_path, lidar_map_path, lidar_yaml_path)

    print("\n✅ Visualization complete!")
    print("   Now run: python3 src/stretch/app/calibrate_maps.py")


if __name__ == '__main__':
    main()