#!/usr/bin/env python3
"""
Verify calibration by overlaying transformed Record3D trajectory onto LIDAR map.
This visually confirms if the alignment is correct.
"""

import numpy as np
import pickle
import cv2
import yaml
from pathlib import Path


def load_calibration(cal_path='calibration_record3d_to_lidar.yaml'):
    """Load calibration parameters"""
    with open(cal_path, 'r') as f:
        cal = yaml.safe_load(f)
    return cal


def apply_transform(points_r3d, calibration):
    """Apply calibration transform to Record3D points"""
    translation = np.array(calibration['translation'])
    rotation = calibration['rotation']
    scale = calibration['scale']

    # Scale
    points_scaled = points_r3d * scale

    # Rotate
    cos_a = np.cos(rotation)
    sin_a = np.sin(rotation)
    R = np.array([[cos_a, -sin_a],
                  [sin_a,  cos_a]])

    points_rotated = (R @ points_scaled.T).T

    # Translate
    return points_rotated + translation


def world_to_pixel(world_x, world_y, lidar_metadata):
    """Convert world coordinates to pixel coordinates in LIDAR map"""
    origin = lidar_metadata['origin']
    resolution = lidar_metadata['resolution']
    height = int((lidar_metadata.get('height', 1984)))  # Default from multi_room.yaml

    pixel_x = int((world_x - origin[0]) / resolution)
    pixel_y = height - int((world_y - origin[1]) / resolution)  # Flip Y

    return pixel_x, pixel_y


def visualize_alignment(pkl_path, lidar_map_path, lidar_yaml_path, cal_path):
    """
    Visualize the calibration by overlaying Record3D trajectory on LIDAR map.
    """
    print("\n" + "="*80)
    print("CALIBRATION VERIFICATION")
    print("="*80)

    # Load calibration
    try:
        calibration = load_calibration(cal_path)
        print(f"✅ Loaded calibration from: {cal_path}")
        print(f"   Translation: ({calibration['translation'][0]:.3f}, {calibration['translation'][1]:.3f})")
        print(f"   Rotation: {np.degrees(calibration['rotation']):.2f}°")
        print(f"   Scale: {calibration['scale']:.4f}")
        print(f"   Mean error: {calibration.get('mean_error', 'N/A')}m")
    except FileNotFoundError:
        print(f"❌ Calibration file not found: {cal_path}")
        print("   Run calibrate_maps.py first!")
        return

    # Load Record3D data
    with open(pkl_path, 'rb') as f:
        pkl_data = pickle.load(f)

    camera_poses = [np.array(pose) if not hasattr(pose, 'numpy') else pose.numpy()
                    for pose in pkl_data['camera_poses']]
    r3d_positions = np.array([pose[:3, 3] for pose in camera_poses])[:, :2]  # XY only

    print(f"✅ Loaded {len(r3d_positions)} camera positions from PKL")

    # Transform Record3D positions to LIDAR frame
    r3d_transformed = apply_transform(r3d_positions, calibration)
    print(f"✅ Transformed positions to LIDAR frame")

    # Load LIDAR map
    lidar_img = cv2.imread(str(lidar_map_path), cv2.IMREAD_GRAYSCALE)
    lidar_img_color = cv2.cvtColor(lidar_img, cv2.COLOR_GRAY2BGR)

    with open(lidar_yaml_path, 'r') as f:
        lidar_metadata = yaml.safe_load(f)
    lidar_metadata['height'] = lidar_img.shape[0]
    lidar_metadata['width'] = lidar_img.shape[1]

    print(f"✅ Loaded LIDAR map: {lidar_img.shape[0]}x{lidar_img.shape[1]}")

    # Draw transformed trajectory on LIDAR map
    overlay = lidar_img_color.copy()

    # Draw trajectory
    points_in_bounds = 0
    points_out_of_bounds = 0

    for i in range(len(r3d_transformed) - 1):
        x1, y1 = r3d_transformed[i]
        x2, y2 = r3d_transformed[i + 1]

        px1, py1 = world_to_pixel(x1, y1, lidar_metadata)
        px2, py2 = world_to_pixel(x2, y2, lidar_metadata)

        # Check if in bounds
        in_bounds1 = (0 <= px1 < lidar_img.shape[1] and 0 <= py1 < lidar_img.shape[0])
        in_bounds2 = (0 <= px2 < lidar_img.shape[1] and 0 <= py2 < lidar_img.shape[0])

        if in_bounds1 and in_bounds2:
            # Draw line in cyan
            cv2.line(overlay, (px1, py1), (px2, py2), (255, 255, 0), 2)
            points_in_bounds += 1
        else:
            points_out_of_bounds += 1

    # Mark start (green) and end (red)
    start_x, start_y = r3d_transformed[0]
    end_x, end_y = r3d_transformed[-1]

    start_px, start_py = world_to_pixel(start_x, start_y, lidar_metadata)
    end_px, end_py = world_to_pixel(end_x, end_y, lidar_metadata)

    if 0 <= start_px < lidar_img.shape[1] and 0 <= start_py < lidar_img.shape[0]:
        cv2.circle(overlay, (start_px, start_py), 10, (0, 255, 0), -1)  # Green start
        cv2.putText(overlay, "START", (start_px + 15, start_py), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if 0 <= end_px < lidar_img.shape[1] and 0 <= end_py < lidar_img.shape[0]:
        cv2.circle(overlay, (end_px, end_py), 10, (0, 0, 255), -1)  # Red end
        cv2.putText(overlay, "END", (end_px + 15, end_py), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Draw landmark correspondences
    if 'landmarks_record3d' in calibration and 'landmarks_lidar' in calibration:
        r3d_landmarks = np.array(calibration['landmarks_record3d'])
        lidar_landmarks = np.array(calibration['landmarks_lidar'])

        for i, (r3d_lm, lidar_lm) in enumerate(zip(r3d_landmarks, lidar_landmarks)):
            # Transform Record3D landmark
            r3d_transformed_lm = apply_transform(r3d_lm.reshape(1, -1), calibration)[0]

            # Convert to pixels
            r3d_px, r3d_py = world_to_pixel(r3d_transformed_lm[0], r3d_transformed_lm[1], lidar_metadata)
            lidar_px, lidar_py = world_to_pixel(lidar_lm[0], lidar_lm[1], lidar_metadata)

            # Draw transformed Record3D landmark (cyan)
            if 0 <= r3d_px < lidar_img.shape[1] and 0 <= r3d_py < lidar_img.shape[0]:
                cv2.circle(overlay, (r3d_px, r3d_py), 8, (255, 255, 0), 2)
                cv2.putText(overlay, f"R3D-{i+1}", (r3d_px + 12, r3d_py - 12),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # Draw LIDAR landmark (magenta)
            if 0 <= lidar_px < lidar_img.shape[1] and 0 <= lidar_py < lidar_img.shape[0]:
                cv2.circle(overlay, (lidar_px, lidar_py), 8, (255, 0, 255), 2)
                cv2.putText(overlay, f"L-{i+1}", (lidar_px + 12, lidar_py + 12),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

                # Draw line connecting them (should be very short if calibration is good)
                cv2.line(overlay, (r3d_px, r3d_py), (lidar_px, lidar_py), (255, 128, 0), 1)

    # Add statistics
    total_points = points_in_bounds + points_out_of_bounds
    coverage = (points_in_bounds / total_points * 100) if total_points > 0 else 0

    print(f"\n📊 Alignment Statistics:")
    print(f"   Trajectory points in map bounds: {points_in_bounds}/{total_points} ({coverage:.1f}%)")
    print(f"   Points out of bounds: {points_out_of_bounds}")

    # Add legend
    legend_y = 30
    cv2.putText(overlay, "Cyan: Record3D trajectory (transformed)", (10, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(overlay, "Green: START | Red: END", (10, legend_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(overlay, "Cyan circles: R3D landmarks | Magenta: LIDAR landmarks", (10, legend_y + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Assessment
    if coverage < 50:
        status = "❌ POOR - Most trajectory outside map bounds!"
        status_color = (0, 0, 255)
    elif coverage < 80:
        status = "⚠️  MODERATE - Some trajectory outside map"
        status_color = (0, 165, 255)
    else:
        status = "✅ GOOD - Trajectory well aligned"
        status_color = (0, 255, 0)

    cv2.putText(overlay, status, (10, legend_y + 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    print(f"   Status: {status}")

    # Display
    window_name = "Calibration Verification - Press 'q' to close"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, overlay)

    print("\n✨ Verification window opened!")
    print("   - Cyan trajectory should align with room structure in the map")
    print("   - Landmarks (cyan circles) should be very close to magenta circles")
    print("   - START (green) and END (red) should be in valid locations")
    print("\n   Press 'q' to close")

    # Wait for user
    while True:
        key = cv2.waitKey(100)
        if key == ord('q') or key == 27:  # q or ESC
            break

    cv2.destroyAllWindows()

    # Final verdict
    print("\n" + "="*80)
    print("CALIBRATION VERDICT:")
    print("="*80)

    if coverage >= 80 and calibration.get('mean_error', 1.0) < 0.3:
        print("✅ EXCELLENT calibration! Safe to use for navigation.")
    elif coverage >= 60 and calibration.get('mean_error', 1.0) < 0.5:
        print("✅ GOOD calibration. Should work well for navigation.")
    elif coverage >= 40:
        print("⚠️  ACCEPTABLE calibration but may have issues.")
        print("   Recommendation: Add more landmarks or re-calibrate")
    else:
        print("❌ POOR calibration. Do NOT use for navigation!")
        print("   Recommendation: Re-check landmarks and re-calibrate")

    print("="*80)


def main():
    import sys

    pkl_path = "scripts/visual_grounding_benchmark/sample9_unaligned.pkl"
    lidar_map_path = "maps/multi_room.pgm"
    lidar_yaml_path = "maps/multi_room.yaml"
    cal_path = "calibration_record3d_to_lidar.yaml"

    if len(sys.argv) > 1:
        cal_path = sys.argv[1]

    if not Path(cal_path).exists():
        print(f"❌ Calibration file not found: {cal_path}")
        print("\nRun calibration first:")
        print("  1. python3 src/stretch/app/visualize_maps_for_calibration.py")
        print("  2. python3 src/stretch/app/calibrate_maps.py")
        return

    visualize_alignment(pkl_path, lidar_map_path, lidar_yaml_path, cal_path)


if __name__ == '__main__':
    main()