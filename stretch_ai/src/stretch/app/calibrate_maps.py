#!/usr/bin/env python3
"""
Calibrate coordinate transformation between Record3D PKL map and LIDAR 2D map.
Uses manual landmark matching to compute transform.
"""

import numpy as np
import pickle
from pathlib import Path
import cv2
import yaml


def load_pkl_map(pkl_path):
    """Load Record3D PKL file and extract camera poses"""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # Get camera trajectory bounds
    camera_poses = [np.array(pose) for pose in data['camera_poses']]
    positions = np.array([pose[:3, 3] for pose in camera_poses])

    print(f"✅ Loaded PKL with {len(camera_poses)} frames")
    print(f"   Camera range: X[{positions[:,0].min():.2f}, {positions[:,0].max():.2f}], "
          f"Y[{positions[:,1].min():.2f}, {positions[:,1].max():.2f}]")

    return data, positions


def load_lidar_map(map_path, yaml_path):
    """Load LIDAR PGM map and metadata"""
    # Load map image
    img = cv2.imread(str(map_path), cv2.IMREAD_GRAYSCALE)

    # Load metadata
    with open(yaml_path, 'r') as f:
        metadata = yaml.safe_load(f)

    print(f"✅ Loaded LIDAR map: {img.shape[0]}x{img.shape[1]}")
    print(f"   Resolution: {metadata['resolution']}m/pixel")
    print(f"   Origin: {metadata['origin']}")

    return img, metadata


def pixel_to_world(pixel_x, pixel_y, metadata):
    """Convert LIDAR map pixel to world coordinates"""
    resolution = metadata['resolution']
    origin = metadata['origin']

    world_x = origin[0] + pixel_x * resolution
    world_y = origin[1] + (metadata['image'].replace('.pgm', '').replace('.png', '')) # Get height from img
    # Actually need image height
    return world_x, world_y


def compute_transform_from_correspondences(record3d_points, lidar_points):
    """
    Compute transformation from Record3D frame to LIDAR frame.

    Args:
        record3d_points: Nx2 array of points in Record3D frame
        lidar_points: Nx2 array of corresponding points in LIDAR frame

    Returns:
        translation: [tx, ty]
        rotation: angle in radians
        scale: scaling factor
    """
    assert len(record3d_points) == len(lidar_points) >= 2

    # Center the points
    r3d_center = record3d_points.mean(axis=0)
    lidar_center = lidar_points.mean(axis=0)

    r3d_centered = record3d_points - r3d_center
    lidar_centered = lidar_points - lidar_center

    # Compute scale
    r3d_scale = np.sqrt((r3d_centered ** 2).sum())
    lidar_scale = np.sqrt((lidar_centered ** 2).sum())
    scale = lidar_scale / r3d_scale if r3d_scale > 0 else 1.0

    # Compute rotation using SVD
    H = r3d_centered.T @ lidar_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Extract angle
    angle = np.arctan2(R[1, 0], R[0, 0])

    # Compute translation
    translation = lidar_center - scale * (R @ r3d_center)

    return translation, angle, scale


def apply_transform(point_r3d, translation, rotation, scale):
    """Apply transformation to convert Record3D point to LIDAR frame"""
    # Scale
    point_scaled = point_r3d * scale

    # Rotate
    cos_a = np.cos(rotation)
    sin_a = np.sin(rotation)
    R = np.array([[cos_a, -sin_a],
                  [sin_a,  cos_a]])
    point_rotated = R @ point_scaled

    # Translate
    point_lidar = point_rotated + translation

    return point_lidar


def interactive_calibration(pkl_path, map_path, yaml_path, visualize_first=True):
    """
    Interactive calibration workflow.
    User provides corresponding landmarks in both frames.
    """
    print("\n" + "="*60)
    print("RECORD3D ↔ LIDAR MAP CALIBRATION")
    print("="*60)

    # Load data
    pkl_data, r3d_positions = load_pkl_map(pkl_path)
    lidar_img, lidar_metadata = load_lidar_map(map_path, yaml_path)

    # Show visualization helper if requested
    if visualize_first:
        print("\n💡 TIP: Run visualization tool first to identify landmarks:")
        print("   python3 src/stretch/app/visualize_maps_for_calibration.py")
        proceed = input("\nHave you identified landmarks? [y/N]: ").strip().lower()
        if proceed not in ['y', 'yes']:
            print("Please run the visualization tool first!")
            return

    print("\n📍 LANDMARK MATCHING")
    print("-"*60)
    print("Identify the SAME physical locations in both maps.")
    print("You need at least 3 landmarks (4+ recommended for best accuracy).")
    print("\nGood landmarks:")
    print("  ✅ Room corners (easy to identify)")
    print("  ✅ Doorways (distinctive features)")
    print("  ✅ Large furniture corners")
    print("  ✅ Wall intersections")
    print("\nAvoid:")
    print("  ❌ Small/movable objects")
    print("  ❌ Featureless areas")
    print("-"*60)

    # Collect correspondences
    record3d_landmarks = []
    lidar_landmarks = []

    print("\nFor each landmark, provide coordinates in BOTH frames:")

    while True:
        print(f"\n--- Landmark {len(record3d_landmarks) + 1} ---")

        # Record3D coordinates
        print("Record3D frame (from PKL):")
        try:
            r3d_x = float(input("  X coordinate (meters): "))
            r3d_y = float(input("  Y coordinate (meters): "))

            # LIDAR coordinates
            print("LIDAR map frame:")
            lidar_x = float(input("  X coordinate (meters): "))
            lidar_y = float(input("  Y coordinate (meters): "))

            record3d_landmarks.append([r3d_x, r3d_y])
            lidar_landmarks.append([lidar_x, lidar_y])

            print(f"✅ Added landmark: R3D({r3d_x:.2f}, {r3d_y:.2f}) → LIDAR({lidar_x:.2f}, {lidar_y:.2f})")

        except (ValueError, EOFError):
            print("Invalid input, skipping...")
            continue

        if len(record3d_landmarks) >= 2:
            more = input("\nAdd another landmark? [y/N]: ").strip().lower()
            if more not in ['y', 'yes']:
                break

    if len(record3d_landmarks) < 3:
        print("❌ Need at least 3 landmarks for robust calibration!")
        print("   (2 landmarks only give translation+rotation, no scale verification)")
        use_anyway = input("Continue with only 2 landmarks? [y/N]: ").strip().lower()
        if use_anyway not in ['y', 'yes']:
            return

    # Compute transformation
    r3d_points = np.array(record3d_landmarks)
    lidar_points = np.array(lidar_landmarks)

    translation, rotation, scale = compute_transform_from_correspondences(r3d_points, lidar_points)

    print("\n" + "="*60)
    print("🎯 CALIBRATION RESULTS")
    print("="*60)
    print(f"Translation: ({translation[0]:.6f}, {translation[1]:.6f}) meters")
    print(f"Rotation: {np.degrees(rotation):.3f}° ({rotation:.6f} radians)")
    print(f"Scale: {scale:.6f}")

    # Compute error
    errors = []
    print("\n📊 Verification (reprojection error):")
    for i, (r3d, lidar_true) in enumerate(zip(r3d_points, lidar_points)):
        lidar_pred = apply_transform(r3d, translation, rotation, scale)
        error = np.linalg.norm(lidar_pred - lidar_true)
        errors.append(error)
        print(f"  Landmark {i+1}: error = {error:.4f}m")

    mean_error = np.mean(errors)
    max_error = np.max(errors)
    print(f"\nMean error: {mean_error:.4f}m")
    print(f"Max error:  {max_error:.4f}m")

    # Quality assessment
    if mean_error > 1.0:
        print("❌ POOR calibration quality! Landmarks may be incorrect.")
        print("   Recommendation: Re-check landmark coordinates")
    elif mean_error > 0.5:
        print("⚠️  MODERATE calibration quality. May work but could be better.")
        print("   Recommendation: Add more landmarks or verify existing ones")
    elif mean_error > 0.2:
        print("✅ GOOD calibration quality!")
    else:
        print("✅ EXCELLENT calibration quality!")

    # Check if scale is reasonable (should be close to 1.0 for real-world data)
    if scale < 0.5 or scale > 2.0:
        print(f"\n⚠️  WARNING: Unusual scale factor ({scale:.3f})")
        print("   Expected: close to 1.0 for real-world measurements")
        print("   This might indicate coordinate system mismatch")

    # Save calibration
    calibration = {
        'translation': translation.tolist(),
        'rotation': float(rotation),
        'scale': float(scale),
        'landmarks_record3d': record3d_landmarks,
        'landmarks_lidar': lidar_landmarks,
        'mean_error': float(mean_error)
    }

    output_path = 'calibration_record3d_to_lidar.yaml'
    with open(output_path, 'w') as f:
        yaml.dump(calibration, f)

    print(f"\n💾 Saved calibration to: {output_path}")

    # Show usage
    print("\n" + "="*60)
    print("📝 HOW TO USE THIS CALIBRATION:")
    print("="*60)
    print("In your navigation code:")
    print()
    print("```python")
    print("import yaml")
    print("import numpy as np")
    print()
    print("# Load calibration")
    print(f"with open('{output_path}', 'r') as f:")
    print("    cal = yaml.safe_load(f)")
    print()
    print("# Transform Record3D point to LIDAR frame")
    print("def r3d_to_lidar(r3d_x, r3d_y):")
    print(f"    translation = np.array(cal['translation'])")
    print(f"    rotation = cal['rotation']")
    print(f"    scale = cal['scale']")
    print()
    print("    # Apply transform")
    print("    point = np.array([r3d_x, r3d_y]) * scale")
    print("    cos_a, sin_a = np.cos(rotation), np.sin(rotation)")
    print("    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])")
    print("    point_rotated = R @ point")
    print("    return point_rotated + translation")
    print()
    print("# Example usage")
    print("lidar_x, lidar_y = r3d_to_lidar(instance_center[0], instance_center[1])")
    print("```")
    print("="*60)


if __name__ == '__main__':
    import sys

    # Default paths
    pkl_path = "scripts/visual_grounding_benchmark/sample9_unaligned.pkl"
    map_path = "maps/multi_room.pgm"
    yaml_path = "maps/multi_room.yaml"

    # Allow override from command line
    if len(sys.argv) > 1:
        pkl_path = sys.argv[1]
    if len(sys.argv) > 2:
        map_path = sys.argv[2]
    if len(sys.argv) > 3:
        yaml_path = sys.argv[3]

    interactive_calibration(pkl_path, map_path, yaml_path)