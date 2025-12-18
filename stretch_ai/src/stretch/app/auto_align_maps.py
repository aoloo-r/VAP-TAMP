#!/usr/bin/env python3
"""
Automatic map alignment using 2D projection from 3D voxel map.
Projects PKL voxel map to 2D, then uses image registration to align with LIDAR map.
"""

import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'  # Headless mode
os.environ['MPLBACKEND'] = 'Agg'  # Matplotlib headless

import numpy as np
import pickle
import cv2
import yaml
from pathlib import Path
from scipy.ndimage import rotate
from skimage.feature import match_template
from skimage.transform import resize


def load_pkl_and_generate_2d_map(pkl_path, resolution=0.05):
    """
    Load PKL file and generate 2D occupancy grid using the full voxel map with obstacles.
    This properly loads semantic instances and obstacle data.
    """
    print(f"📦 Loading PKL file: {pkl_path}")

    # We need to load it properly through vlm_planning's infrastructure
    # Import the necessary modules
    from stretch.agent import RobotAgent
    from stretch.core import get_parameters
    from stretch.perception import create_semantic_sensor
    from stretch.utils.dummy_stretch_client import DummyStretchClient

    # Load parameters
    print("⚙️  Loading configuration...")
    try:
        vlm_parameters = get_parameters("default_planner.yaml")
    except:
        print("❌ Could not load default_planner.yaml")
        raise

    print("🔍 Creating semantic sensor...")
    semantic_sensor = create_semantic_sensor(parameters=vlm_parameters)

    print("🗺️  Creating voxel map directly...")
    from stretch.mapping.voxel.voxel_map import SparseVoxelMap

    voxel_map = SparseVoxelMap(
        resolution=vlm_parameters.get("voxel_size", 0.05),
        local_radius=vlm_parameters.get("local_radius", 0.8),
        obs_min_height=vlm_parameters.get("obs_min_height", 0.05),
        obs_max_height=vlm_parameters.get("obs_max_height", 2.5),
        obs_min_density=vlm_parameters.get("obs_min_density", 5),
        pad_obstacles=vlm_parameters.get("pad_obstacles", 2),
    )

    # Load from pickle
    print(f"🔄 Reading voxel map from pickle (with perception)...")
    voxel_map.read_from_pickle(pkl_path, num_frames=-1, perception=semantic_sensor)

    print(f"✅ Loaded voxel map:")
    print(f"   Observations: {len(voxel_map.observations)}")
    print(f"   Instances: {len(voxel_map.get_instances())}")

    # FIX: Transform 3D coordinates before projection
    # Record3D may have different coordinate system orientation
    print("🔄 Checking 3D coordinate system orientation...")

    # Get the point cloud
    xyz, rgb, counts, _ = voxel_map.voxel_pcd.get_pointcloud()

    if xyz is not None and xyz.nelement() > 0:
        print(f"   3D points: {xyz.shape}")
        print(f"   X range: [{xyz[:, 0].min():.2f}, {xyz[:, 0].max():.2f}]")
        print(f"   Y range: [{xyz[:, 1].min():.2f}, {xyz[:, 1].max():.2f}]")
        print(f"   Z range: [{xyz[:, 2].min():.2f}, {xyz[:, 2].max():.2f}]")

        # Try different coordinate transformations
        # Option 1: Flip Z axis (bottom to top view)
        # Option 2: Swap axes if needed

        print("\n🔧 Applying coordinate transformation...")
        print("   Using: Flip X axis (correct Record3D orientation)")

        # Flip X coordinates (correct orientation for Record3D)
        import torch
        transformed_points = torch.stack([
            -voxel_map.voxel_pcd._points[:, 0],  # Flip X
            voxel_map.voxel_pcd._points[:, 1],   # Keep Y
            voxel_map.voxel_pcd._points[:, 2]    # Keep Z
        ], dim=1)

        voxel_map.voxel_pcd._points = transformed_points

        print(f"   ✅ Transformed coordinates:")
        print(f"      X: [{voxel_map.voxel_pcd._points[:, 0].min():.2f}, {voxel_map.voxel_pcd._points[:, 0].max():.2f}]")
        print(f"      Y: [{voxel_map.voxel_pcd._points[:, 1].min():.2f}, {voxel_map.voxel_pcd._points[:, 1].max():.2f}]")
        print(f"      Z: [{voxel_map.voxel_pcd._points[:, 2].min():.2f}, {voxel_map.voxel_pcd._points[:, 2].max():.2f}]")

    # Generate 2D map from voxel map (this includes obstacles!)
    print("🗺️  Generating 2D occupancy grid from 3D voxels...")
    obstacles, explored = voxel_map.get_2d_map()

    if obstacles is None or explored is None:
        raise ValueError("Failed to generate 2D map from voxel map")

    # Convert to numpy arrays
    obstacles_np = obstacles.cpu().numpy() if hasattr(obstacles, 'cpu') else np.array(obstacles)
    explored_np = explored.cpu().numpy() if hasattr(explored, 'cpu') else np.array(explored)

    # Create occupancy grid image
    # IMPORTANT: Match LIDAR convention
    # 255 = free space (white)
    # 127 = unknown (gray)
    # 0 = obstacle (black)
    height, width = obstacles_np.shape
    occupancy_grid = np.ones((height, width), dtype=np.uint8) * 127  # Unknown

    # Mark explored free space as WHITE (to match LIDAR)
    free_space = explored_np & (~obstacles_np)
    occupancy_grid[free_space] = 255  # White = free

    # Mark obstacles as BLACK (to match LIDAR)
    occupancy_grid[obstacles_np] = 0  # Black = obstacle

    obstacle_count = np.sum(obstacles_np)
    free_count = np.sum(free_space)
    print(f"✅ Generated 2D occupancy grid:")
    print(f"   Size: {width}x{height}")
    print(f"   Resolution: {voxel_map.grid_resolution:.4f}m/pixel")
    print(f"   Obstacles: {obstacle_count} cells")
    print(f"   Free space: {free_count} cells")
    print(f"   Coverage: {(obstacle_count + free_count) / (width * height) * 100:.1f}%")

    # Save the generated 2D map for visual comparison
    import cv2
    output_pgm = 'record3d_2d_map.pgm'
    output_yaml = 'record3d_2d_map.yaml'

    cv2.imwrite(output_pgm, occupancy_grid)
    print(f"💾 Saved 2D occupancy grid to: {output_pgm}")

    # Also save rotated versions for comparison
    rotated_90 = cv2.rotate(occupancy_grid, cv2.ROTATE_90_CLOCKWISE)
    rotated_180 = cv2.rotate(occupancy_grid, cv2.ROTATE_180)
    rotated_270 = cv2.rotate(occupancy_grid, cv2.ROTATE_90_COUNTERCLOCKWISE)
    flipped_h = cv2.flip(occupancy_grid, 1)
    flipped_v = cv2.flip(occupancy_grid, 0)

    cv2.imwrite('record3d_2d_map_rot90.pgm', rotated_90)
    cv2.imwrite('record3d_2d_map_rot180.pgm', rotated_180)
    cv2.imwrite('record3d_2d_map_rot270.pgm', rotated_270)
    cv2.imwrite('record3d_2d_map_flipped_h.pgm', flipped_h)
    cv2.imwrite('record3d_2d_map_flipped_v.pgm', flipped_v)
    print(f"💾 Saved rotated versions for comparison:")
    print(f"   - record3d_2d_map_rot90.pgm (90° clockwise)")
    print(f"   - record3d_2d_map_rot180.pgm (180°)")
    print(f"   - record3d_2d_map_rot270.pgm (270° clockwise)")
    print(f"   - record3d_2d_map_flipped_h.pgm (horizontal flip)")
    print(f"   - record3d_2d_map_flipped_v.pgm (vertical flip)")

    # Convert grid_origin from grid cells to world meters
    # grid_origin is in grid cells, need to convert to meters
    # Origin in meters = -grid_origin * resolution (since grid_origin is the offset in cells)
    origin_meters_x = -float(voxel_map.grid_origin[0]) * float(voxel_map.grid_resolution)
    origin_meters_y = -float(voxel_map.grid_origin[1]) * float(voxel_map.grid_resolution)

    # Save metadata YAML
    map_metadata = {
        'image': output_pgm,
        'resolution': float(voxel_map.grid_resolution),
        'origin': [origin_meters_x, origin_meters_y, 0.0],
        'negate': 0,
        'occupied_thresh': 0.65,
        'free_thresh': 0.196,
        # Save grid_origin for coordinate conversion
        'grid_origin': [float(voxel_map.grid_origin[0]), float(voxel_map.grid_origin[1])]
    }

    with open(output_yaml, 'w') as f:
        yaml.dump(map_metadata, f)
    print(f"💾 Saved metadata to: {output_yaml}")
    print(f"   Origin (world meters): ({origin_meters_x:.2f}, {origin_meters_y:.2f})")

    # Get metadata from voxel map
    metadata = {
        'resolution': float(voxel_map.grid_resolution),
        'origin': [origin_meters_x, origin_meters_y, 0.0],
        'size': [width, height]
    }

    return occupancy_grid, metadata, voxel_map


def load_lidar_map(map_path, yaml_path):
    """Load LIDAR PGM map and metadata"""
    img = cv2.imread(str(map_path), cv2.IMREAD_GRAYSCALE)

    with open(yaml_path, 'r') as f:
        metadata = yaml.safe_load(f)

    print(f"✅ Loaded LIDAR map: {img.shape[1]}x{img.shape[0]} @ {metadata['resolution']}m/pixel")

    return img, metadata


def normalize_map_for_registration(img):
    """
    Normalize map for image registration.
    Convert to binary: obstacle=1, free=0, unknown=0
    """
    normalized = np.zeros_like(img, dtype=np.float32)

    # Threshold: high values = obstacles
    normalized[img > 200] = 1.0  # Obstacles

    return normalized


def find_best_alignment_multiscale(source_map, target_map, source_meta, target_meta,
                                   angle_range=(-180, 180), angle_step=5):
    """
    Find best alignment using multi-scale phase correlation and brute-force rotation search.

    Returns:
        translation: (tx, ty) in pixels of target map
        rotation: angle in radians
        scale: scaling factor
        confidence: match confidence (0-1)
    """
    print("\n🔍 Searching for optimal alignment...")
    print(f"   Angle range: {angle_range[0]}° to {angle_range[1]}° (step: {angle_step}°)")

    # Normalize maps
    source_norm = normalize_map_for_registration(source_map)
    target_norm = normalize_map_for_registration(target_map)

    # Resize source to match target resolution
    source_res = source_meta['resolution']
    target_res = target_meta['resolution']
    scale_factor = target_res / source_res

    print(f"   Source resolution: {source_res:.4f}m/px")
    print(f"   Target resolution: {target_res:.4f}m/px")
    print(f"   Scale factor: {scale_factor:.4f}")

    new_height = int(source_norm.shape[0] * scale_factor)
    new_width = int(source_norm.shape[1] * scale_factor)
    source_scaled = cv2.resize(source_norm, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    print(f"   Scaled source: {source_scaled.shape[1]}x{source_scaled.shape[0]}")
    print(f"   Target: {target_norm.shape[1]}x{target_norm.shape[0]}")

    # Brute force rotation search
    best_score = -np.inf
    best_rotation = 0
    best_translation = (0, 0)

    angles = range(angle_range[0], angle_range[1], angle_step)

    for angle in angles:
        # Rotate source
        rotated = rotate(source_scaled, angle, reshape=True, order=1, mode='constant', cval=0)

        # Make sure rotated fits in target
        if rotated.shape[0] > target_norm.shape[0] or rotated.shape[1] > target_norm.shape[1]:
            continue

        # Template matching to find translation
        result = match_template(target_norm, rotated, pad_input=True)

        # Find peak
        max_score = result.max()
        max_loc = np.unravel_index(result.argmax(), result.shape)

        if max_score > best_score:
            best_score = max_score
            best_rotation = angle
            best_translation = max_loc

        if angle % 45 == 0:  # Progress update
            print(f"   Angle {angle:4d}°: score={max_score:.4f}")

    print(f"\n✅ Best match found:")
    print(f"   Rotation: {best_rotation}°")
    print(f"   Translation (pixels): {best_translation}")
    print(f"   Confidence: {best_score:.4f}")

    # Convert rotation to radians
    rotation_rad = np.radians(best_rotation)

    return best_translation, rotation_rad, scale_factor, float(best_score)


def compute_world_transform(pixel_translation, rotation, scale,
                           source_meta, target_meta, source_shape, target_shape):
    """
    Convert pixel-space transformation to world-space transformation.

    Returns:
        translation: [tx, ty] in world coordinates (meters)
        rotation: angle in radians
        scale: scaling factor
    """
    # Pixel translation is in target map coordinates
    pixel_tx, pixel_ty = pixel_translation

    # Convert to world coordinates
    target_res = target_meta['resolution']
    target_origin = target_meta['origin']

    # Translation in world frame
    world_tx = target_origin[0] + pixel_tx * target_res
    world_ty = target_origin[1] + (target_shape[0] - pixel_ty) * target_res  # Flip Y

    # Source origin
    source_origin = source_meta['origin']

    # Final transformation from source world to target world
    # T_target = R * S * (p_source - origin_source) + [world_tx, world_ty]
    # Simplifying: translation = [world_tx, world_ty] - R * S * origin_source

    cos_a = np.cos(rotation)
    sin_a = np.sin(rotation)
    R = np.array([[cos_a, -sin_a],
                  [sin_a,  cos_a]])

    source_origin_vec = np.array(source_origin[:2])
    translation = np.array([world_tx, world_ty]) - scale * (R @ source_origin_vec)

    return translation, rotation, scale


def save_calibration(translation, rotation, scale, confidence, output_path):
    """Save calibration to YAML file"""
    calibration = {
        'translation': translation.tolist(),
        'rotation': float(rotation),
        'scale': float(scale),
        'confidence': float(confidence),
        'method': 'automatic_image_registration',
        'mean_error': 'N/A (automatic alignment)'
    }

    with open(output_path, 'w') as f:
        yaml.dump(calibration, f)

    print(f"\n💾 Saved calibration to: {output_path}")


def visualize_alignment(source_map, target_map, translation, rotation, scale,
                       source_meta, target_meta):
    """Save alignment visualization to file (headless mode)"""
    print("\n🎨 Creating alignment visualization...")

    # Resize and rotate source to match target
    source_res = source_meta['resolution']
    target_res = target_meta['resolution']
    scale_factor = target_res / source_res

    new_height = int(source_map.shape[0] * scale_factor)
    new_width = int(source_map.shape[1] * scale_factor)
    source_scaled = cv2.resize(source_map, (new_width, new_height))

    # Rotate
    angle_deg = np.degrees(rotation)
    source_rotated = rotate(source_scaled, angle_deg, reshape=True, order=1, mode='constant', cval=127)

    # Create overlay
    target_color = cv2.cvtColor(target_map, cv2.COLOR_GRAY2BGR)
    overlay = target_color.copy()

    # Apply translation and composite
    pixel_tx = int((translation[0] - target_meta['origin'][0]) / target_res)
    pixel_ty = int(target_map.shape[0] - (translation[1] - target_meta['origin'][1]) / target_res)

    # Draw source in cyan
    for y in range(source_rotated.shape[0]):
        for x in range(source_rotated.shape[1]):
            ty = pixel_ty + y
            tx = pixel_tx + x

            if 0 <= tx < overlay.shape[1] and 0 <= ty < overlay.shape[0]:
                if source_rotated[y, x] > 200:  # Obstacle in source
                    overlay[ty, tx] = (255, 255, 0)  # Cyan

    # Save to file instead of showing
    output_path = 'alignment_visualization.png'
    cv2.imwrite(output_path, overlay)
    print(f"✅ Visualization saved to: {output_path}")
    print("   (Cyan=PKL obstacles, White/Gray=LIDAR map)")


def main():
    import sys

    # Default paths
    pkl_path = "scripts/visual_grounding_benchmark/sample9_unaligned.pkl"
    lidar_map_path = "maps/multi_room.pgm"
    lidar_yaml_path = "maps/multi_room.yaml"
    output_path = "calibration_record3d_to_lidar.yaml"

    if len(sys.argv) > 1:
        pkl_path = sys.argv[1]

    print("="*80)
    print("AUTOMATIC MAP ALIGNMENT")
    print("="*80)
    print(f"PKL file: {pkl_path}")
    print(f"LIDAR map: {lidar_map_path}")
    print("="*80)

    # Step 1: Generate 2D map from PKL
    voxel_2d_map, voxel_metadata, voxel_map = load_pkl_and_generate_2d_map(pkl_path)

    # Step 2: Load LIDAR map
    lidar_map, lidar_metadata = load_lidar_map(lidar_map_path, lidar_yaml_path)

    # Step 3: Find alignment
    pixel_translation, rotation, scale, confidence = find_best_alignment_multiscale(
        voxel_2d_map, lidar_map, voxel_metadata, lidar_metadata
    )

    # Step 4: Compute world transformation
    translation, rotation, scale = compute_world_transform(
        pixel_translation, rotation, scale,
        voxel_metadata, lidar_metadata,
        voxel_2d_map.shape, lidar_map.shape
    )

    # Step 5: Save calibration
    save_calibration(translation, rotation, scale, confidence, output_path)

    print("\n" + "="*80)
    print("🎯 CALIBRATION RESULTS")
    print("="*80)
    print(f"Translation: ({translation[0]:.6f}, {translation[1]:.6f}) meters")
    print(f"Rotation: {np.degrees(rotation):.3f}° ({rotation:.6f} radians)")
    print(f"Scale: {scale:.6f}")
    print(f"Confidence: {confidence:.4f}")
    print("="*80)

    # Step 6: Visualize
    visualize_alignment(voxel_2d_map, lidar_map, translation, rotation, scale,
                       voxel_metadata, lidar_metadata)

    print("\n✅ Automatic alignment complete!")
    print(f"   Run verification: python3 src/stretch/app/verify_calibration.py")
    print(f"   Then use for navigation with vlm_planning.py")


if __name__ == '__main__':
    main()