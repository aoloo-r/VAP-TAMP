#!/usr/bin/env python3
"""
Export voxel map to PGM image format for annotation.
"""

import argparse
import yaml
import numpy as np
import cv2
from stretch.mapping.voxel import SparseVoxelMap
from stretch.perception import create_semantic_sensor


def export_map(map_file: str, output_prefix: str):
    """Export voxel map to PGM image and YAML metadata."""

    print(f"📦 Loading map from {map_file}...")

    # Create semantic sensor
    semantic_sensor = create_semantic_sensor(
        device_id=0,
        verbose=False,
        module="detic",
        category_map_file=None,
    )

    # Load voxel map
    voxel_map = SparseVoxelMap(resolution=0.01)

    import matplotlib
    matplotlib.use('Agg')

    voxel_map.read_from_pickle(
        map_file,
        num_frames=-1,
        perception=semantic_sensor
    )

    print("✅ Map loaded")

    # Get 2D occupancy map
    obstacles, explored = voxel_map.get_2d_map()

    # Convert to numpy
    obstacles_np = obstacles.cpu().numpy().astype(np.uint8)
    explored_np = explored.cpu().numpy().astype(np.uint8)

    # Create image: 255=free space, 0=obstacle, 205=unexplored
    image = np.zeros_like(obstacles_np, dtype=np.uint8)
    image[explored_np & ~obstacles_np] = 254  # Free space = white
    image[obstacles_np] = 0  # Obstacles = black
    image[~explored_np] = 205  # Unexplored = gray

    # Get map properties
    origin = voxel_map.grid_origin
    if hasattr(origin, 'cpu'):
        origin = origin.cpu().numpy()
    else:
        origin = np.array(origin)

    resolution = voxel_map.grid_resolution

    # Save as PGM
    pgm_file = f"{output_prefix}.pgm"
    cv2.imwrite(pgm_file, image)
    print(f"💾 Saved image to: {pgm_file}")

    # Save YAML metadata
    yaml_file = f"{output_prefix}.yaml"
    metadata = {
        'image': pgm_file,
        'resolution': float(resolution),
        'origin': [float(origin[0] * resolution), float(origin[1] * resolution), 0.0],
        'negate': 0,
        'occupied_thresh': 0.65,
        'free_thresh': 0.196
    }

    with open(yaml_file, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)

    print(f"💾 Saved metadata to: {yaml_file}")
    print(f"\nImage shape: {image.shape}")
    print(f"Resolution: {resolution}m/pixel")
    print(f"Origin: ({metadata['origin'][0]:.2f}, {metadata['origin'][1]:.2f})")
    print(f"\n✅ Done! Now run:")
    print(f"   python annotate_locations.py --image {pgm_file} --yaml {yaml_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export voxel map to PGM image")
    parser.add_argument("--map-file", required=True, help="Path to voxel map (.pkl)")
    parser.add_argument("--output", default="voxel_map_2d", help="Output file prefix")

    args = parser.parse_args()

    export_map(args.map_file, args.output)
