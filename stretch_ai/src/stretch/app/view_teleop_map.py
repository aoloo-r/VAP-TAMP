#!/usr/bin/env python
import click
import numpy as np
import open3d as o3d
from pathlib import Path
import pickle
from stretch.mapping.voxel import SparseVoxelMap
from stretch.perception.encoders import get_encoder
from stretch.utils.point_cloud import numpy_to_pcd

@click.command()
@click.option("--map-file", "-i", default="teleop_map.pkl", help="Input map file")
@click.option("--show-3d", is_flag=True, help="Show 3D point cloud")
@click.option("--show-2d", is_flag=True, help="Show 2D map")
@click.option("--export-ply", is_flag=True, help="Export to PLY format")
@click.option("--export-pcd", is_flag=True, help="Export to PCD format")
def view_teleop_map(map_file: str, show_3d: bool, show_2d: bool, 
                   export_ply: bool, export_pcd: bool):
    """View teleoperation map without full robot agent"""
    
    print(f"Loading map from {map_file}...")
    
    # Load the map pickle file
    with open(map_file, "rb") as f:
        data = pickle.load(f)
    
    # Extract map info
    if "parameters" in data:
        voxel_size = data["parameters"].get("voxel_size", 0.05)
        print(f"Voxel size: {voxel_size}")
    
    if "encoder" in data:
        encoder_type = data["encoder"]
        print(f"Encoder type: {encoder_type}")
    
    # Create a minimal encoder
    encoder = get_encoder("siglip", {})
    
    # Create voxel map and load data
    voxel_map = SparseVoxelMap(
        resolution=data.get("resolution", 0.05),
        feature_dim=encoder.feature_dim,
        encoder=encoder,
        use_instance_memory=False,
        device="cpu",  # Use CPU for viewing
    )
    
    # Load the actual map data
    print("Loading map data...")
    voxel_map.read_from_pickle(map_file)
    
    # Get statistics
    points, _, _, rgb = voxel_map.get_pointcloud()
    if points is not None:
        num_points = len(points)
        print(f"Total points: {num_points}")
        
        # Get bounds
        points_np = points.detach().cpu().numpy()
        min_bounds = points_np.min(axis=0)
        max_bounds = points_np.max(axis=0)
        size = max_bounds - min_bounds
        print(f"Map bounds: [{min_bounds[0]:.2f}, {min_bounds[1]:.2f}, {min_bounds[2]:.2f}] to [{max_bounds[0]:.2f}, {max_bounds[1]:.2f}, {max_bounds[2]:.2f}]")
        print(f"Map size: {size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f} meters")
        
        # Show 3D visualization
        if show_3d:
            print("\nShowing 3D point cloud...")
            pcd = numpy_to_pcd(points_np, (rgb.detach().cpu().numpy() / 255.0))
            
            # Add coordinate frame
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
            
            # Visualize
            o3d.visualization.draw_geometries([pcd, coord_frame], window_name="Teleoperation Map")
        
        # Show 2D map
        if show_2d:
            print("\nGenerating 2D map...")
            import matplotlib.pyplot as plt
            
            obstacles, explored = voxel_map.get_2d_map()
            obstacles_np = obstacles.cpu().numpy()
            explored_np = explored.cpu().numpy()
            
            # Create visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            
            # Explored map
            ax1.imshow(explored_np, cmap='gray', origin='lower')
            ax1.set_title('Explored Area')
            ax1.set_xlabel('X (grid cells)')
            ax1.set_ylabel('Y (grid cells)')
            
            # Obstacles map
            ax2.imshow(obstacles_np, cmap='gray', origin='lower')
            ax2.set_title('Obstacles')
            ax2.set_xlabel('X (grid cells)')
            ax2.set_ylabel('Y (grid cells)')
            
            plt.tight_layout()
            plt.show()
        
        # Export options
        base_name = Path(map_file).stem
        
        if export_ply:
            ply_file = f"{base_name}.ply"
            print(f"\nExporting to {ply_file}...")
            pcd = numpy_to_pcd(points_np, (rgb.detach().cpu().numpy() / 255.0))
            o3d.io.write_point_cloud(ply_file, pcd)
            print(f"Exported {num_points} points to {ply_file}")
        
        if export_pcd:
            pcd_file = f"{base_name}.pcd"
            print(f"\nExporting to {pcd_file}...")
            pcd = numpy_to_pcd(points_np, (rgb.detach().cpu().numpy() / 255.0))
            o3d.io.write_point_cloud(pcd_file, pcd)
            print(f"Exported {num_points} points to {pcd_file}")
    else:
        print("No points found in the map!")

if __name__ == "__main__":
    view_teleop_map()