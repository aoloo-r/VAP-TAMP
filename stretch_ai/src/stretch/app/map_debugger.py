#!/usr/bin/env python3
# Map Debugging and Repair Tool - Fix empty point cloud and visualization issues

import pickle
import numpy as np
import click
import os
import sys

try:
    import open3d as o3d
except ImportError:
    print("Warning: Open3D not available for visualization")
    o3d = None

def analyze_pickle_file(filepath):
    """Analyze the structure and content of a pickle map file"""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {filepath}")
    print(f"{'='*60}")
    
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        print(f"✓ File loaded successfully")
        print(f"✓ File size: {os.path.getsize(filepath) / 1024 / 1024:.2f} MB")
        
        # Analyze top-level structure
        if isinstance(data, dict):
            print(f"✓ Data type: Dictionary with {len(data)} keys")
            print(f"✓ Keys: {list(data.keys())}")
            
            # Look for common map data structures
            point_fields = ['points', '_points', 'point_cloud', 'xyz']
            voxel_fields = ['voxels', '_voxels', 'occupied_voxels']
            obs_fields = ['observations', '_observations', 'obs_history']
            
            for field in point_fields:
                if field in data:
                    points = data[field]
                    if points is not None:
                        print(f"✓ Found points in '{field}': {len(points) if hasattr(points, '__len__') else 'unknown length'}")
                        if hasattr(points, 'shape'):
                            print(f"  Shape: {points.shape}")
                        if hasattr(points, 'dtype'):
                            print(f"  Dtype: {points.dtype}")
                    else:
                        print(f"⚠ Found '{field}' but it's None")
            
            for field in voxel_fields:
                if field in data:
                    voxels = data[field]
                    if voxels is not None:
                        print(f"✓ Found voxels in '{field}': {len(voxels) if hasattr(voxels, '__len__') else 'unknown length'}")
                    else:
                        print(f"⚠ Found '{field}' but it's None")
                        
            for field in obs_fields:
                if field in data:
                    obs = data[field]
                    if obs is not None:
                        print(f"✓ Found observations in '{field}': {len(obs) if hasattr(obs, '__len__') else 'unknown length'}")
                    else:
                        print(f"⚠ Found '{field}' but it's None")
        
        elif hasattr(data, '__dict__'):
            print(f"✓ Data type: Object of class {type(data).__name__}")
            attrs = [attr for attr in dir(data) if not attr.startswith('_')]
            print(f"✓ Attributes: {attrs[:10]}{'...' if len(attrs) > 10 else ''}")
            
            # Check for point cloud data in object
            point_attrs = ['points', '_points', 'point_cloud', 'xyz']
            for attr in point_attrs:
                if hasattr(data, attr):
                    points = getattr(data, attr)
                    if points is not None:
                        print(f"✓ Found points in attribute '{attr}': {len(points) if hasattr(points, '__len__') else 'unknown length'}")
                        if hasattr(points, 'shape'):
                            print(f"  Shape: {points.shape}")
                    else:
                        print(f"⚠ Found attribute '{attr}' but it's None")
        
        else:
            print(f"✓ Data type: {type(data)}")
            if hasattr(data, '__len__'):
                print(f"✓ Length: {len(data)}")
        
        return data
        
    except Exception as e:
        print(f"✗ Error loading file: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_point_cloud(data):
    """Extract point cloud data from various map formats"""
    points = None
    colors = None
    
    # Strategy 1: Direct point access
    if isinstance(data, dict):
        point_fields = ['points', '_points', 'point_cloud', 'xyz']
        color_fields = ['colors', '_colors', 'rgb']
        
        for field in point_fields:
            if field in data and data[field] is not None:
                candidate = data[field]
                if hasattr(candidate, 'shape') and len(candidate.shape) >= 2 and candidate.shape[1] >= 3:
                    points = candidate
                    print(f"✓ Extracted {len(points)} points from '{field}'")
                    break
        
        for field in color_fields:
            if field in data and data[field] is not None:
                colors = data[field]
                print(f"✓ Found colors in '{field}'")
                break
    
    # Strategy 2: Object attribute access
    elif hasattr(data, '__dict__'):
        point_attrs = ['points', '_points', 'point_cloud', 'xyz']
        for attr in point_attrs:
            if hasattr(data, attr):
                candidate = getattr(data, attr)
                if candidate is not None and hasattr(candidate, 'shape') and len(candidate.shape) >= 2:
                    points = candidate
                    print(f"✓ Extracted {len(points)} points from attribute '{attr}'")
                    break
        
        # Try to get colors
        color_attrs = ['colors', '_colors', 'rgb']
        for attr in color_attrs:
            if hasattr(data, attr):
                colors = getattr(data, attr)
                if colors is not None:
                    print(f"✓ Found colors in attribute '{attr}'")
                    break
    
    # Strategy 3: Voxel-based extraction
    if points is None:
        print("⚠ No direct point cloud found, trying to extract from voxels...")
        # Try to extract from voxel data structures
        voxel_data = None
        if isinstance(data, dict):
            for field in ['voxels', '_voxels', 'occupied_voxels']:
                if field in data:
                    voxel_data = data[field]
                    break
        elif hasattr(data, '__dict__'):
            for attr in ['voxels', '_voxels', 'occupied_voxels']:
                if hasattr(data, attr):
                    voxel_data = getattr(data, attr)
                    break
        
        if voxel_data is not None:
            # Try to convert voxels to points
            try:
                if hasattr(voxel_data, 'keys'):
                    # Dictionary-like voxel structure
                    voxel_coords = list(voxel_data.keys())
                    if voxel_coords and len(voxel_coords[0]) == 3:
                        points = np.array(voxel_coords)
                        print(f"✓ Extracted {len(points)} points from voxel coordinates")
            except Exception as e:
                print(f"⚠ Failed to extract from voxels: {e}")
    
    return points, colors

def create_visualization_safe_map(points, colors=None, output_file=None):
    """Create a visualization-safe version of the map"""
    if points is None:
        print("✗ No points available for visualization")
        return False
    
    try:
        # Ensure points are in the right format
        if not isinstance(points, np.ndarray):
            points = np.array(points)
        
        # Ensure we have 3D points
        if points.shape[1] < 3:
            print(f"✗ Points have insufficient dimensions: {points.shape}")
            return False
        
        # Take only first 3 dimensions if more are present
        points = points[:, :3].astype(np.float64)
        
        # Remove any invalid points
        valid_mask = np.all(np.isfinite(points), axis=1)
        valid_points = points[valid_mask]
        
        if len(valid_points) == 0:
            print("✗ No valid points after filtering")
            return False
        
        print(f"✓ Processed {len(valid_points)} valid points")
        
        if o3d is not None:
            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(valid_points)
            
            # Add colors if available
            if colors is not None:
                try:
                    if not isinstance(colors, np.ndarray):
                        colors = np.array(colors)
                    
                    if colors.shape[0] == len(points):
                        valid_colors = colors[valid_mask]
                        if valid_colors.shape[1] >= 3:
                            # Normalize colors to [0, 1] if needed
                            if valid_colors.max() > 1.0:
                                valid_colors = valid_colors / 255.0
                            pcd.colors = o3d.utility.Vector3dVector(valid_colors[:, :3])
                            print("✓ Added colors to point cloud")
                except Exception as e:
                    print(f"⚠ Failed to add colors: {e}")
            
            # Save as PLY for easy visualization
            if output_file:
                ply_file = output_file.replace('.pkl', '_fixed.ply')
                o3d.io.write_point_cloud(ply_file, pcd)
                print(f"✓ Saved visualization-safe PLY to: {ply_file}")
            
            # Try visualization
            print("✓ Attempting visualization...")
            try:
                o3d.visualization.draw_geometries([pcd])
                return True
            except Exception as e:
                print(f"⚠ Visualization failed: {e}")
                return False
        else:
            print("⚠ Open3D not available, cannot create visualization")
            return False
            
    except Exception as e:
        print(f"✗ Error creating visualization: {e}")
        import traceback
        traceback.print_exc()
        return False

def repair_map_file(input_file, output_file=None):
    """Attempt to repair a corrupted map file"""
    print(f"\n{'='*60}")
    print(f"REPAIRING: {input_file}")
    print(f"{'='*60}")
    
    # Analyze the file
    data = analyze_pickle_file(input_file)
    if data is None:
        return False
    
    # Extract point cloud
    points, colors = extract_point_cloud(data)
    
    if points is None:
        print("✗ No repairable point cloud data found")
        return False
    
    # Create a minimal working map structure
    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_repaired{ext}"
    
    try:
        # Create a simple map structure that should work with visualization
        repaired_data = {
            'points': points,
            'num_points': len(points),
            'bounds': {
                'min': np.min(points, axis=0) if len(points) > 0 else np.array([0, 0, 0]),
                'max': np.max(points, axis=0) if len(points) > 0 else np.array([1, 1, 1])
            }
        }
        
        if colors is not None:
            repaired_data['colors'] = colors
        
        # Save repaired data
        with open(output_file, 'wb') as f:
            pickle.dump(repaired_data, f)
        
        print(f"✓ Saved repaired map to: {output_file}")
        
        # Try to visualize the repaired map
        create_visualization_safe_map(points, colors, output_file)
        
        return True
        
    except Exception as e:
        print(f"✗ Error repairing map: {e}")
        import traceback
        traceback.print_exc()
        return False

@click.command()
@click.option("--file", "-f", required=True, help="Map file to analyze/repair")
@click.option("--repair", is_flag=True, help="Attempt to repair the map file")
@click.option("--visualize", is_flag=True, help="Attempt to visualize the map")
@click.option("--output", "-o", help="Output file for repaired map")
def main(file, repair, visualize, output):
    """Map debugging and repair tool for Stretch AI mapping files"""
    
    if not os.path.exists(file):
        print(f"✗ File not found: {file}")
        sys.exit(1)
    
    # Always analyze first
    data = analyze_pickle_file(file)
    if data is None:
        sys.exit(1)
    
    # Extract and analyze point cloud
    points, colors = extract_point_cloud(data)
    
    if points is not None:
        print(f"\n✓ Point cloud summary:")
        print(f"  Total points: {len(points)}")
        print(f"  Shape: {points.shape}")
        print(f"  Bounds: min={np.min(points, axis=0)}, max={np.max(points, axis=0)}")
        
        if colors is not None:
            print(f"  Colors: {colors.shape}")
    else:
        print(f"\n✗ No point cloud data found")
    
    # Repair if requested
    if repair:
        if repair_map_file(file, output):
            print("\n✓ Repair completed successfully")
        else:
            print("\n✗ Repair failed")
            sys.exit(1)
    
    # Visualize if requested
    if visualize:
        if points is not None:
            if create_visualization_safe_map(points, colors):
                print("\n✓ Visualization completed")
            else:
                print("\n✗ Visualization failed")
        else:
            print("\n✗ No points available for visualization")

if __name__ == "__main__":
    main()