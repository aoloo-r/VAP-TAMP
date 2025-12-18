#!/usr/bin/env python3

import pickle
import numpy as np
import sys
from pathlib import Path

def debug_pkl_file(pkl_path):
    """Debug the contents of a pkl file to understand data structure"""
    
    print(f"=== Debugging PKL file: {pkl_path} ===")
    
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"\nData structure keys: {list(data.keys())}")
        
        # Check each field
        for key, values in data.items():
            print(f"\n{key}:")
            print(f"  Length: {len(values)}")
            
            if len(values) > 0:
                first_val = values[0]
                if first_val is None:
                    print(f"  First value: None")
                elif hasattr(first_val, 'shape'):
                    print(f"  First value shape: {first_val.shape}")
                    print(f"  First value dtype: {first_val.dtype}")
                    
                    if key == 'depth':
                        print(f"  Depth range: [{np.min(first_val):.3f}, {np.max(first_val):.3f}]")
                        valid_depths = first_val[(first_val > 0.1) & (first_val < 10.0)]
                        print(f"  Valid depth pixels: {len(valid_depths)} / {first_val.size}")
                        if len(valid_depths) > 0:
                            print(f"  Valid depth range: [{np.min(valid_depths):.3f}, {np.max(valid_depths):.3f}]")
                    
                    elif key == 'xyz':
                        if first_val is not None:
                            print(f"  XYZ range: x=[{np.min(first_val[:,:,0]):.3f}, {np.max(first_val[:,:,0]):.3f}]")
                            print(f"            y=[{np.min(first_val[:,:,1]):.3f}, {np.max(first_val[:,:,1]):.3f}]")
                            print(f"            z=[{np.min(first_val[:,:,2]):.3f}, {np.max(first_val[:,:,2]):.3f}]")
                    
                    elif key == 'camera_K':
                        print(f"  Camera intrinsics matrix:")
                        print(f"    {first_val}")
                    
                    elif key == 'camera_poses':
                        if first_val is not None:
                            print(f"  Camera pose matrix:")
                            print(f"    {first_val}")
                        else:
                            print(f"  Camera pose: None")
                            
                else:
                    print(f"  First value type: {type(first_val)}")
                    print(f"  First value: {first_val}")
        
        # Check for issues
        print(f"\n=== POTENTIAL ISSUES ===")
        
        depth_values = data.get('depth', [])
        if len(depth_values) > 0 and depth_values[0] is not None:
            depth = depth_values[0]
            valid_depth_count = np.sum((depth > 0.1) & (depth < 10.0))
            total_pixels = depth.size
            valid_percentage = (valid_depth_count / total_pixels) * 100
            
            print(f"Depth validity: {valid_depth_count}/{total_pixels} ({valid_percentage:.1f}%) pixels have valid depth")
            if valid_percentage < 10:
                print("⚠️  WARNING: Very few valid depth pixels! This will result in empty point clouds.")
            
            if np.max(depth) > 1000:
                print("⚠️  WARNING: Depth values seem very large (>1000). Are they in millimeters instead of meters?")
        
        poses = data.get('camera_poses', [])
        if len(poses) > 0 and poses[0] is not None:
            pose = poses[0]
            if np.allclose(pose, np.eye(4)):
                print("⚠️  WARNING: Camera poses are identity matrices. No camera movement will result in no mapping.")
        else:
            print("⚠️  WARNING: No camera poses available.")
        
        xyz_values = data.get('xyz', [])
        if len(xyz_values) > 0 and xyz_values[0] is None:
            print("⚠️  WARNING: XYZ coordinates are None. Point cloud cannot be generated.")
        
        print(f"\n=== SUMMARY ===")
        print(f"Total frames: {len(data.get('rgb', []))}")
        print(f"All fields have data: {all(len(v) == len(data['rgb']) for v in data.values())}")
        
    except Exception as e:
        print(f"Error reading pkl file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python debug_pkl.py <pkl_file>")
        sys.exit(1)
    
    pkl_path = sys.argv[1]
    debug_pkl_file(pkl_path)