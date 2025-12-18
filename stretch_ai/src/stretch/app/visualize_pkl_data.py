#!/usr/bin/env python3

import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
from pathlib import Path

def visualize_pkl_data(pkl_path, frame_idx=0):
    """Visualize the contents of a pkl file to debug issues"""
    
    print(f"=== Visualizing PKL file: {pkl_path} ===")
    
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Total frames: {len(data.get('rgb', []))}")
        
        if frame_idx >= len(data['rgb']):
            print(f"Frame {frame_idx} not available, using frame 0")
            frame_idx = 0
        
        # Get data for specific frame
        rgb = data['rgb'][frame_idx]
        depth = data['depth'][frame_idx] 
        camera_K = data['camera_K'][frame_idx]
        pose = data['camera_poses'][frame_idx]
        xyz = data['xyz'][frame_idx] if data['xyz'][frame_idx] is not None else None
        
        print(f"\n=== Frame {frame_idx} Analysis ===")
        print(f"RGB shape: {rgb.shape}, dtype: {rgb.dtype}")
        print(f"RGB range: [{np.min(rgb)}, {np.max(rgb)}]")
        
        print(f"Depth shape: {depth.shape}, dtype: {depth.dtype}")
        print(f"Depth range: [{np.min(depth):.3f}, {np.max(depth):.3f}]")
        
        # Check depth validity
        valid_depth = depth[(depth > 0.1) & (depth < 10.0)]
        print(f"Valid depth pixels: {len(valid_depth)}/{depth.size} ({100*len(valid_depth)/depth.size:.1f}%)")
        if len(valid_depth) > 0:
            print(f"Valid depth range: [{np.min(valid_depth):.3f}, {np.max(valid_depth):.3f}]")
        
        print(f"Camera intrinsics:\n{camera_K}")
        print(f"Camera pose:\n{pose}")
        
        if xyz is not None:
            print(f"XYZ shape: {xyz.shape}")
            print(f"XYZ ranges:")
            print(f"  X: [{np.min(xyz[:,:,0]):.3f}, {np.max(xyz[:,:,0]):.3f}]")
            print(f"  Y: [{np.min(xyz[:,:,1]):.3f}, {np.max(xyz[:,:,1]):.3f}]") 
            print(f"  Z: [{np.min(xyz[:,:,2]):.3f}, {np.max(xyz[:,:,2]):.3f}]")
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # RGB image
        if hasattr(rgb, 'numpy'):  # PyTorch tensor
            rgb_np = rgb.numpy()
        else:
            rgb_np = rgb
        axes[0,0].imshow(rgb_np)
        axes[0,0].set_title(f'RGB Frame {frame_idx}')
        axes[0,0].axis('off')
        
        # Depth image
        if hasattr(depth, 'numpy'):  # PyTorch tensor
            depth_np = depth.numpy()
        else:
            depth_np = depth
        depth_vis = axes[0,1].imshow(depth_np, cmap='viridis')
        axes[0,1].set_title(f'Depth Frame {frame_idx} (meters)')
        axes[0,1].axis('off')
        plt.colorbar(depth_vis, ax=axes[0,1])
        
        # Depth histogram
        valid_depths = depth_np[(depth_np > 0.01) & (depth_np < 10.0)].flatten()
        if len(valid_depths) > 0:
            axes[0,2].hist(valid_depths, bins=50, alpha=0.7)
            axes[0,2].set_title('Valid Depth Distribution')
            axes[0,2].set_xlabel('Depth (meters)')
            axes[0,2].set_ylabel('Count')
        else:
            axes[0,2].text(0.5, 0.5, 'No valid depth data', ha='center', va='center')
            axes[0,2].set_title('Depth Histogram')
        
        # Point cloud projection (XY view)
        if xyz is not None:
            # Sample some points for visualization
            h, w = xyz.shape[:2]
            step = max(1, h//100)  # Sample ~10k points max
            
            xyz_sample = xyz[::step, ::step].reshape(-1, 3)
            rgb_sample = rgb_np[::step, ::step].reshape(-1, 3) / 255.0
            
            # Filter valid points
            valid_mask = (xyz_sample[:, 2] > 0.1) & (xyz_sample[:, 2] < 10.0)
            if np.any(valid_mask):
                xyz_valid = xyz_sample[valid_mask]
                rgb_valid = rgb_sample[valid_mask]
                
                # XY view (top-down)
                scatter = axes[1,0].scatter(xyz_valid[:, 0], xyz_valid[:, 1], 
                                         c=rgb_valid, s=1, alpha=0.6)
                axes[1,0].set_title('Point Cloud - XY View (Top Down)')
                axes[1,0].set_xlabel('X (meters)')
                axes[1,0].set_ylabel('Y (meters)')
                axes[1,0].axis('equal')
                
                # XZ view (side view)
                scatter = axes[1,1].scatter(xyz_valid[:, 0], xyz_valid[:, 2], 
                                         c=rgb_valid, s=1, alpha=0.6)
                axes[1,1].set_title('Point Cloud - XZ View (Side View)')
                axes[1,1].set_xlabel('X (meters)')
                axes[1,1].set_ylabel('Z (meters)')
                
                # 3D extent
                axes[1,2].text(0.1, 0.8, f"3D Point Cloud Extents:", fontsize=12, fontweight='bold')
                axes[1,2].text(0.1, 0.7, f"X: {np.min(xyz_valid[:,0]):.2f} to {np.max(xyz_valid[:,0]):.2f} m")
                axes[1,2].text(0.1, 0.6, f"Y: {np.min(xyz_valid[:,1]):.2f} to {np.max(xyz_valid[:,1]):.2f} m")
                axes[1,2].text(0.1, 0.5, f"Z: {np.min(xyz_valid[:,2]):.2f} to {np.max(xyz_valid[:,2]):.2f} m")
                axes[1,2].text(0.1, 0.4, f"Valid points: {len(xyz_valid):,}")
                axes[1,2].text(0.1, 0.3, f"Camera position: [{pose[0,3]:.2f}, {pose[1,3]:.2f}, {pose[2,3]:.2f}]")
                
                # Check for common issues
                if np.max(np.abs(xyz_valid)) > 100:
                    axes[1,2].text(0.1, 0.1, "⚠️ WARNING: Very large coordinates!", color='red')
                if np.min(xyz_valid[:,2]) < 0:
                    axes[1,2].text(0.1, 0.05, "⚠️ WARNING: Negative Z coordinates!", color='red')
                    
            else:
                axes[1,0].text(0.5, 0.5, 'No valid 3D points', ha='center', va='center')
                axes[1,1].text(0.5, 0.5, 'No valid 3D points', ha='center', va='center')
                axes[1,2].text(0.5, 0.5, 'No valid 3D points', ha='center', va='center')
        
        axes[1,2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'pkl_debug_frame_{frame_idx}.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Additional debugging info
        print(f"\n=== Debugging Summary ===")
        if xyz is not None and len(valid_depths) > 0:
            print("✓ Has valid RGB data")
            print("✓ Has valid depth data") 
            print("✓ Has computed XYZ coordinates")
            
            xyz_flat = xyz.reshape(-1, 3)
            valid_xyz = xyz_flat[(xyz_flat[:, 2] > 0.1) & (xyz_flat[:, 2] < 10.0)]
            
            if len(valid_xyz) == 0:
                print("❌ No valid 3D points after filtering")
            elif np.max(np.abs(valid_xyz)) > 50:
                print("⚠️  3D coordinates seem very large - coordinate system issue?")
            elif np.allclose(pose[:3, :3], np.eye(3)):
                print("⚠️  Camera pose has no rotation - might be identity matrix")
            else:
                print("✓ Data looks reasonable")
        else:
            print("❌ Missing critical data")
        
    except Exception as e:
        print(f"Error visualizing pkl file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_pkl_data.py <pkl_file> [frame_index]")
        sys.exit(1)
    
    pkl_path = sys.argv[1]
    frame_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    
    visualize_pkl_data(pkl_path, frame_idx)