#!/usr/bin/env python3
import os
import sys
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def main():
    # Paths to raw files and outputs
    raw_path = "/home/remon/EZ_SP_CFDVri/superpoint_transformer/data/diode/raw_ply_data/train/outdoor/scene_00010/scan_00100/00010_00100_outdoor_010_010.ply"
    out_dir = "/home/remon/EZ_SP_CFDVri/superpoint_transformer/outputs"
    os.makedirs(out_dir, exist_ok=True)
    
    if not os.path.exists(raw_path):
        # Fallback to alternative val path
        alt_path = "/home/remon/EZ_SP_CFDVri/superpoint_transformer/data/diode/raw_ply/val/outdoor/scene_00007/scan_00082/00007_00082_outdoor_000_010.ply"
        if os.path.exists(alt_path):
            raw_path = alt_path
        else:
            print(f"Error: Raw PLY file not found at {raw_path} or {alt_path}")
            sys.exit(1)
            
    print(f"Loading raw point cloud from: {raw_path}")
    pcd_raw = o3d.io.read_point_cloud(raw_path)
    points_raw = np.asarray(pcd_raw.points)
    colors_raw = np.asarray(pcd_raw.colors)
    num_raw = len(points_raw)
    print(f"Loaded {num_raw:,} raw points.")

    # Downsampling parameter
    voxel_size = 0.03
    print(f"Applying voxel downsampling with grid size = {voxel_size}m...")
    pcd_down = pcd_raw.voxel_down_sample(voxel_size)
    points_down = np.asarray(pcd_down.points)
    colors_down = np.asarray(pcd_down.colors)
    num_down = len(points_down)
    print(f"Downsampled to {num_down:,} points.")

    # Create a 2D orthographic plot projection (XY plane) to compare density
    fig, axes = plt.subplots(1, 2, figsize=(18, 9), dpi=150, facecolor="white")
    
    # 1. Raw Point Cloud Plot
    axes[0].set_facecolor("white")
    axes[0].scatter(points_raw[:, 0], points_raw[:, 1], c=colors_raw, s=0.2, alpha=0.8)
    axes[0].set_title(f"Raw Point Cloud ({num_raw:,} points)", color="black", fontsize=14, fontweight="bold")
    axes[0].axis("off")
    axes[0].set_aspect("equal")
    
    # 2. Downsampled Point Cloud Plot
    axes[1].set_facecolor("white")
    axes[1].scatter(points_down[:, 0], points_down[:, 1], c=colors_down, s=0.8, alpha=0.9)
    axes[1].set_title(f"Voxel Downsampled (voxel={voxel_size}m, {num_down:,} points)", color="black", fontsize=14, fontweight="bold")
    axes[1].axis("off")
    axes[1].set_aspect("equal")

    # Add spacing and title
    fig.suptitle(f"DIODE Point Cloud Downsampling Comparison\nScan: {os.path.basename(raw_path)}", 
                 color="black", fontsize=16, fontweight="bold", y=0.96)
    
    out_img_path = os.path.join(out_dir, "voxel_downsampling_comparison.png")
    plt.savefig(out_img_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    
    print(f"✓ Comparison image successfully saved to: {out_img_path}")

if __name__ == "__main__":
    main()
