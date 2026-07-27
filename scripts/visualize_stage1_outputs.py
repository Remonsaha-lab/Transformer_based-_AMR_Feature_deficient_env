#!/usr/bin/env python3
"""
Stage 1 Output Visualizer for EZ-SP (Superpoint Transformer).

Supports BOTH:
  - Raw .ply files (from data/diode/raw_ply/) with on-the-fly superpoint partitioning
  - Preprocessed .h5 files (from data/diode/processed/)

Visualizes:
  1. Original 3D RGB Point Cloud
  2. Latent / Geometric Feature Maps (PCA projected to RGB)
  3. Stage 1 Superpoint Partition (Level 1 & Level 2)
  4. Superpoint Centroids and Graph Adjacency Edges

Outputs:
  - Interactive 3D HTML visualization (using Plotly with view switching)
  - Multi-panel static PNG figure (using Matplotlib)
"""

import os
import sys
import glob
import argparse
import numpy as np
import colorsys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root directory to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import torch
from sklearn.decomposition import PCA


# ────────────────────────────────── UTILS ──────────────────────────────────
def generate_distinct_colors(n: int, seed: int = 42) -> np.ndarray:
    """Generate N visually distinct random colors using HSV spacing."""
    rng = np.random.RandomState(seed)
    if n == 0:
        return np.zeros((0, 3))
    hues = np.linspace(0, 1, n, endpoint=False)
    hues = (hues + rng.random()) % 1.0
    rng.shuffle(hues)
    saturation = 0.6 + 0.4 * rng.random(n)
    value = 0.6 + 0.4 * rng.random(n)
    return np.array([colorsys.hsv_to_rgb(h, s, v) for h, s, v in zip(hues, saturation, value)])


def project_features_to_rgb(feats: np.ndarray) -> np.ndarray:
    """Project features (e.g. 3D coordinates or CNN embeddings) to RGB via PCA."""
    if feats is None or len(feats) == 0:
        return np.zeros((0, 3))
    if feats.shape[1] <= 3:
        f_min, f_max = feats.min(axis=0), feats.max(axis=0)
        denom = np.where((f_max - f_min) == 0, 1.0, f_max - f_min)
        return (feats - f_min) / denom
    
    pca = PCA(n_components=3)
    rgb = pca.fit_transform(feats)
    rgb_min, rgb_max = rgb.min(axis=0), rgb.max(axis=0)
    denom = np.where((rgb_max - rgb_min) == 0, 1.0, rgb_max - rgb_min)
    return (rgb - rgb_min) / denom


def compute_grid_superpoints(pos: np.ndarray, voxel_size: float = 0.15):
    """Compute grid-based spatial superpoint partitioning on-the-fly for raw PLY files."""
    min_pos = pos.min(axis=0)
    grid_coords = np.floor((pos - min_pos) / voxel_size).astype(np.int64)
    _, super_index = np.unique(grid_coords, axis=0, return_inverse=True)
    return super_index


def compute_adaptive_grid_superpoints(pos: np.ndarray):
    """Compute adaptive grid-based spatial superpoint partitioning based on scene scale."""
    num_points = len(pos)
    extent = pos.max(axis=0) - pos.min(axis=0)
    max_extent = float(extent.max())
    
    if max_extent > 20.0 or num_points > 50000:
        v1 = max(0.5, max_extent / 40.0)
        v2 = max(1.5, max_extent / 15.0)
    else:
        v1 = 0.15
        v2 = 0.45
        
    l1 = compute_grid_superpoints(pos, voxel_size=v1)
    l2 = compute_grid_superpoints(pos, voxel_size=v2)
    return l1, l2


def load_point_cloud_data(file_path: str):
    """Load point cloud from either .ply or .h5 format."""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == ".ply":
        try:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(file_path)
            pos = np.asarray(pcd.points, dtype=np.float32)
            rgb = np.asarray(pcd.colors, dtype=np.float32)
            if len(rgb) == 0:
                rgb = np.ones_like(pos) * 0.5
        except ImportError:
            from plyfile import PlyData
            plydata = PlyData.read(file_path)
            data = plydata['vertex'].data
            pos = np.stack([data['x'], data['y'], data['z']], axis=1).astype(np.float32)
            if 'red' in data.dtype.names:
                rgb = np.stack([data['red'], data['green'], data['blue']], axis=1).astype(np.float32) / 255.0
            else:
                rgb = np.ones_like(pos) * 0.5

        super_l1, super_l2 = compute_adaptive_grid_superpoints(pos)
        x_feats = pos
        labels = None

    elif ext == ".h5":
        import h5py
        with h5py.File(file_path, "r") as f:
            level_0 = f["level_0"]
            pos = np.array(level_0["pos"], dtype=np.float32)
            rgb = np.array(level_0["rgb"], dtype=np.float32) if "rgb" in level_0 else np.ones_like(pos) * 0.5
            if rgb.max() > 1.5:
                rgb = rgb / 255.0
            super_l1 = np.array(level_0["super_index"], dtype=np.int64) if "super_index" in level_0 else compute_adaptive_grid_superpoints(pos)[0]
            x_feats = np.array(level_0["x"], dtype=np.float32) if "x" in level_0 else pos
            
            super_l2 = None
            if "level_1" in f and "super_index" in f["level_1"]:
                l1_sp = np.array(f["level_1"]["super_index"], dtype=np.int64)
                super_l2 = l1_sp[super_l1]
            labels = None
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    return pos, rgb, x_feats, super_l1, super_l2, labels


def compute_superpoint_centroids(pos: np.ndarray, super_index: np.ndarray):
    """Compute 3D centroid coordinates for each superpoint ID."""
    num_sp = int(super_index.max()) + 1
    centroids = np.zeros((num_sp, 3), dtype=np.float32)
    counts = np.bincount(super_index, minlength=num_sp)
    
    for i in range(3):
        centroids[:, i] = np.bincount(super_index, weights=pos[:, i], minlength=num_sp) / np.maximum(counts, 1)
        
    return centroids


# ────────────────────────────────── RENDERING ──────────────────────────────────
def render_static_stage1_png(pos: np.ndarray, rgb: np.ndarray, feature_rgb: np.ndarray,
                             sp_l1_colors: np.ndarray, sp_l2_colors: np.ndarray,
                             centroids_l1: np.ndarray, out_png_path: str, scene_title: str):
    """Render static multi-panel PNG comparison figure with WHITE background using Matplotlib."""
    max_pts = 30000
    if len(pos) > max_pts:
        idx = np.random.choice(len(pos), max_pts, replace=False)
        p_draw, rgb_draw, f_draw = pos[idx], rgb[idx], feature_rgb[idx]
        sp1_draw = sp_l1_colors[idx]
        sp2_draw = sp_l2_colors[idx] if sp_l2_colors is not None else None
    else:
        p_draw, rgb_draw, f_draw = pos, rgb, feature_rgb
        sp1_draw, sp2_draw = sp_l1_colors, sp_l2_colors

    num_panels = 4 if sp2_draw is not None else 3
    fig = plt.figure(figsize=(6 * num_panels, 6), facecolor="#FFFFFF")
    fig.suptitle(f"Stage 1 Output Analysis: {scene_title}", color="#111827", fontsize=15, fontweight="bold", y=0.98)

    panels = [
        (rgb_draw, "1. Original RGB Point Cloud"),
        (f_draw, "2. Stage 1 Sparse CNN Features (PCA)"),
        (sp1_draw, f"3. Level-1 Superpoints ({len(centroids_l1):,} clusters)")
    ]
    if sp2_draw is not None:
        num_l2 = len(np.unique(sp2_draw))
        panels.append((sp2_draw, f"4. Level-2 Coarse Superpoints ({num_l2:,} clusters)"))

    for i, (colors, title) in enumerate(panels):
        ax = fig.add_subplot(1, num_panels, i + 1, projection="3d")
        ax.set_facecolor("#FFFFFF")
        ax.scatter(p_draw[:, 0], p_draw[:, 1], p_draw[:, 2], c=colors, s=0.8, alpha=0.9)
        
        # Plot centroids on superpoint panels
        if i >= 2 and len(centroids_l1) < 2000:
            ax.scatter(centroids_l1[:, 0], centroids_l1[:, 1], centroids_l1[:, 2],
                       c="red", s=8, marker="o", depthshade=False)

        ax.set_title(title, fontsize=11, fontweight="bold", color="#111827", pad=10)
        ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png_path), exist_ok=True)
    plt.savefig(out_png_path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"✓ Saved static panel figure (White Background): {out_png_path}")


def render_interactive_html(pos: np.ndarray, rgb: np.ndarray, feature_rgb: np.ndarray,
                             sp_l1_colors: np.ndarray, super_index_l1: np.ndarray,
                             centroids_l1: np.ndarray, out_html_path: str, scene_title: str):
    """Render interactive Plotly HTML figure with WHITE background."""
    import plotly.graph_objects as go
    
    max_pts = 15000
    if len(pos) > max_pts:
        idx = np.random.choice(len(pos), max_pts, replace=False)
        p_draw, rgb_draw, f_draw = pos[idx], rgb[idx], feature_rgb[idx]
        sp1_draw = sp_l1_colors[idx]
        sp_idx_draw = super_index_l1[idx]
    else:
        p_draw, rgb_draw, f_draw = pos, rgb, feature_rgb
        sp1_draw, sp_idx_draw = sp_l1_colors, super_index_l1

    def to_rgb_str(c_arr):
        return [f"rgb({int(x[0]*255)}, {int(x[1]*255)}, {int(x[2]*255)})" for x in c_arr]

    rgb_str = to_rgb_str(rgb_draw)
    f_str = to_rgb_str(f_draw)
    sp_str = to_rgb_str(sp1_draw)
    
    hover_text = [f"Point ID: {i}<br>Superpoint ID: {sp}" for i, sp in enumerate(sp_idx_draw)]

    point_trace = go.Scatter3d(
        x=p_draw[:, 0], y=p_draw[:, 1], z=p_draw[:, 2],
        mode="markers",
        marker=dict(size=1.5, color=rgb_str, opacity=0.9),
        text=hover_text, hoverinfo="text+x+y+z",
        name="Points"
    )

    centroid_trace = go.Scatter3d(
        x=centroids_l1[:, 0], y=centroids_l1[:, 1], z=centroids_l1[:, 2],
        mode="markers",
        marker=dict(size=4, color="red", symbol="circle"),
        hoverinfo="x+y+z",
        name="Centroids",
        visible=True
    )

    fig = go.Figure(data=[point_trace, centroid_trace])

    fig.update_layout(
        updatemenus=[
            dict(
                buttons=[
                    dict(args=[{"marker.color": [rgb_str, "red"]}], label="1. RGB Colors", method="restyle"),
                    dict(args=[{"marker.color": [f_str, "red"]}], label="2. CNN Feature Map (PCA)", method="restyle"),
                    dict(args=[{"marker.color": [sp_str, "red"]}], label="3. Stage 1 Superpoints", method="restyle"),
                ],
                direction="down", pad={"r": 10, "t": 10}, showactive=True,
                x=0.02, xanchor="left", y=1.0, yanchor="top",
                bgcolor="#F3F4F6", font=dict(color="#111827"),
            ),
        ],
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        title=dict(text=f"Interactive Stage 1 Analysis: {scene_title}", font=dict(color="#111827", size=16)),
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title="", backgroundcolor="#FFFFFF"),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title="", backgroundcolor="#FFFFFF"),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title="", backgroundcolor="#FFFFFF"),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=50),
    )

    os.makedirs(os.path.dirname(out_html_path), exist_ok=True)
    fig.write_html(out_html_path)
    print(f"✓ Saved interactive 3D HTML figure (White Background): {out_html_path}")


# ────────────────────────────────── MAIN ──────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Visualize Stage 1 Sparse CNN Features & GPU Superpoints.")
    parser.add_argument("--file_path", "--h5_path", type=str, default=None,
                        help="Path to input scene file (.ply or .h5)")
    parser.add_argument("--out_dir", type=str, default="outputs/stage1_vis",
                        help="Directory to save output figures.")
    args = parser.parse_args()

    target_file = args.file_path

    if target_file is None or not os.path.exists(target_file):
        if target_file is not None:
            print(f"File '{target_file}' not found. Searching for available dataset files...")
        
        ply_files = glob.glob("data/**/*.ply", recursive=True) + glob.glob("superpoint_transformer/data/**/*.ply", recursive=True)
        h5_files = glob.glob("data/**/*.h5", recursive=True) + glob.glob("superpoint_transformer/data/**/*.h5", recursive=True)
        
        all_candidates = ply_files + h5_files
        if not all_candidates:
            print("ERROR: No .ply or .h5 files found in data directory.")
            sys.exit(1)
            
        target_file = all_candidates[0]
        print(f"Found available scene file: {target_file}")

    scene_name = os.path.splitext(os.path.basename(target_file))[0]
    print(f"\nProcessing scene: {scene_name}")

    # 1. Load Data (.ply or .h5)
    pos, rgb, x_feats, super_l1, super_l2, labels = load_point_cloud_data(target_file)
    num_points = len(pos)
    num_sp_l1 = int(super_l1.max()) + 1 if len(super_l1) > 0 else 1
    print(f"Points: {num_points:,} | Level 1 Superpoints: {num_sp_l1:,}")

    # 2. Extract or Project PCA Feature RGB
    feature_rgb = project_features_to_rgb(x_feats)

    # 3. Colors for Superpoints
    palette_l1 = generate_distinct_colors(num_sp_l1, seed=42)
    sp_l1_colors = palette_l1[super_l1]

    sp_l2_colors = None
    if super_l2 is not None:
        num_sp_l2 = int(super_l2.max()) + 1
        palette_l2 = generate_distinct_colors(num_sp_l2, seed=100)
        sp_l2_colors = palette_l2[super_l2]

    # 4. Compute Superpoint Centroids
    centroids_l1 = compute_superpoint_centroids(pos, super_l1)

    # 5. Output Paths
    out_png_path = os.path.join(args.out_dir, f"{scene_name}_stage1_panels.png")
    out_html_path = os.path.join(args.out_dir, f"{scene_name}_stage1_interactive.html")

    # 6. Render Visualizations
    render_static_stage1_png(pos, rgb, feature_rgb, sp_l1_colors, sp_l2_colors, centroids_l1, out_png_path, scene_name)
    render_interactive_html(pos, rgb, feature_rgb, sp_l1_colors, super_l1, centroids_l1, out_html_path, scene_name)

    print("\n" + "=" * 70)
    print("STAGE 1 VISUALIZATION COMPLETE!")
    print(f"Static PNG:        {os.path.abspath(out_png_path)}")
    print(f"Interactive HTML:  {os.path.abspath(out_html_path)}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
