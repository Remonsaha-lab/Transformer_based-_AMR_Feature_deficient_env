#!/usr/bin/env python3
"""
Visualize a specific scene's SegFormer labels and Stage 1 superpoint boundaries.
Loads directly from the preprocessed H5 file to show the exact coordinates,
colors, boundary partitions (superpoints), and SegFormer labels.

Computes and displays superpoint purity (accuracy of boundary creation) per class.
Saves a 4-panel static PNG comparing:
  1. Original RGB colors
  2. SegFormer Semantic Labels
  3. Stage 1 Superpoints
  4. Superpoint Purity (Green = pure, Red = impure)
Also embeds a detailed legend with class-wise purity and point counts.
"""

import os
import sys
import glob
import argparse
import colorsys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import h5py

# ──────────────────────────── CONFIG ────────────────────────────
CLASS_NAMES = [
    "wall", "floor", "ceiling", "ground", "road",
    "sky", "building", "tree", "vegetation", "water",
    "mountain", "vehicle", "person", "chair", "table",
    "sofa", "cabinet", "window", "door", "other",
    "ignored"
]
NUM_CLASSES = len(CLASS_NAMES) - 1  # 20 classes (excluding ignored)

CLASS_COLORS = np.array([
    [200, 50, 50],    # 0  wall (red-brown)
    [130, 130, 200],  # 1  floor (blue-grey)
    [230, 230, 130],  # 2  ceiling (yellow)
    [140, 100, 60],   # 3  ground (brown)
    [100, 100, 100],  # 4  road (grey)
    [80, 180, 230],   # 5  sky (light blue)
    [150, 100, 50],   # 6  building (dark orange)
    [30, 140, 30],    # 7  tree (dark green)
    [100, 200, 100],  # 8  vegetation (light green)
    [30, 100, 200],   # 9  water (deep blue)
    [160, 140, 120],  # 10 mountain (tan)
    [230, 130, 30],   # 11 vehicle (orange)
    [230, 180, 180],  # 12 person (pink)
    [130, 50, 160],   # 13 chair (purple)
    [180, 120, 60],   # 14 table (wood brown)
    [200, 80, 120],   # 15 sofa (magenta)
    [80, 140, 140],   # 16 cabinet (teal)
    [200, 230, 230],  # 17 window (ice blue)
    [100, 50, 30],    # 18 door (dark brown)
    [180, 180, 180],  # 19 other (light grey)
    [0, 0, 0],        # 20 ignored (black)
], dtype=np.float64) / 255.0

DEFAULT_SCAN_ID = "outdoor/scene_00007/scan_00083/00007_00083_outdoor_140_020"


# ──────────────────────────── UTILS ────────────────────────────
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
    colors = np.array([colorsys.hsv_to_rgb(h, s, v) for h, s, v in zip(hues, saturation, value)])
    return colors


def resolve_h5_path(scan_id: str, processed_dir: str) -> str:
    """Search train/val/test splits for the H5 file matching scan_id."""
    scan_id = scan_id.strip()
    h5_name = scan_id.split("/")[-1] + ".h5"
    base_dir = os.path.dirname(os.path.dirname(processed_dir))

    for split in ["test", "train", "val"]:
        # Find directories inside split matching the preprocessing hash (e.g. 4db5043ca280d54530b842a239a31d90)
        split_dir = os.path.join(base_dir, split)
        if os.path.exists(split_dir):
            hash_dirs = os.listdir(split_dir)
            for hd in hash_dirs:
                candidate = os.path.join(split_dir, hd, *scan_id.split("/")[:-1], h5_name)
                if os.path.exists(candidate):
                    print(f"Found H5 file in split '{split}': {candidate}")
                    return candidate
    raise FileNotFoundError(f"Could not find H5 file for scan_id: {scan_id}")


def load_h5_data(h5_path: str):
    """Load coordinates, colors, superpoints, and reconstructed labels from H5."""
    with h5py.File(h5_path, "r") as f:
        pos = np.array(f["level_0/pos"], dtype=np.float32)
        rgb = np.array(f["level_0/rgb"], dtype=np.float32)
        if rgb.max() > 1.5:
            rgb = rgb / 255.0
        super_index = np.array(f["level_0/super_index"], dtype=np.int64)

        y_ptrs = np.array(f["level_0/_csr_/y/pointers"])
        y_cols = np.array(f["level_0/_csr_/y/columns"])
        y_vals = np.array(f["level_0/_csr_/y/values"])

    num_points = pos.shape[0]

    # Reconstruct dominant point labels from sparse CSR representation
    point_labels = np.full(num_points, NUM_CLASSES, dtype=np.int64)  # default ignored
    counts = np.diff(y_ptrs)
    starts = y_ptrs[:-1]
    for p in range(num_points):
        start = starts[p]
        end = start + counts[p]
        if end == start:
            continue
        cols = y_cols[start:end]
        vals = y_vals[start:end]
        valid = cols < NUM_CLASSES
        if not np.any(valid):
            continue
        cols, vals = cols[valid], vals[valid]
        point_labels[p] = cols[np.argmax(vals)]

    return pos, rgb, super_index, point_labels


def compute_purity_metrics(super_index: np.ndarray, labels: np.ndarray):
    """
    For each superpoint, calculate its purity = (dominant class count) / (total points).
    Then compute the average purity per semantic class.
    """
    num_points = len(super_index)
    num_sp = int(super_index.max()) + 1
    
    # Calculate purity for each superpoint
    sp_purities = np.zeros(num_sp)
    sp_dominant_classes = np.zeros(num_sp, dtype=np.int64)
    
    # Map points to their superpoints
    sp_to_points = {i: [] for i in range(num_sp)}
    for p_idx, sp_idx in enumerate(super_index):
        sp_to_points[sp_idx].append(p_idx)
        
    point_purities = np.zeros(num_points)
    
    for sp_idx in range(num_sp):
        pt_indices = sp_to_points[sp_idx]
        if not pt_indices:
            continue
        sp_labels = labels[pt_indices]
        
        # Exclude ignored label (NUM_CLASSES) if possible, unless all are ignored
        valid_labels = sp_labels[sp_labels < NUM_CLASSES]
        if len(valid_labels) == 0:
            dominant_class = NUM_CLASSES
            purity = 1.0
        else:
            classes, counts = np.unique(valid_labels, return_counts=True)
            dominant_idx = np.argmax(counts)
            dominant_class = classes[dominant_idx]
            purity = counts[dominant_idx] / len(sp_labels)
            
        sp_purities[sp_idx] = purity
        sp_dominant_classes[sp_idx] = dominant_class
        point_purities[pt_indices] = purity

    # Compute average point-level purity per class
    class_purities = {}
    class_pt_counts = {}
    for c in range(NUM_CLASSES):
        pt_mask = labels == c
        count = np.sum(pt_mask)
        if count > 0:
            class_pt_counts[c] = count
            class_purities[c] = np.mean(point_purities[pt_mask])
            
    return point_purities, sp_purities, class_purities, class_pt_counts


def get_purity_colors(point_purities: np.ndarray) -> np.ndarray:
    """Map purity [0.5, 1.0] to a Red-Yellow-Green colormap."""
    # Scale purity from [0.5, 1.0] to [0.0, 1.0] for color mapping
    scaled = np.clip((point_purities - 0.5) / 0.5, 0.0, 1.0)
    
    colors = np.zeros((len(point_purities), 3))
    # Red: [0.9, 0.1, 0.1]
    # Yellow: [0.9, 0.9, 0.1]
    # Green: [0.1, 0.8, 0.1]
    for i, t in enumerate(scaled):
        if t < 0.5:
            # Interpolate Red to Yellow
            factor = t / 0.5
            colors[i] = [0.9, 0.1 + 0.8 * factor, 0.1]
        else:
            # Interpolate Yellow to Green
            factor = (t - 0.5) / 0.5
            colors[i] = [0.9 - 0.8 * factor, 0.9 - 0.1 * factor, 0.1 + 0.7 * factor]
    return colors


def render_static_panels(pos: np.ndarray, rgb: np.ndarray, sem_colors: np.ndarray, 
                          sp_colors: np.ndarray, purity_colors: np.ndarray, 
                          class_purities: dict, class_pt_counts: dict,
                          scan_id: str, out_path: str):
    """Render and save 4-panel static comparison plot."""
    # Subsample if too many points for speed and clarity in Matplotlib
    max_pts = 30000
    num_points = pos.shape[0]
    if num_points > max_pts:
        indices = np.random.choice(num_points, max_pts, replace=False)
        pos_draw = pos[indices]
        rgb_draw = rgb[indices]
        sem_draw = sem_colors[indices]
        sp_draw = sp_colors[indices]
        pur_draw = purity_colors[indices]
    else:
        pos_draw = pos
        rgb_draw = rgb
        sem_draw = sem_colors
        sp_draw = sp_colors
        pur_draw = purity_colors

    fig = plt.figure(figsize=(24, 7), facecolor="#0D1117")
    fig.suptitle(
        f"Point Cloud Analysis: {scan_id}\nSegFormer Segmentation vs Stage 1 Superpoint Partition Purity",
        color="white", fontsize=15, fontweight="bold", y=0.98
    )

    panels = [
        (rgb_draw, "Original RGB Point Cloud"),
        (sem_draw, "SegFormer Semantic Pseudo-labels"),
        (sp_draw, f"Stage 1 Superpoints ({len(np.unique(sp_colors)):,} clusters)"),
        (pur_draw, "Superpoint Partition Purity (Green=Pure, Red=Impure)")
    ]

    for idx, (colors, title) in enumerate(panels):
        ax = fig.add_subplot(1, 4, idx + 1, projection="3d")
        ax.set_facecolor("#161B22")
        
        # Plot point cloud
        ax.scatter(pos_draw[:, 0], pos_draw[:, 1], pos_draw[:, 2], c=colors, s=0.8, alpha=0.9)
        
        ax.set_title(title, fontsize=11, fontweight="bold", color="white", pad=10)
        ax.set_xlabel("X (m)", color="white", fontsize=8)
        ax.set_ylabel("Y (m)", color="white", fontsize=8)
        ax.set_zlabel("Z (m)", color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=7)
        ax.grid(False)

    # ──────────────── Add a Legend with Class-wise Purity ────────────────
    legend_elements = []
    # Sort classes by point count descending
    sorted_classes = sorted(class_purities.keys(), key=lambda c: -class_pt_counts[c])
    
    for c in sorted_classes:
        purity_val = class_purities[c] * 100.0
        pt_count = class_pt_counts[c]
        label_text = f"{CLASS_NAMES[c].capitalize()}: {purity_val:.1f}% purity ({pt_count:,} pts)"
        legend_elements.append(
            Patch(facecolor=CLASS_COLORS[c], label=label_text)
        )
        
    # Place legend below or on the right side
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=min(5, len(legend_elements)),
        facecolor="#161B22",
        edgecolor="#30363D",
        labelcolor="white",
        fontsize=9,
        title="Class-wise Superpoint Differentiation (Purity)",
        title_fontsize=11
    )
    
    # Adjust layout
    plt.subplots_adjust(bottom=0.25, top=0.88, wspace=0.15)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"✓ Saved static 4-panel comparison to: {out_path}")


def save_interactive_html(pos: np.ndarray, rgb: np.ndarray, sem_colors: np.ndarray, 
                          sp_colors: np.ndarray, purity_colors: np.ndarray, 
                          labels: np.ndarray, super_index: np.ndarray, point_purities: np.ndarray,
                          scan_id: str, out_path: str):
    """Save an interactive 3D HTML plot using Plotly with dropdown menus."""
    import plotly.graph_objects as go
    
    # Subsample points if they are too large for smooth browser rendering
    max_pts = 15000
    num_points = pos.shape[0]
    if num_points > max_pts:
        indices = np.random.choice(num_points, max_pts, replace=False)
        pos_draw = pos[indices]
        rgb_draw = rgb[indices]
        sem_draw = sem_colors[indices]
        sp_draw = sp_colors[indices]
        pur_draw = purity_colors[indices]
        labels_draw = labels[indices]
        sp_idx_draw = super_index[indices]
        purities_draw = point_purities[indices]
    else:
        pos_draw = pos
        rgb_draw = rgb
        sem_draw = sem_colors
        sp_draw = sp_colors
        pur_draw = purity_colors
        labels_draw = labels
        sp_idx_draw = super_index
        purities_draw = point_purities

    # Helper function to convert RGB array to CSS string colors
    def to_rgb_str(c_arr):
        return [f"rgb({int(x[0]*255)}, {int(x[1]*255)}, {int(x[2]*255)})" for x in c_arr]
        
    rgb_str = to_rgb_str(rgb_draw)
    sem_str = to_rgb_str(sem_draw)
    sp_str = to_rgb_str(sp_draw)
    pur_str = to_rgb_str(pur_draw)
    
    hover_text = [
        f"Class: {CLASS_NAMES[l]}<br>Superpoint ID: {sp}<br>Purity: {p*100.0:.1f}%"
        for l, sp, p in zip(labels_draw, sp_idx_draw, purities_draw)
    ]
    
    # Create the primary trace
    point_trace = go.Scatter3d(
        x=pos_draw[:, 0], y=pos_draw[:, 1], z=pos_draw[:, 2],
        mode="markers",
        marker=dict(size=1.5, color=rgb_str, opacity=0.9),
        text=hover_text, hoverinfo="text+x+y+z",
        name="Points"
    )
    
    fig = go.Figure(data=[point_trace])
    
    # Define dropdown menus to toggle color arrays
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=[
                    dict(args=[{"marker.color": [rgb_str]}],
                         label="Original RGB Colors", method="restyle"),
                    dict(args=[{"marker.color": [sem_str]}],
                         label="SegFormer Semantic Labels", method="restyle"),
                    dict(args=[{"marker.color": [sp_str]}],
                         label="Stage 1 Superpoints", method="restyle"),
                    dict(args=[{"marker.color": [pur_str]}],
                         label="Superpoint Purity Map", method="restyle"),
                ],
                direction="down", pad={"r": 10, "t": 10}, showactive=True,
                x=0.02, xanchor="left", y=1.0, yanchor="top",
                bgcolor="#21262D", font=dict(color="white"),
            ),
        ],
        paper_bgcolor="#0D1117", plot_bgcolor="#0D1117",
        title=dict(
            text=f"Interactive Point Cloud Analysis: {scan_id}", 
            font=dict(color="white", size=16)
        ),
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title="", backgroundcolor="#161B22"),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title="", backgroundcolor="#161B22"),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title="", backgroundcolor="#161B22"),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=50),
    )
    
    fig.write_html(out_path)
    print(f"✓ Saved interactive 3D HTML visualization to: {out_path}")


# ──────────────────────────── MAIN ────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Compare SegFormer vs Stage 1 boundaries by calculating purity.")
    parser.add_argument("--scan_id", type=str, default=DEFAULT_SCAN_ID,
                        help="DIODE scan ID to visualize.")
    parser.add_argument("--processed_dir", type=str,
                        default="data/diode/diode/processed/test/4db5043ca280d54530b842a239a31d90",
                        help="Path to any processed subdirectory.")
    parser.add_argument("--out_dir", type=str, default="outputs/specific_scene_vis",
                        help="Directory to save the visualization PNG.")
    args = parser.parse_args()

    # 1. Load scene data from H5
    try:
        h5_path = resolve_h5_path(args.scan_id, args.processed_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        # Try fallback glob in processed directory
        print("Searching for any H5 files in processed directory...")
        h5_files = glob.glob(os.path.join(args.processed_dir, "**/*.h5"), recursive=True)
        if h5_files:
            h5_path = h5_files[0]
            print(f"Fallback to first available H5 file: {h5_path}")
            # Recompute scan_id from fallback path
            base_dir = os.path.dirname(os.path.dirname(args.processed_dir))
            for split in ["test", "train", "val"]:
                if f"/{split}/" in h5_path:
                    # extract path relative to hash dir
                    rel = h5_path.split(f"/{split}/")[1].split("/", 1)[1]
                    args.scan_id = os.path.splitext(rel)[0]
                    break
        else:
            print("No processed H5 files found anywhere. Please run preprocessing first.")
            sys.exit(1)

    print(f"Loading scene data: {args.scan_id}")
    pos, rgb, super_index, labels = load_h5_data(h5_path)
    num_points = pos.shape[0]
    num_sp = int(super_index.max()) + 1
    print(f"Loaded {num_points:,} points, {num_sp:,} superpoints.")

    # 2. Compute purity metrics
    point_purities, sp_purities, class_purities, class_pt_counts = compute_purity_metrics(super_index, labels)
    
    # Calculate overall scene purity
    overall_purity = np.mean(point_purities) * 100.0
    print(f"=== Scene Superpoint Purity: {overall_purity:.2f}% ===")
    
    # 3. Create point colors for display
    sem_colors = CLASS_COLORS[labels]
    
    sp_color_palette = generate_distinct_colors(num_sp, seed=42)
    sp_colors = sp_color_palette[super_index]
    
    purity_colors = get_purity_colors(point_purities)

    # 4. Print Markdown report
    print("\n" + "=" * 80)
    print(f"SUPERPOINT DIFFERENTIATION REPORT FOR SCENE: {args.scan_id}")
    print("=" * 80)
    print(f"{'Class Name':15s} | {'Point Count':12s} | {'Superpoint Purity':20s}")
    print("-" * 80)
    
    # Sort by point count descending
    sorted_classes = sorted(class_purities.keys(), key=lambda c: -class_pt_counts[c])
    for c in sorted_classes:
        purity_val = class_purities[c] * 100.0
        pt_count = class_pt_counts[c]
        print(f"{CLASS_NAMES[c]:15s} | {pt_count:12,d} | {purity_val:18.2f}%")
    print("-" * 80)
    print(f"{'OVERALL SCENE':15s} | {num_points:12,d} | {overall_purity:18.2f}%")
    print("=" * 80 + "\n")

    # 5. Render static PNG
    safe_name = args.scan_id.replace("/", "_")
    out_png_path = os.path.join(args.out_dir, f"{safe_name}_comparison.png")
    render_static_panels(pos, rgb, sem_colors, sp_colors, purity_colors, 
                         class_purities, class_pt_counts, args.scan_id, out_png_path)
    
    # 6. Save interactive HTML file
    out_html_path = os.path.join(args.out_dir, f"{safe_name}_interactive.html")
    save_interactive_html(pos, rgb, sem_colors, sp_colors, purity_colors, 
                          labels, super_index, point_purities, args.scan_id, out_html_path)
    
    # Print success
    print("Visualization completed successfully!")
    print(f"PNG Output:  {os.path.abspath(out_png_path)}")
    print(f"HTML Output: {os.path.abspath(out_html_path)}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
