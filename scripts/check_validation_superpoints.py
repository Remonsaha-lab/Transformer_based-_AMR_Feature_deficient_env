#!/usr/bin/env python3
"""
Superpoint Visualization Script
================================
Visualizes superpoint partitions from preprocessed DIODE data.

Produces 3 views for each scene and saves them as PNG images:
  1. Original RGB colors
  2. Level-1 superpoints (fine segments) — each superpoint gets a random color
  3. Level-2 superpoints (coarse segments) — each superpoint gets a random color

Usage:
  cd /home/remon/EZ_SP_CFDVri/superpoint_transformer

  # Visualize a single validation scene (headless, saves PNG):
  python scripts/check_validation_superpoints.py \
      experiment=semantic/diode_ezsp.yaml \
      datamodule.data_dir=data/diode \
      datamodule.pretrained_cnn_ckpt_path=logs/ezsp_diode_encoder_v3_full/runs/2026-07-04_12-31-47/checkpoints/epoch_044.ckpt \
      +vis_num_scenes=3 \
      +vis_output_dir=outputs/superpoint_vis

  # Change the split (train/val/test):
  +vis_stage=val

  # Select specific scene indices:
  +vis_scene_indices=[0,5,10]
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path

import pyrootutils

root = Path(pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "README.md"],
    pythonpath=True,
    dotenv=True,
))

import hydra
from omegaconf import DictConfig, OmegaConf

from src.data import NAG
from src import utils

log = utils.get_pylogger(__name__)

if not OmegaConf.has_resolver("eval"):
    OmegaConf.register_new_resolver("eval", eval)


def generate_distinct_colors(n: int, seed: int = 42) -> np.ndarray:
    """Generate N visually distinct random colors using HSV spacing."""
    rng = np.random.RandomState(seed)
    if n == 0:
        return np.zeros((0, 3))
    # Use golden ratio to spread hues evenly
    hues = np.linspace(0, 1, n, endpoint=False)
    hues = (hues + rng.random()) % 1.0  # random offset
    rng.shuffle(hues)
    saturation = 0.7 + 0.3 * rng.random(n)
    value = 0.7 + 0.3 * rng.random(n)

    # HSV to RGB
    import colorsys
    colors = np.array([colorsys.hsv_to_rgb(h, s, v)
                       for h, s, v in zip(hues, saturation, value)])
    return colors


def colorize_by_superpoints(pos: np.ndarray, super_index: np.ndarray,
                            seed: int = 42) -> np.ndarray:
    """Assign a random color to each superpoint and map to points."""
    num_sp = int(super_index.max()) + 1
    sp_colors = generate_distinct_colors(num_sp, seed=seed)
    point_colors = sp_colors[super_index]
    return point_colors


def colorize_by_labels(labels: np.ndarray, num_classes: int = 20) -> np.ndarray:
    """Color points by their semantic label."""
    # DIODE class colors
    class_colors = np.array([
        [200, 50, 50],    # 0  wall
        [130, 130, 200],  # 1  floor
        [230, 230, 130],  # 2  ceiling
        [140, 100, 60],   # 3  ground
        [100, 100, 100],  # 4  road
        [80, 180, 230],   # 5  sky
        [150, 100, 50],   # 6  building
        [30, 140, 30],    # 7  tree
        [100, 200, 100],  # 8  vegetation
        [30, 100, 200],   # 9  water
        [160, 140, 120],  # 10 mountain
        [230, 130, 30],   # 11 vehicle
        [230, 180, 180],  # 12 person
        [130, 50, 160],   # 13 chair
        [180, 120, 60],   # 14 table
        [200, 80, 120],   # 15 sofa
        [80, 140, 140],   # 16 cabinet
        [200, 230, 230],  # 17 window
        [100, 50, 30],    # 18 door
        [180, 180, 180],  # 19 other
        [0, 0, 0],        # 20 ignored
    ], dtype=np.float64) / 255.0

    # Handle multi-dimensional labels (e.g. CSR histogram format)
    if labels.ndim > 1:
        labels = labels.argmax(axis=-1)

    labels = labels.astype(int).flatten()
    # Clamp out-of-range labels to the "ignored" class
    labels = np.clip(labels, 0, len(class_colors) - 1)
    return class_colors[labels]


def render_pointcloud_to_image(pos: np.ndarray, colors: np.ndarray,
                               title: str, output_path: str,
                               point_size: float = 2.0,
                               width: int = 1280, height: int = 720,
                               interactive: bool = False) -> None:
    """Render a point cloud to a PNG image using Open3D offscreen rendering.
    Optionally opens an interactive 3D window first.
    """
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pos)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1))

    if interactive:
        log.info(f"Opening interactive 3D window for: {title}")
        log.info("Controls:")
        log.info("  - Left click + drag: Rotate camera")
        log.info("  - Right click + drag: Translate/pan camera")
        log.info("  - Mouse wheel: Zoom in/out")
        log.info("  - Press [Q] or close the window to go to the next view")
        o3d.visualization.draw_geometries([pcd], window_name=title, width=width, height=height)

    # Try offscreen rendering
    try:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=title, width=width, height=height, visible=False)
        vis.add_geometry(pcd)

        # Set render options
        opt = vis.get_render_option()
        opt.point_size = point_size
        opt.background_color = np.array([0.05, 0.05, 0.1])  # dark background

        # Set camera viewpoint
        ctr = vis.get_view_control()
        ctr.set_zoom(0.7)

        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(output_path, do_render=True)
        vis.destroy_window()
        log.info(f"Saved: {output_path}")
    except Exception as e:
        log.warning(f"Open3D offscreen rendering failed: {e}")
        log.info("Falling back to matplotlib 3D scatter plot...")
        render_pointcloud_matplotlib(pos, colors, title, output_path)


def render_pointcloud_matplotlib(pos: np.ndarray, colors: np.ndarray,
                                  title: str, output_path: str) -> None:
    """Fallback: render point cloud using matplotlib 3D scatter."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # Subsample if too many points for matplotlib
    max_points = 50000
    if len(pos) > max_points:
        idx = np.random.choice(len(pos), max_points, replace=False)
        pos = pos[idx]
        colors = colors[idx]

    fig = plt.figure(figsize=(16, 10), dpi=120)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
               c=colors, s=0.5, alpha=0.8)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_facecolor((0.05, 0.05, 0.1))
    fig.patch.set_facecolor((0.05, 0.05, 0.1))
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.zaxis.label.set_color('white')
    ax.title.set_color('white')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    log.info(f"Saved (matplotlib): {output_path}")


def print_nag_summary(nag: NAG, cloud_id: str) -> None:
    """Print a summary of the NAG structure."""
    num_points = nag.num_points
    print(f"\n{'='*70}")
    print(f"Scene: {cloud_id}")
    print(f"{'='*70}")
    for level in nag.level_range:
        data = nag[level]
        n_nodes = int(data.num_nodes)
        n_edges = int(data.edge_index.shape[1]) if hasattr(data, 'edge_index') and data.edge_index is not None else 0

        sp_info = ""
        if data.super_index is not None:
            n_sp = int(data.super_index.max().item()) + 1
            counts = torch.bincount(data.super_index)
            sp_info = (f" → {n_sp} superpoints "
                       f"(size: min={counts.min().item()}, "
                       f"mean={counts.float().mean().item():.1f}, "
                       f"max={counts.max().item()})")

        print(f"  Level {level}: {n_nodes:>7} nodes, {n_edges:>7} edges{sp_info}")
    print()


def visualize_scene(nag: NAG, cloud_id: str, output_dir: Path, scene_idx: int, interactive: bool = False) -> None:
    """Generate all visualizations for a single scene."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean up the cloud_id for filenames
    safe_name = cloud_id.replace("/", "_").replace("\\", "_")

    # Print the NAG summary
    print_nag_summary(nag, cloud_id)

    # Get point-level data (level 0)
    level0 = nag[0]
    pos = level0.pos.detach().cpu().numpy()

    # --- View 1: Original RGB ---
    rgb = getattr(level0, 'rgb', None)
    if rgb is not None:
        rgb_np = rgb.detach().cpu().numpy()
        # Ensure RGB is in [0, 1]
        if rgb_np.max() > 1.5:
            rgb_np = rgb_np / 255.0
        render_pointcloud_to_image(
            pos, rgb_np,
            title=f"RGB — {cloud_id}",
            output_path=str(output_dir / f"{scene_idx:03d}_{safe_name}_1_rgb.png"),
            interactive=interactive,
        )

    # --- View 2: Semantic Labels ---
    y = getattr(level0, 'y', None)
    if y is not None:
        y_np = y.detach().cpu().numpy().astype(int)
        label_colors = colorize_by_labels(y_np)
        render_pointcloud_to_image(
            pos, label_colors,
            title=f"Semantic Labels — {cloud_id}",
            output_path=str(output_dir / f"{scene_idx:03d}_{safe_name}_2_labels.png"),
            interactive=interactive,
        )

    # --- View 3: Level-0 → Level-1 superpoints ---
    if level0.super_index is not None:
        sp_index_0 = level0.super_index.detach().cpu().numpy()
        n_sp_1 = int(sp_index_0.max()) + 1
        sp_colors_1 = colorize_by_superpoints(pos, sp_index_0, seed=42)
        render_pointcloud_to_image(
            pos, sp_colors_1,
            title=f"Level-1 Superpoints ({n_sp_1} segments) — {cloud_id}",
            output_path=str(output_dir / f"{scene_idx:03d}_{safe_name}_3_sp_level1.png"),
            interactive=interactive,
        )

        # --- View 4: Level-0 → Level-2 superpoints (via chaining) ---
        if len(nag) > 2:
            level1 = nag[1]
            if level1.super_index is not None:
                sp_index_1 = level1.super_index.detach().cpu().numpy()
                # Chain: point → level-1 → level-2
                sp_index_0_to_2 = sp_index_1[sp_index_0]
                n_sp_2 = int(sp_index_0_to_2.max()) + 1
                sp_colors_2 = colorize_by_superpoints(pos, sp_index_0_to_2, seed=123)
                render_pointcloud_to_image(
                    pos, sp_colors_2,
                    title=f"Level-2 Superpoints ({n_sp_2} segments) — {cloud_id}",
                    output_path=str(output_dir / f"{scene_idx:03d}_{safe_name}_4_sp_level2.png"),
                    interactive=interactive,
                )

        # --- View 5: Level-0 → Level-3 superpoints (if exists) ---
        if len(nag) > 3:
            try:
                level2 = nag[2]
                if level2.super_index is not None:
                    sp_index_2 = level2.super_index.detach().cpu().numpy()
                    sp_index_1 = nag[1].super_index.detach().cpu().numpy()
                    # Chain: point → level-1 → level-2 → level-3
                    sp_index_0_to_3 = sp_index_2[sp_index_1[sp_index_0]]
                    n_sp_3 = int(sp_index_0_to_3.max()) + 1
                    sp_colors_3 = colorize_by_superpoints(pos, sp_index_0_to_3, seed=456)
                    render_pointcloud_to_image(
                        pos, sp_colors_3,
                        title=f"Level-3 Superpoints ({n_sp_3} segments) — {cloud_id}",
                        output_path=str(output_dir / f"{scene_idx:03d}_{safe_name}_5_sp_level3.png"),
                        interactive=interactive,
                    )
            except Exception as e:
                log.warning(f"Could not visualize level-3 superpoints: {e}")

    print(f"  ✅ Visualizations saved to: {output_dir}")


@hydra.main(version_base="1.2", config_path=str(root / "configs"), config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    output_dir = Path(cfg.get("vis_output_dir", "outputs/superpoint_vis"))
    num_scenes = int(cfg.get("vis_num_scenes", 3))
    stage = str(cfg.get("vis_stage", "val"))
    scene_indices = cfg.get("vis_scene_indices", None)
    interactive = bool(cfg.get("vis_interactive", False))

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    # Build the dataset for the requested stage
    dataset = datamodule.dataset_class(
        datamodule.hparams.data_dir,
        stage=stage,
        transform=None,  # No augmentation for visualization
        pre_transform=datamodule.pre_transform,
        on_device_transform=None,
        **datamodule.kwargs,
    )

    total = len(dataset)
    log.info(f"Dataset stage='{stage}' has {total} scenes")

    if total == 0:
        log.error("No scenes found! Check your data_dir and stage settings.")
        return

    # Determine which scenes to visualize
    if scene_indices is not None:
        if isinstance(scene_indices, (list, tuple)):
            indices = [int(i) for i in scene_indices]
        else:
            indices = [int(scene_indices)]
    else:
        # Pick evenly spaced scenes
        if num_scenes >= total:
            indices = list(range(total))
        else:
            step = total / num_scenes
            indices = [int(i * step) for i in range(num_scenes)]

    log.info(f"Visualizing {len(indices)} scenes: {indices}")

    for i, idx in enumerate(indices):
        if idx >= total:
            log.warning(f"Scene index {idx} out of range (max={total-1}), skipping")
            continue

        cloud_id = dataset.cloud_ids[idx]
        log.info(f"Loading scene {idx}/{total}: {cloud_id}")

        try:
            nag = dataset[idx]
            if not isinstance(nag, NAG):
                log.warning(f"Scene {idx} returned {type(nag)}, expected NAG. Skipping.")
                continue
            visualize_scene(nag, cloud_id, output_dir, i, interactive=interactive)
        except Exception as e:
            log.error(f"Failed to load/visualize scene {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*70}")
    print(f"All visualizations saved to: {output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    main()
