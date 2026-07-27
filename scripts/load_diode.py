#!/usr/bin/env python3
"""
DIODE RGBD → PLY Conversion Script
====================================
Back-projects each DIODE (RGB, depth, mask) triplet into a 3D colored
point cloud, applies voxel-grid downsampling, and saves as .ply.

Usage:
    python scripts/load_diode.py \
        --diode_root data/diode/val \
        --out_dir    data/diode/raw_ply/val \
        --voxel_size 0.05

The script:
  1. Recursively finds all *.png RGB images under --diode_root
  2. Loads the matching *_depth.npy and *_depth_mask.npy files
  3. Applies the depth validity mask (mask == 1 only)
  4. Back-projects valid pixels to 3D using DIODE camera intrinsics
  5. Voxel-downsamples to --voxel_size (meters) and saves as .ply

Camera intrinsics (from DIODE devkit intrinsics.txt):
  fx = 886.81, fy = 927.06, cx = 512, cy = 384
  Image resolution: 1024 × 768
"""

import argparse
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import open3d as o3d
from PIL import Image


# ── DIODE camera intrinsics (fixed for all scans) ──────────────────────
FX = 886.81
FY = 927.06
CX = 512.0
CY = 384.0
WIDTH = 1024
HEIGHT = 768

# Minimum number of valid depth pixels to accept a scene
MIN_VALID_PIXELS = 1000


def make_intrinsic_matrix():
    """Create the Open3D PinholeCameraIntrinsic from DIODE parameters."""
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(
        width=WIDTH,
        height=HEIGHT,
        fx=FX,
        fy=FY,
        cx=CX,
        cy=CY,
    )
    return intrinsic


def find_rgb_images(diode_root: str) -> list[str]:
    """Recursively find all .png RGB images (exclude any *_depth* pngs)."""
    root = Path(diode_root)
    pngs = sorted(root.rglob("*.png"))
    # Filter out anything that has _depth in the filename (shouldn't exist
    # as depth is .npy, but be safe)
    rgb_paths = [str(p) for p in pngs if "_depth" not in p.name]
    return rgb_paths


def rgb_to_triplet_paths(rgb_path: str):
    """
    Given a path like .../00019_00183_indoors_160_050.png, derive the
    depth and mask paths.
    """
    base = rgb_path.rsplit(".png", 1)[0]
    depth_path = f"{base}_depth.npy"
    mask_path = f"{base}_depth_mask.npy"
    return depth_path, mask_path


def convert_single(
    rgb_path: str,
    out_path: str,
    voxel_size: float,
    intrinsic: o3d.camera.PinholeCameraIntrinsic | None = None,
) -> dict:
    """
    Convert a single DIODE RGBD frame to a .ply point cloud.

    Returns a dict with stats: {ok, num_points_raw, num_points_voxel, reason}.
    """
    depth_path, mask_path = rgb_to_triplet_paths(rgb_path)

    # ── Validate files exist ────────────────────────────────────────
    for p in (rgb_path, depth_path, mask_path):
        if not os.path.isfile(p):
            return {"ok": False, "reason": f"Missing file: {p}"}

    # ── Load data ───────────────────────────────────────────────────
    rgb = np.array(Image.open(rgb_path))             # (H, W, 3) uint8
    depth = np.load(depth_path).squeeze()             # (H, W) float32
    mask = np.load(mask_path).squeeze().astype(bool)  # (H, W) bool

    # ── Load semantic labels ────────────────────────────────────────
    semantic_path = rgb_path.replace(".png", "_semantic.npy")
    if os.path.isfile(semantic_path):
        semantic = np.load(semantic_path).squeeze()  # (H, W) int32
    else:
        # Default to ignored/void label (index 20)

        print(f"  WARNING: No semantic labels for {rgb_path}, using zeros")
        semantic = np.full((depth.shape[0], depth.shape[1]), 20, dtype=np.int32)

    # ── Filter scenes with too few valid pixels ─────────────────────
    n_valid = mask.sum()
    if n_valid < MIN_VALID_PIXELS:
        return {
            "ok": False,
            "reason": f"Too few valid pixels ({n_valid} < {MIN_VALID_PIXELS})",
        }

    # ── Back-project manually to carry labels ───────────────────────
    H, W = depth.shape
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    
    # Keep pixels that are masked as valid, have depth > 0, and depth < 300
    valid = mask & (depth > 0) & (depth < 300.0)
    
    d = depth[valid]
    u = uu[valid].astype(np.float64)
    v = vv[valid].astype(np.float64)
    
    X = (u - CX) * d / FX
    Y = (v - CY) * d / FY
    Z = d
    
    points = np.stack([X, Y, Z], axis=-1)
    colors = rgb[valid].astype(np.float64) / 255.0
    labels = semantic[valid]
    
    num_raw = len(points)
    
    # ── Build point cloud ───────────────────────────────────────────
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # ── Voxel downsample (need to track label indices) ──────────────
    if voxel_size > 0:
        # Open3D's voxel_down_sample_and_trace needs boundaries
        min_bound = pcd.get_min_bound() - voxel_size * 0.5
        max_bound = pcd.get_max_bound() + voxel_size * 0.5
        pcd, _, idx_map = pcd.voxel_down_sample_and_trace(voxel_size, min_bound, max_bound)
        # For each voxel, pick the label of the first point in that voxel
        ds_labels = np.array([labels[idx_list[0]] for idx_list in idx_map])
    else:
        ds_labels = labels
    
    num_voxel = len(pcd.points)
    
    # ── Save as PLY + separate label file ───────────────────────────
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    o3d.io.write_point_cloud(out_path, pcd, write_ascii=False)
    
    # Save labels alongside PLY
    label_path = out_path.replace(".ply", "_labels.npy")
    np.save(label_path, ds_labels.astype(np.int32))

    return {
        "ok": True,
        "num_points_raw": num_raw,
        "num_points_voxel": num_voxel,
    }


def derive_out_path(rgb_path: str, diode_root: str, out_dir: str) -> str:
    """
    Mirror the DIODE directory tree in the output folder, replacing .png
    with .ply.

    Example:
      rgb_path   = data/diode/val/indoors/scene_00019/scan_00183/xxx.png
      diode_root = data/diode/val
      out_dir    = data/diode/raw_ply/val
      → data/diode/raw_ply/val/indoors/scene_00019/scan_00183/xxx.ply
    """
    rel = os.path.relpath(rgb_path, diode_root)
    ply_rel = rel.rsplit(".png", 1)[0] + ".ply"
    return os.path.join(out_dir, ply_rel)


def _worker(args):
    """Wrapper for ProcessPoolExecutor — unpacks args tuple."""
    rgb_path, out_path, voxel_size = args
    return rgb_path, convert_single(rgb_path, out_path, voxel_size)


def main():
    parser = argparse.ArgumentParser(
        description="Convert DIODE RGBD frames to PLY point clouds.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--diode_root",
        type=str,
        required=True,
        help="Path to DIODE split directory (e.g. data/diode/val).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for PLY files (e.g. data/diode/raw_ply/val).",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.05,
        help="Voxel grid size in meters for downsampling (default: 0.05 = 5cm). "
             "Set to 0 to disable downsampling.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4). Use 1 for serial.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip scenes whose .ply output already exists.",
    )
    args = parser.parse_args()

    # ── Discover all RGB images ─────────────────────────────────────
    print(f"Scanning {args.diode_root} for RGB images...")
    rgb_paths = find_rgb_images(args.diode_root)
    if not rgb_paths:
        print(f"ERROR: No .png files found under {args.diode_root}")
        sys.exit(1)
    print(f"Found {len(rgb_paths)} RGB images.")

    # ── Build work list ─────────────────────────────────────────────
    work = []
    skipped_existing = 0
    for rgb_path in rgb_paths:
        out_path = derive_out_path(rgb_path, args.diode_root, args.out_dir)
        if args.skip_existing and os.path.isfile(out_path):
            skipped_existing += 1
            continue
        work.append((rgb_path, out_path, args.voxel_size))

    if skipped_existing:
        print(f"Skipping {skipped_existing} already-converted scenes.")
    if not work:
        print("All scenes already converted. Nothing to do.")
        return

    print(f"Converting {len(work)} scenes (voxel_size={args.voxel_size}m, "
          f"workers={args.workers})...")
    print()

    # ── Process ─────────────────────────────────────────────────────
    t0 = time.time()
    ok_count = 0
    fail_count = 0
    total_raw = 0
    total_voxel = 0

    if args.workers <= 1:
        # Serial — easier to debug
        intrinsic = make_intrinsic_matrix()
        for i, (rgb_path, out_path, voxel_size) in enumerate(work, 1):
            result = convert_single(rgb_path, out_path, voxel_size, intrinsic)
            _report_progress(i, len(work), rgb_path, result)
            if result["ok"]:
                ok_count += 1
                total_raw += result["num_points_raw"]
                total_voxel += result["num_points_voxel"]
            else:
                fail_count += 1
    else:
        # Parallel
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_worker, w): w for w in work}
            for i, future in enumerate(as_completed(futures), 1):
                rgb_path, result = future.result()
                _report_progress(i, len(work), rgb_path, result)
                if result["ok"]:
                    ok_count += 1
                    total_raw += result["num_points_raw"]
                    total_voxel += result["num_points_voxel"]
                else:
                    fail_count += 1

    elapsed = time.time() - t0

    # ── Summary ─────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"  Conversion complete in {elapsed:.1f}s")
    print(f"  Success: {ok_count}  |  Failed/Skipped: {fail_count}")
    if ok_count > 0:
        print(f"  Avg raw points:   {total_raw // ok_count:,}")
        print(f"  Avg voxel points: {total_voxel // ok_count:,}")
    print(f"  Output directory: {args.out_dir}")
    print("=" * 60)


def _report_progress(i, total, rgb_path, result):
    """Print a single-line progress update."""
    name = os.path.basename(rgb_path).replace(".png", "")
    if result["ok"]:
        raw = result["num_points_raw"]
        vox = result["num_points_voxel"]
        print(f"  [{i:4d}/{total}] ✓ {name}  "
              f"({raw:,} → {vox:,} pts)")
    else:
        print(f"  [{i:4d}/{total}] ✗ {name}  "
              f"({result['reason']})")


if __name__ == "__main__":
    main()
