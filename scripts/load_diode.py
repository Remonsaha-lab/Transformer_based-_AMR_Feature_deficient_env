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
from collections import Counter
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
    """Recursively find all .png RGB images (exclude any *_depth* or *_semantic* pngs)."""
    root = Path(diode_root)
    pngs = sorted(root.rglob("*.png"))
    rgb_paths = [
        str(p) for p in pngs
        if "_depth" not in p.name and "_semantic" not in p.name
    ]
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


def voxel_majority_voting(
    labels: np.ndarray,
    confidences: np.ndarray,
    idx_map: list,
    min_confidence: float = 0.7,
    min_consensus: float = 0.6,
    ignore_label: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform majority voting with confidence and consensus thresholding.

    Args:
        labels: 1D array of point semantic labels.
        confidences: 1D array of teacher confidence scores in [0.0, 1.0].
        idx_map: list of point index lists mapping voxels to points.
        min_confidence: minimum teacher confidence threshold (default 0.7).
        min_consensus: minimum voxel majority consensus ratio (default 0.6).
        ignore_label: label ID assigned to invalid/unconfident/ambiguous points.

    Returns:
        voxel_labels: 1D int32 array (num_voxels,)
        voxel_confs: 1D float32 array (num_voxels,)
        voxel_consensus: 1D float32 array (num_voxels,)
        voxel_counts: 1D int32 array (num_voxels,)
    """
    num_voxels = len(idx_map)
    voxel_labels = np.empty(num_voxels, dtype=np.int32)
    voxel_confs = np.empty(num_voxels, dtype=np.float32)
    voxel_consensus = np.empty(num_voxels, dtype=np.float32)
    voxel_counts = np.empty(num_voxels, dtype=np.int32)

    for i, indices in enumerate(idx_map):
        n = len(indices)
        voxel_counts[i] = n

        if n == 1:
            idx = indices[0]
            c = float(confidences[idx])
            lbl = int(labels[idx])
            if c >= min_confidence and lbl != ignore_label:
                voxel_labels[i] = lbl
                voxel_confs[i] = c
                voxel_consensus[i] = 1.0
            else:
                voxel_labels[i] = ignore_label
                voxel_confs[i] = c
                voxel_consensus[i] = 0.0
            continue

        pts_lbl = labels[indices]
        pts_cnf = confidences[indices]

        # Consider points meeting the confidence threshold and not initially ignore
        valid_conf = (pts_cnf >= min_confidence) & (pts_lbl != ignore_label)
        if not np.any(valid_conf):
            voxel_labels[i] = ignore_label
            voxel_confs[i] = float(np.mean(pts_cnf))
            voxel_consensus[i] = 0.0
            continue

        v_lbl = pts_lbl[valid_conf]
        v_cnf = pts_cnf[valid_conf]

        # Majority voting among confident non-ignore points
        counts = Counter(v_lbl)
        maj_lbl, maj_cnt = counts.most_common(1)[0]
        consensus = maj_cnt / n  # consensus ratio relative to all points in voxel

        if consensus >= min_consensus:
            voxel_labels[i] = maj_lbl
            voxel_confs[i] = float(np.mean(v_cnf[v_lbl == maj_lbl]))
            voxel_consensus[i] = float(consensus)
        else:
            voxel_labels[i] = ignore_label
            voxel_confs[i] = float(np.mean(v_cnf))
            voxel_consensus[i] = float(consensus)

    return voxel_labels, voxel_confs, voxel_consensus, voxel_counts


def convert_single(
    rgb_path: str,
    out_path: str,
    voxel_size: float = 0.05,
    intrinsic: o3d.camera.PinholeCameraIntrinsic | None = None,
    min_confidence: float = 0.7,
    min_consensus: float = 0.6,
    ignore_label: int = 20,
    filter_boundaries: bool = True,
    save_sidecars: bool = True,
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

    # ── Load semantic labels & confidence ───────────────────────────
    semantic_path = rgb_path.replace(".png", "_semantic.npy")
    if os.path.isfile(semantic_path):
        semantic = np.load(semantic_path).squeeze()  # (H, W) int32
    else:
        # Default to ignored/void label (index 20)
        print(f"  WARNING: No semantic labels for {rgb_path}, using zeros")
        semantic = np.full((depth.shape[0], depth.shape[1]), ignore_label, dtype=np.int32)

    confidence_path = rgb_path.replace(".png", "_semantic_confidence.npy")
    if os.path.isfile(confidence_path):
        confidence = np.load(confidence_path).squeeze().astype(np.float32)
        if np.max(confidence) < 0.1:
            # Auto-calibrate legacy logsumexp compression over 150 ADE20K classes
            exp_s = 149.0 * confidence / np.clip(1.0 - confidence, 1e-6, 1.0)
            confidence = np.clip(np.log(np.clip(exp_s, 1e-6, None)), 0.0, 1.0).astype(np.float32)
    else:
        # If no confidence sidecar, give 1.0 to valid labels and 0.0 to ignored
        confidence = np.where(semantic != ignore_label, 1.0, 0.0).astype(np.float32)

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
    
    # Keep pixels that are masked as valid, have depth > 0, depth < 300, and finite
    valid = mask & (depth > 0) & (depth < 300.0) & np.isfinite(depth)

    # Ambiguous boundary filtering via depth gradient
    if filter_boundaries:
        depth_safe = np.where(valid, depth, 0.0)
        grad_x = np.abs(np.diff(depth_safe, axis=1, prepend=depth_safe[:, :1]))
        grad_y = np.abs(np.diff(depth_safe, axis=0, prepend=depth_safe[:1, :]))
        is_boundary = (grad_x > (0.05 * depth_safe + 0.1)) | (grad_y > (0.05 * depth_safe + 0.1))
        # Mask boundary pixels as unconfident so they don't corrupt voxel voting
        confidence = np.where(is_boundary, 0.0, confidence)
    
    d = depth[valid]
    u = uu[valid].astype(np.float64)
    v = vv[valid].astype(np.float64)
    
    X = (u - CX) * d / FX
    Y = (v - CY) * d / FY
    Z = d
    
    points = np.stack([X, Y, Z], axis=-1)
    colors = rgb[valid].astype(np.float64) / 255.0
    labels = semantic[valid]
    confs = confidence[valid]
    
    num_raw = len(points)
    
    # ── Build point cloud ───────────────────────────────────────────
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # ── Voxel downsample (with majority voting & confidence filtering)
    if voxel_size > 0:
        # Open3D's voxel_down_sample_and_trace needs boundaries
        min_bound = pcd.get_min_bound() - voxel_size * 0.5
        max_bound = pcd.get_max_bound() + voxel_size * 0.5
        pcd, _, idx_map = pcd.voxel_down_sample_and_trace(voxel_size, min_bound, max_bound)
        ds_labels, ds_confs, ds_consensus, ds_counts = voxel_majority_voting(
            labels, confs, idx_map, min_confidence, min_consensus, ignore_label)
    else:
        num_voxel = len(pcd.points)
        ds_labels = np.where((confs >= min_confidence), labels, ignore_label).astype(np.int32)
        ds_confs = confs.astype(np.float32)
        ds_consensus = np.ones(num_voxel, dtype=np.float32)
        ds_counts = np.ones(num_voxel, dtype=np.int32)
    
    num_voxel = len(pcd.points)
    ignored_voxels = int(np.count_nonzero(ds_labels == ignore_label))
    
    # ── Save as PLY + separate label & sidecar files ────────────────
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    o3d.io.write_point_cloud(out_path, pcd, write_ascii=False)
    
    # Save labels alongside PLY
    label_path = out_path.replace(".ply", "_labels.npy")
    np.save(label_path, ds_labels.astype(np.int32))

    if save_sidecars:
        conf_path = out_path.replace(".ply", "_confidence.npy")
        consensus_path = out_path.replace(".ply", "_consensus.npy")
        counts_path = out_path.replace(".ply", "_observations.npy")
        np.save(conf_path, ds_confs.astype(np.float32))
        np.save(consensus_path, ds_consensus.astype(np.float32))
        np.save(counts_path, ds_counts.astype(np.int32))

    return {
        "ok": True,
        "num_points_raw": num_raw,
        "num_points_voxel": num_voxel,
        "num_ignored_voxel": ignored_voxels,
        "ignored_ratio": float(ignored_voxels / max(num_voxel, 1)),
        "mean_confidence": float(np.mean(ds_confs)) if num_voxel > 0 else 0.0,
        "mean_consensus": float(np.mean(ds_consensus)) if num_voxel > 0 else 0.0,
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
    (
        rgb_path,
        out_path,
        voxel_size,
        min_confidence,
        min_consensus,
        ignore_label,
        filter_boundaries,
        save_sidecars,
    ) = args
    return rgb_path, convert_single(
        rgb_path=rgb_path,
        out_path=out_path,
        voxel_size=voxel_size,
        intrinsic=None,
        min_confidence=min_confidence,
        min_consensus=min_consensus,
        ignore_label=ignore_label,
        filter_boundaries=filter_boundaries,
        save_sidecars=save_sidecars,
    )


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
        "--min_confidence",
        type=float,
        default=0.7,
        help="Minimum teacher confidence threshold (default: 0.7).",
    )
    parser.add_argument(
        "--min_consensus",
        type=float,
        default=0.6,
        help="Minimum voxel majority voting consensus ratio (default: 0.6).",
    )
    parser.add_argument(
        "--ignore_label",
        type=int,
        default=20,
        help="Class ID for ignored / void points (default: 20).",
    )
    parser.add_argument(
        "--no_filter_boundaries",
        action="store_true",
        help="Disable depth gradient boundary filtering.",
    )
    parser.add_argument(
        "--no_sidecars",
        action="store_true",
        help="Do not save confidence, consensus, and observation sidecars.",
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
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Process at most this many RGB frames.",
    )
    args = parser.parse_args()

    filter_boundaries = not args.no_filter_boundaries
    save_sidecars = not args.no_sidecars

    # ── Discover all RGB images ─────────────────────────────────────
    print(f"Scanning {args.diode_root} for RGB images...")
    rgb_paths = find_rgb_images(args.diode_root)
    if not rgb_paths:
        print(f"ERROR: No .png files found under {args.diode_root}")
        sys.exit(1)
    if args.limit > 0:
        rgb_paths = rgb_paths[:args.limit]
    print(f"Found {len(rgb_paths)} RGB images.")

    # ── Build work list ─────────────────────────────────────────────
    work = []
    skipped_existing = 0
    for rgb_path in rgb_paths:
        out_path = derive_out_path(rgb_path, args.diode_root, args.out_dir)
        if args.skip_existing and os.path.isfile(out_path):
            skipped_existing += 1
            continue
        work.append((
            rgb_path,
            out_path,
            args.voxel_size,
            args.min_confidence,
            args.min_consensus,
            args.ignore_label,
            filter_boundaries,
            save_sidecars,
        ))

    if skipped_existing:
        print(f"Skipping {skipped_existing} already-converted scenes.")
    if not work:
        print("All scenes already converted. Nothing to do.")
        return

    print(f"Converting {len(work)} scenes (voxel_size={args.voxel_size}m, "
          f"min_conf={args.min_confidence}, min_cons={args.min_consensus}, "
          f"workers={args.workers})...")
    print()

    # ── Process ─────────────────────────────────────────────────────
    t0 = time.time()
    ok_count = 0
    fail_count = 0
    total_raw = 0
    total_voxel = 0
    total_ignored = 0

    if args.workers <= 1:
        # Serial — easier to debug
        intrinsic = make_intrinsic_matrix()
        for i, item in enumerate(work, 1):
            (
                rgb_path,
                out_path,
                voxel_size,
                min_conf,
                min_cons,
                ign_lbl,
                f_bounds,
                s_sides,
            ) = item
            result = convert_single(
                rgb_path=rgb_path,
                out_path=out_path,
                voxel_size=voxel_size,
                intrinsic=intrinsic,
                min_confidence=min_conf,
                min_consensus=min_cons,
                ignore_label=ign_lbl,
                filter_boundaries=f_bounds,
                save_sidecars=s_sides,
            )
            _report_progress(i, len(work), rgb_path, result)
            if result["ok"]:
                ok_count += 1
                total_raw += result["num_points_raw"]
                total_voxel += result["num_points_voxel"]
                total_ignored += result.get("num_ignored_voxel", 0)
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
                    total_ignored += result.get("num_ignored_voxel", 0)
                else:
                    fail_count += 1

    elapsed = time.time() - t0

    # ── Summary ─────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"  Conversion complete in {elapsed:.1f}s")
    print(f"  Success: {ok_count}  |  Failed/Skipped: {fail_count}")
    if ok_count > 0:
        print(f"  Avg raw points:       {total_raw // ok_count:,}")
        print(f"  Avg voxel points:     {total_voxel // ok_count:,}")
        print(f"  Total voxel points:   {total_voxel:,}")
        print(f"  Total ignored points: {total_ignored:,} ({100.0 * total_ignored / max(total_voxel, 1):.2f}%)")
    print(f"  Output directory: {args.out_dir}")
    print("=" * 60)


def _report_progress(i, total, rgb_path, result):
    """Print a single-line progress update."""
    name = os.path.basename(rgb_path).replace(".png", "")
    if result["ok"]:
        raw = result["num_points_raw"]
        vox = result["num_points_voxel"]
        ign = result.get("num_ignored_voxel", 0)
        ign_pct = result.get("ignored_ratio", 0.0) * 100
        print(f"  [{i:4d}/{total}] ✓ {name}  "
              f"({raw:,} → {vox:,} pts, {ign:,} ignored [{ign_pct:.1f}%])")
    else:
        print(f"  [{i:4d}/{total}] ✗ {name}  "
              f"({result['reason']})")


if __name__ == "__main__":
    main()
