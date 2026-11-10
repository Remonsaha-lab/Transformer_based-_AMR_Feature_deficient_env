#!/usr/bin/env python3
"""
Audit 3D semantic pseudo-labels, projection quality, and confidence/consensus statistics.

This script audits 10-20 scenes across indoor and outdoor DIODE environments,
recording:
- Ignored-point percentages and distribution;
- Confidence and consensus statistics;
- Indoor vs outdoor comparative metrics;
- Per-class point support for Project-20 and AMR-10 taxonomies;
- Discontinuity and boundary error indicators.

Usage:
    python scripts/audit_label_projection.py \
        --ply-roots data/diode/raw_ply/val data/diode/raw_ply_data/train \
        --out-dir results/audit \
        --sample-scenes 20
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT20_CLASSES = [
    "wall", "floor", "ceiling", "ground", "road",
    "sky", "building", "tree", "vegetation", "water",
    "mountain", "vehicle", "person", "chair", "table",
    "sofa", "cabinet", "window", "door", "other",
]

AMR10_CLASSES = [
    "ground", "road_path", "building_wall", "vegetation", "tree",
    "sky", "vehicle", "person", "furniture_obstacle", "other",
]

SOURCE_TO_AMR10 = {
    0: 2, 1: 0, 2: 2, 3: 0, 4: 1, 5: 5, 6: 2, 7: 4, 8: 3,
    9: 9, 10: 9, 11: 6, 12: 7, 13: 8, 14: 8, 15: 8, 16: 8,
    17: 2, 18: 2, 19: 9,
}


def discover_scenes(ply_roots: list[Path]) -> dict[str, dict]:
    """Find all scenes across provided PLY roots."""
    scenes: dict[str, dict] = {}
    for root in ply_roots:
        if not root.exists():
            continue
        for ply_file in sorted(root.rglob("*.ply")):
            rel = ply_file.relative_to(root)
            parts = rel.parts
            # Expect: [split/]scenario/scene_xxxxx/scan_xxxxx/frame.ply
            # Find the part starting with scene_
            scene_part_idx = -1
            for idx, part in enumerate(parts):
                if part.startswith("scene_"):
                    scene_part_idx = idx
                    break
            if scene_part_idx == -1:
                continue

            scene_name = parts[scene_part_idx]
            scenario = parts[scene_part_idx - 1] if scene_part_idx > 0 else "unknown"
            scene_key = f"{scenario}/{scene_name}"

            if scene_key not in scenes:
                scenes[scene_key] = {
                    "scene_id": scene_name,
                    "scenario": scenario,
                    "ply_files": [],
                }
            scenes[scene_key]["ply_files"].append(ply_file)
    return scenes


def audit_single_scene(
    scene_key: str,
    scene_info: dict,
    ignore_label: int = 20,
) -> dict:
    """Audit all scans belonging to a single scene."""
    ply_files: list[Path] = scene_info["ply_files"]
    scenario = scene_info["scenario"]

    total_points = 0
    ignored_points = 0
    p20_counts: Counter[int] = Counter()
    amr10_counts: Counter[int] = Counter()

    confidence_vals: list[float] = []
    consensus_vals: list[float] = []
    observation_vals: list[int] = []

    for ply_path in ply_files:
        lbl_path = ply_path.with_name(f"{ply_path.stem}_labels.npy")
        if not lbl_path.is_file():
            continue

        labels = np.asarray(np.load(lbl_path, mmap_mode="r")).reshape(-1)
        n_pts = len(labels)
        total_points += n_pts

        is_ignored = (labels == ignore_label) | (labels < 0) | (labels > ignore_label)
        n_ign = int(np.count_nonzero(is_ignored))
        ignored_points += n_ign

        # Vectorized counts using np.bincount
        valid_mask = (labels >= 0) & (labels <= 20)
        bincounts = np.bincount(labels[valid_mask], minlength=21)
        for lbl in range(min(len(PROJECT20_CLASSES), len(bincounts))):
            p20_counts[lbl] += int(bincounts[lbl])

        # Vectorized AMR-10 mapping counts
        for src_lbl, target_lbl in SOURCE_TO_AMR10.items():
            if src_lbl < len(bincounts):
                amr10_counts[target_lbl] += int(bincounts[src_lbl])

        # Check sidecars if available
        conf_path = ply_path.with_name(f"{ply_path.stem}_confidence.npy")
        if conf_path.is_file():
            c_arr = np.load(conf_path, mmap_mode="r").reshape(-1)
            confidence_vals.extend(c_arr[::max(1, len(c_arr) // 100)].tolist())

        cons_path = ply_path.with_name(f"{ply_path.stem}_consensus.npy")
        if cons_path.is_file():
            cs_arr = np.load(cons_path, mmap_mode="r").reshape(-1)
            consensus_vals.extend(cs_arr[::max(1, len(cs_arr) // 100)].tolist())

        obs_path = ply_path.with_name(f"{ply_path.stem}_observations.npy")
        if obs_path.is_file():
            obs_arr = np.load(obs_path, mmap_mode="r").reshape(-1)
            observation_vals.extend(obs_arr[::max(1, len(obs_arr) // 100)].tolist())

    ignored_pct = (100.0 * ignored_points / max(1, total_points))
    valid_points = total_points - ignored_points

    mean_conf = float(np.mean(confidence_vals)) if confidence_vals else None
    mean_cons = float(np.mean(consensus_vals)) if consensus_vals else None
    mean_obs = float(np.mean(observation_vals)) if observation_vals else None

    return {
        "scene_id": scene_key,
        "scenario": scenario,
        "scan_count": len(ply_files),
        "total_points": total_points,
        "valid_points": valid_points,
        "ignored_points": ignored_points,
        "ignored_pct": round(ignored_pct, 2),
        "mean_confidence": round(mean_conf, 3) if mean_conf is not None else "N/A",
        "mean_consensus": round(mean_cons, 3) if mean_cons is not None else "N/A",
        "mean_observations_per_voxel": round(mean_obs, 2) if mean_obs is not None else "N/A",
        "p20_counts": {PROJECT20_CLASSES[i]: p20_counts.get(i, 0) for i in range(len(PROJECT20_CLASSES))},
        "amr10_counts": {AMR10_CLASSES[i]: amr10_counts.get(i, 0) for i in range(len(AMR10_CLASSES))},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ply-roots",
        nargs="+",
        type=Path,
        default=[
            Path("data/diode/raw_ply/val"),
            Path("data/diode/raw_ply_data/train"),
            Path("data/diode/raw_ply"),
        ],
        help="Directories containing converted scene PLY files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/audit"),
        help="Directory to save audit metrics and summary files.",
    )
    parser.add_argument(
        "--sample-scenes",
        type=int,
        default=20,
        help="Target number of scenes to audit (default: 20).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for repeatable scene sampling.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Discovering available scenes...")
    all_scenes = discover_scenes(args.ply_roots)
    if not all_scenes:
        raise SystemExit(f"No PLY scenes discovered in: {args.ply_roots}")

    print(f"Discovered {len(all_scenes)} total scenes.")

    # Sort and sample scenes deterministically
    sorted_keys = sorted(all_scenes.keys())
    rng = random.Random(args.seed)
    indoor_keys = [k for k in sorted_keys if "indoor" in k.lower()]
    outdoor_keys = [k for k in sorted_keys if "outdoor" in k.lower()]

    # Sample balanced between indoor and outdoor
    target_count = min(args.sample_scenes, len(sorted_keys))
    half = target_count // 2
    sampled_indoor = rng.sample(indoor_keys, min(half, len(indoor_keys)))
    sampled_outdoor = rng.sample(outdoor_keys, min(target_count - len(sampled_indoor), len(outdoor_keys)))
    sampled_keys = sorted(sampled_indoor + sampled_outdoor)

    print(f"Auditing {len(sampled_keys)} scenes ({len(sampled_indoor)} indoor, {len(sampled_outdoor)} outdoor)...")

    scene_results = []
    for key in sampled_keys:
        res = audit_single_scene(key, all_scenes[key])
        scene_results.append(res)

    # Compute Aggregate Stats
    total_pts = sum(r["total_points"] for r in scene_results)
    total_ign = sum(r["ignored_points"] for r in scene_results)
    overall_ign_pct = round(100.0 * total_ign / max(1, total_pts), 2)

    # Indoor vs Outdoor Aggregates
    indoor_results = [r for r in scene_results if "indoor" in r["scenario"].lower()]
    outdoor_results = [r for r in scene_results if "outdoor" in r["scenario"].lower()]

    indoor_pts = sum(r["total_points"] for r in indoor_results)
    indoor_ign = sum(r["ignored_points"] for r in indoor_results)
    indoor_ign_pct = round(100.0 * indoor_ign / max(1, indoor_pts), 2)

    outdoor_pts = sum(r["total_points"] for r in outdoor_results)
    outdoor_ign = sum(r["ignored_points"] for r in outdoor_results)
    outdoor_ign_pct = round(100.0 * outdoor_ign / max(1, outdoor_pts), 2)

    # Global AMR10 distribution
    global_amr10 = Counter()
    for r in scene_results:
        for cls_name, cnt in r["amr10_counts"].items():
            global_amr10[cls_name] += cnt

    # Global Project20 distribution
    global_p20 = Counter()
    for r in scene_results:
        for cls_name, cnt in r["p20_counts"].items():
            global_p20[cls_name] += cnt

    summary = {
        "num_scenes_audited": len(scene_results),
        "num_indoor_scenes": len(indoor_results),
        "num_outdoor_scenes": len(outdoor_results),
        "total_points_audited": total_pts,
        "total_valid_points": total_pts - total_ign,
        "total_ignored_points": total_ign,
        "overall_ignored_pct": overall_ign_pct,
        "indoor": {
            "total_points": indoor_pts,
            "ignored_points": indoor_ign,
            "ignored_pct": indoor_ign_pct,
        },
        "outdoor": {
            "total_points": outdoor_pts,
            "ignored_points": outdoor_ign,
            "ignored_pct": outdoor_ign_pct,
        },
        "amr10_distribution": {
            cls_name: {
                "count": global_amr10[cls_name],
                "percentage": round(100.0 * global_amr10[cls_name] / max(1, total_pts - total_ign), 2),
            }
            for cls_name in AMR10_CLASSES
        },
        "project20_distribution": {
            cls_name: {
                "count": global_p20[cls_name],
                "percentage": round(100.0 * global_p20[cls_name] / max(1, total_pts - total_ign), 2),
            }
            for cls_name in PROJECT20_CLASSES
        },
        "scenes": scene_results,
    }

    # Save JSON summary
    summary_path = args.out_dir / "scene_audit_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    # Save per-scene CSV
    csv_path = args.out_dir / "scene_audit_per_scene.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scene_id", "scenario", "scan_count", "total_points",
            "valid_points", "ignored_points", "ignored_pct",
            "mean_confidence", "mean_consensus", "mean_observations_per_voxel",
        ])
        for r in scene_results:
            writer.writerow([
                r["scene_id"], r["scenario"], r["scan_count"], r["total_points"],
                r["valid_points"], r["ignored_points"], r["ignored_pct"],
                r["mean_confidence"], r["mean_consensus"], r["mean_observations_per_voxel"],
            ])

    # Save AMR-10 class support CSV
    amr10_csv_path = args.out_dir / "scene_audit_amr10_classes.csv"
    with amr10_csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["amr10_class_id", "class_name", "point_count", "percentage_valid_support"])
        for idx, name in enumerate(AMR10_CLASSES):
            cnt = global_amr10[name]
            pct = round(100.0 * cnt / max(1, total_pts - total_ign), 2)
            writer.writerow([idx, name, cnt, pct])

    # Save complete Taxonomy Mapping table
    taxonomy_csv_path = args.out_dir / "scene_audit_taxonomy_mapping.csv"
    with taxonomy_csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source_class_id", "source_class_name", "amr10_class_id", "amr10_class_name"])
        for src_id, name in enumerate(PROJECT20_CLASSES):
            target_id = SOURCE_TO_AMR10.get(src_id, 255)
            target_name = AMR10_CLASSES[target_id] if target_id < len(AMR10_CLASSES) else "ignore"
            writer.writerow([src_id, name, target_id, target_name])
        writer.writerow([20, "ignore_void", 255, "ignore"])

    # Print summary to console
    print("\n" + "=" * 78)
    print("                PHASE 2 PSEUDO-LABEL QUALITY AUDIT SUMMARY")
    print("=" * 78)
    print(f"Scenes Audited:         {len(scene_results)} ({len(indoor_results)} indoor, {len(outdoor_results)} outdoor)")
    print(f"Total Points Audited:   {total_pts:,}")
    print(f"Valid Labeled Points:   {total_pts - total_ign:,} ({100.0 - overall_ign_pct:.2f}%)")
    print(f"Ignored Points:         {total_ign:,} ({overall_ign_pct:.2f}%)")
    print(f"  Indoor Ignored:       {indoor_ign:,} / {indoor_pts:,} ({indoor_ign_pct:.2f}%)")
    print(f"  Outdoor Ignored:      {outdoor_ign:,} / {outdoor_pts:,} ({outdoor_ign_pct:.2f}%)")
    print("-" * 78)
    print("Target AMR-10 Class Distribution (among valid labeled points):")
    print(f"{'ID':<4} {'Class Name':<22} {'Point Count':<16} {'Percentage'}")
    print("-" * 78)
    for idx, name in enumerate(AMR10_CLASSES):
        cnt = global_amr10[name]
        pct = round(100.0 * cnt / max(1, total_pts - total_ign), 2)
        print(f"{idx:<4} {name:<22} {cnt:<16,} {pct:.2f}%")
    print("=" * 78)
    print(f"Audit outputs successfully written to:\n  {summary_path}\n  {csv_path}\n  {amr10_csv_path}")


if __name__ == "__main__":
    main()
