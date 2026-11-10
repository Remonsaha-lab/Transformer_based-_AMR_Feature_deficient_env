#!/usr/bin/env python3
"""Create reproducible scene-level manifests without moving source data."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np


SPLIT_NAMES = ("train", "val", "test")


def discover_scenes(raw_root: Path, source_splits: list[str]) -> dict[str, list[Path]]:
    scenes: dict[str, list[Path]] = {}
    for source_split in source_splits:
        split_root = raw_root / source_split
        for ply_path in sorted(split_root.rglob("*.ply")):
            relative = ply_path.relative_to(split_root)
            if len(relative.parts) < 2:
                raise ValueError(f"Expected a scene directory below {split_root}: {ply_path}")
            scene_key = "/".join((source_split, *relative.parts[:-2]))
            scenes.setdefault(scene_key, []).append(ply_path)
    return scenes


def allocate_scenes(scene_keys: list[str], seed: int) -> dict[str, list[str]]:
    shuffled = list(scene_keys)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    if total < 3:
        raise ValueError("At least three scenes are required for train/val/test manifests")
    train_count = max(1, round(total * 0.70))
    val_count = max(1, round(total * 0.15))
    if train_count + val_count >= total:
        val_count = 1
        train_count = total - 2
    return {
        "train": sorted(shuffled[:train_count]),
        "val": sorted(shuffled[train_count:train_count + val_count]),
        "test": sorted(shuffled[train_count + val_count:]),
    }


def allocate_scenario_balanced(scene_keys: list[str], seed: int) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for scene_key in scene_keys:
        scenario = scene_key.split("/")[-2]
        grouped.setdefault(scenario, []).append(scene_key)
    assignments = {split: [] for split in SPLIT_NAMES}
    rng = random.Random(seed)
    for scenario, keys in sorted(grouped.items()):
        shuffled = list(keys)
        rng.shuffle(shuffled)
        if len(shuffled) < 3:
            raise ValueError(
                f"Scenario '{scenario}' has fewer than three scenes; "
                "scenario-balanced splitting is not possible")
        for index, scene_key in enumerate(shuffled):
            assignments[SPLIT_NAMES[index % len(SPLIT_NAMES)]].append(scene_key)
    return {split: sorted(keys) for split, keys in assignments.items()}


def scene_statistics(scene_key: str, ply_paths: list[Path]) -> dict:
    counts: Counter[int] = Counter()
    label_files = 0
    points = 0
    for ply_path in ply_paths:
        label_path = ply_path.with_name(f"{ply_path.stem}_labels.npy")
        if not label_path.is_file():
            continue
        labels = np.asarray(np.load(label_path, mmap_mode="r")).reshape(-1)
        counts.update(int(value) for value in np.unique(labels))
        points += int(labels.size)
        label_files += 1
    return {
        "scene_id": scene_key,
        "source_split": scene_key.split("/", maxsplit=1)[0],
        "scan_count": len(ply_paths),
        "label_file_count": label_files,
        "label_point_count": points,
        "unique_source_labels": ";".join(str(value) for value in sorted(counts)),
    }


def write_manifests(
    output_dir: Path,
    assignments: dict[str, list[str]],
    source_splits: list[str],
    seed: int,
    scenario_balanced: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, scene_keys in assignments.items():
        (output_dir / f"{split}_scenes.txt").write_text(
            "\n".join(scene_keys) + "\n")
    (output_dir / "split_manifest.json").write_text(json.dumps({
        "seed": seed,
        "source_splits": source_splits,
        "unit": "scene",
        "strategy": "scenario_balanced" if scenario_balanced else "random_scene",
        "assignments": {
            split: scene_keys for split, scene_keys in assignments.items()
        },
    }, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data/diode/raw_ply"))
    parser.add_argument("--source-splits", nargs="+", default=["val"])
    parser.add_argument("--output-dir", type=Path, default=Path("active_learning/splits"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario-balanced", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenes = discover_scenes(args.raw_root, args.source_splits)
    assignments = (
        allocate_scenario_balanced(sorted(scenes), args.seed)
        if args.scenario_balanced
        else allocate_scenes(sorted(scenes), args.seed))
    if set().union(*[set(values) for values in assignments.values()]) != set(scenes):
        raise AssertionError("Some scenes were not assigned to a split")
    if len(set(assignments["train"]) & set(assignments["val"])) > 0:
        raise AssertionError("Train and validation scenes overlap")
    if len(set(assignments["train"]) & set(assignments["test"])) > 0:
        raise AssertionError("Train and test scenes overlap")
    if len(set(assignments["val"]) & set(assignments["test"])) > 0:
        raise AssertionError("Validation and test scenes overlap")

    rows = []
    for scene_key, ply_paths in scenes.items():
        row = scene_statistics(scene_key, ply_paths)
        row["assigned_split"] = next(
            split for split, keys in assignments.items() if scene_key in keys)
        rows.append(row)
    rows.sort(key=lambda row: row["scene_id"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "split_statistics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_manifests(
        args.output_dir,
        assignments,
        args.source_splits,
        args.seed,
        args.scenario_balanced,
    )
    print(json.dumps({split: len(keys) for split, keys in assignments.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
