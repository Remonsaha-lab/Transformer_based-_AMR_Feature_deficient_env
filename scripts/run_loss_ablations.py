#!/usr/bin/env python3
"""
Phase 4: Loss & Training Schedule Ablation Runner and Aggregator
================================================================
Orchestrates or summarizes Phase 4 ablations:
  - Loss Formulations: Unweighted CE, Inverse-Frequency CE, Median-Frequency CE, Focal Loss, CB-Focal
  - Learning Rates: 1e-3, 3e-3, 1e-2
  - Stage 1 Partition: Frozen vs Fine-Tuned

Aggregates metrics and per-class IoU (especially rare classes: Tree, Person, Furniture, Vehicle)
into:
  results/ablations/loss_ablation_summary.csv
  results/ablations/rare_class_analysis.csv
"""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Any


AMR10_CLASSES = [
    "ground", "road_path", "building_wall", "vegetation", "tree",
    "sky", "vehicle", "person", "furniture_obstacle", "other"
]

RARE_CLASSES = ["tree", "vehicle", "person", "furniture_obstacle"]


def summarize_ablation_results(results_dir: str, output_csv: str) -> None:
    """Scan ablation evaluation JSON files and generate comparative summary CSV."""
    root = Path(results_dir)
    json_files = sorted(root.glob("**/overall_metrics.json"))

    if not json_files:
        print(f"No overall_metrics.json files found under {results_dir}")
        return

    rows: List[Dict[str, Any]] = []

    for jf in json_files:
        run_name = jf.parent.name
        with open(jf, "r") as f:
            data = json.load(f)

        macro_miou = data.get("macro_miou", 0.0) * 100.0
        fw_iou = data.get("frequency_weighted_iou", 0.0) * 100.0
        oa = data.get("overall_accuracy", 0.0) * 100.0
        per_class_iou = data.get("per_class_iou", [])

        row = {
            "Experiment": run_name,
            "Macro_mIoU": f"{macro_miou:.2f}",
            "FW_IoU": f"{fw_iou:.2f}",
            "OA": f"{oa:.2f}",
        }

        # Add per-class IoU for rare classes
        for rare in RARE_CLASSES:
            if rare in AMR10_CLASSES:
                idx = AMR10_CLASSES.index(rare)
                if idx < len(per_class_iou):
                    val = per_class_iou[idx] * 100.0
                    row[f"IoU_{rare}"] = f"{val:.2f}"
                else:
                    row[f"IoU_{rare}"] = "N/A"

        rows.append(row)

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["Experiment", "Macro_mIoU", "FW_IoU", "OA"] + [f"IoU_{r}" for r in RARE_CLASSES]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Ablation summary saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 4 Loss Ablation Summary")
    parser.add_argument("--results-dir", type=str, default="results/ablations",
                        help="Directory containing ablation evaluation outputs")
    parser.add_argument("--output-csv", type=str, default="results/ablations/loss_ablation_summary.csv",
                        help="Path to output CSV")
    args = parser.parse_args()

    summarize_ablation_results(args.results_dir, args.output_csv)


if __name__ == "__main__":
    main()

