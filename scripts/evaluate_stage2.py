#!/usr/bin/env python3
"""Evaluate saved Stage 2 semantic predictions independently of training."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_CLASS_NAMES = [
    "ground",
    "road_path",
    "building_wall",
    "vegetation",
    "tree",
    "sky",
    "vehicle",
    "person",
    "furniture_obstacle",
    "other",
]


def _load_array(path: Path, preferred_keys: Iterable[str]) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(path))
    if suffix == ".npz":
        with np.load(path) as archive:
            for key in preferred_keys:
                if key in archive:
                    return np.asarray(archive[key])
            if len(archive.files) == 1:
                return np.asarray(archive[archive.files[0]])
            raise ValueError(
                f"Could not find one of {list(preferred_keys)} in {path}; "
                f"available keys: {archive.files}")
    if suffix in {".h5", ".hdf5"}:
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError("Reading HDF5 requires h5py") from exc
        with h5py.File(path, "r") as handle:
            datasets = {}

            def visit(name, value):
                if isinstance(value, h5py.Dataset):
                    datasets[name] = value

            handle.visititems(visit)
            for key in preferred_keys:
                matches = [
                    dataset for name, dataset in datasets.items()
                    if (name == key or name.endswith(f"/{key}"))
                    and "/_inc_dict/" not in name
                    and "/_slice_dict/" not in name]
                if matches:
                    return np.asarray(matches[0])
            if any(key in {"target", "targets", "y", "labels", "semantic"}
                   for key in preferred_keys):
                group = handle.get("level_0/_csr_/y")
                if group is not None:
                    pointers = np.asarray(group["pointers"], dtype=np.int64)
                    columns = np.asarray(group["columns"], dtype=np.int64)
                    values = np.asarray(group["values"])
                    labels = np.zeros(len(pointers) - 1, dtype=np.int64)
                    for row in range(len(labels)):
                        start, end = pointers[row], pointers[row + 1]
                        if end > start:
                            row_values = values[start:end]
                            row_columns = columns[start:end]
                            labels[row] = row_columns[np.argmax(row_values)]
                    return labels
            available = sorted(datasets)
            raise ValueError(
                f"Could not find one of {list(preferred_keys)} in {path}; "
                f"available datasets: {available}")
    raise ValueError(f"Unsupported label file format: {path.suffix}")


def load_predictions(path: Path) -> np.ndarray:
    return _load_array(path, ("pred", "prediction", "semantic_pred", "labels"))


def load_targets(path: Path) -> np.ndarray:
    return _load_array(path, ("target", "targets", "y", "labels", "semantic"))


def validate_pair(predictions: np.ndarray, targets: np.ndarray, num_classes: int) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.asarray(predictions).reshape(-1)
    targets = np.asarray(targets).reshape(-1)
    if predictions.shape != targets.shape:
        raise ValueError(
            f"Prediction/target alignment failure: {predictions.shape} != {targets.shape}")
    if not np.issubdtype(predictions.dtype, np.integer):
        if not np.all(np.isfinite(predictions)) or not np.all(predictions == predictions.astype(np.int64)):
            raise ValueError("Predictions must contain finite integer class IDs")
        predictions = predictions.astype(np.int64)
    else:
        predictions = predictions.astype(np.int64, copy=False)
    targets = targets.astype(np.int64, copy=False)
    if np.any((predictions < 0) | (predictions >= num_classes)):
        bad = np.unique(predictions[(predictions < 0) | (predictions >= num_classes)])
        raise ValueError(f"Predictions contain invalid class IDs: {bad.tolist()}")
    valid_targets = (targets >= 0) & (targets < num_classes)
    valid_targets |= targets == 255
    if not np.all(valid_targets):
        bad = np.unique(targets[~valid_targets])
        raise ValueError(f"Targets contain invalid class IDs: {bad.tolist()}")
    return predictions, targets


def confusion_matrix(predictions: np.ndarray, targets: np.ndarray, num_classes: int) -> np.ndarray:
    valid = (targets >= 0) & (targets < num_classes)
    indices = targets[valid] * num_classes + predictions[valid]
    return np.bincount(indices, minlength=num_classes * num_classes).reshape(
        num_classes, num_classes)


def metrics_from_confusion(matrix: np.ndarray) -> dict:
    matrix = np.asarray(matrix, dtype=np.int64)
    true_positive = np.diag(matrix).astype(np.float64)
    support = matrix.sum(axis=1).astype(np.float64)
    predicted = matrix.sum(axis=0).astype(np.float64)
    union = support + predicted - true_positive
    iou = np.full(matrix.shape[0], np.nan, dtype=np.float64)
    present = support > 0
    iou[present] = true_positive[present] / union[present]
    accuracy = np.full(matrix.shape[0], np.nan, dtype=np.float64)
    has_support = support > 0
    accuracy[has_support] = true_positive[has_support] / support[has_support]
    total = matrix.sum()
    return {
        "macro_miou": float(np.nanmean(iou)) if np.any(present) else 0.0,
        "fixed_closed_set_miou": float(np.nan_to_num(iou, nan=0.0).mean()),
        "frequency_weighted_iou": float(np.nansum(iou * support) / total) if total else 0.0,
        "overall_accuracy": float(true_positive.sum() / total) if total else 0.0,
        "mean_class_accuracy": float(np.nanmean(accuracy)) if np.any(has_support) else 0.0,
        "per_class_iou": iou.tolist(),
        "class_support": support.astype(np.int64).tolist(),
        "present_class": present.tolist(),
        "valid_count": int(total),
        "ignored_count": int(matrix.size and 0),
    }


def evaluate_pair(prediction_path: Path, target_path: Path, num_classes: int) -> tuple[dict, np.ndarray]:
    predictions, targets = validate_pair(
        load_predictions(prediction_path), load_targets(target_path), num_classes)
    matrix = confusion_matrix(predictions, targets, num_classes)
    result = metrics_from_confusion(matrix)
    result["ignored_count"] = int(np.count_nonzero(targets == 255))
    result["prediction_file"] = str(prediction_path)
    result["target_file"] = str(target_path)
    return result, matrix


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(result: dict, matrix: np.ndarray, output_dir: Path, class_names: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {key: value for key, value in result.items() if key != "per_class_iou"}
    summary["class_names"] = class_names
    summary["per_class_iou"] = {
        name: None if np.isnan(value) else value
        for name, value in zip(class_names, result["per_class_iou"])}
    (output_dir / "overall_metrics.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n")
    _write_csv(
        output_dir / "per_class_metrics.csv",
        [{
            "class_id": index,
            "class_name": name,
            "iou": result["per_class_iou"][index],
            "support": result["class_support"][index],
            "present": result["present_class"][index],
        } for index, name in enumerate(class_names)])
    _write_csv(
        output_dir / "per_scene_metrics.csv",
        [{key: value for key, value in result.items() if not isinstance(value, list)}])
    np.savetxt(output_dir / "confusion_matrix.csv", matrix, fmt="%d", delimiter=",")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-file", type=Path, required=True)
    parser.add_argument("--target-file", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--class-names", nargs="+", default=DEFAULT_CLASS_NAMES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.class_names) != args.num_classes:
        raise ValueError("--class-names count must equal --num-classes")
    result, matrix = evaluate_pair(args.pred_file, args.target_file, args.num_classes)
    write_outputs(result, matrix, args.out_dir, args.class_names)
    print(json.dumps({
        "macro_miou": result["macro_miou"],
        "fixed_closed_set_miou": result["fixed_closed_set_miou"],
        "frequency_weighted_iou": result["frequency_weighted_iou"],
        "overall_accuracy": result["overall_accuracy"],
        "ignored_count": result["ignored_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
