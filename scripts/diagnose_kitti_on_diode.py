#!/usr/bin/env python3
"""
Run a semantic checkpoint on DIODE scenes for qualitative diagnostics.

This script is intentionally tolerant of dataset/class mismatch. It loads only
the checkpoint tensors whose names and shapes match the instantiated model, then
runs forward inference on DIODE scenes and saves visual comparisons:

  - RGB
  - DIODE labels from the processed data, if present
  - checkpoint predictions, colored by checkpoint class index
  - DIODE superpoints

It also writes per-scene and aggregate raw-index metrics. These metrics compare
prediction index N against DIODE label index N. They are useful for debugging
transfer behavior, but are not a valid semantic benchmark unless you define a
proper KITTI-to-DIODE class mapping.

Example:
  cd /home/remon/EZ_SP_CFDVri/superpoint_transformer

  python scripts/diagnose_kitti_on_diode.py \
      experiment=semantic/kitti360_ezsp \
      datamodule=semantic/diode_ezsp \
      datamodule.data_dir=data/diode \
      model.train_cnn_from_scratch=True \
      +diag_ckpt_path=/path/to/kitti360_ezsp.ckpt \
      +diag_stage=val \
      +diag_num_scenes=5 \
      +diag_num_images=5 \
      +diag_output_dir=outputs/kitti_on_diode_diag
"""

import csv
import glob
import os
import os.path as osp
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import hydra
import matplotlib
import numpy as np
import pyrootutils
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

root = Path(pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "README.md"],
    pythonpath=True,
    dotenv=True,
))

from src import utils  # noqa: E402
from src.data import NAG, NAGBatch  # noqa: E402

log = utils.get_pylogger(__name__)

if not OmegaConf.has_resolver("eval"):
    OmegaConf.register_new_resolver("eval", eval)


DIODE_COLORS = np.array([
    [200, 50, 50], [130, 130, 200], [230, 230, 130], [140, 100, 60],
    [100, 100, 100], [80, 180, 230], [150, 100, 50], [30, 140, 30],
    [100, 200, 100], [30, 100, 200], [160, 140, 120], [230, 130, 30],
    [230, 180, 180], [130, 50, 160], [180, 120, 60], [200, 80, 120],
    [80, 140, 140], [200, 230, 230], [100, 50, 30], [180, 180, 180],
    [0, 0, 0],
], dtype=np.float32) / 255.0

KITTI360_COLORS = np.array([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70],
], dtype=np.float32) / 255.0

DIODE_CLASS_NAMES = [
    "wall", "floor", "ceiling", "ground", "road",
    "sky", "building", "tree", "vegetation", "water",
    "mountain", "vehicle", "person", "chair", "table",
    "sofa", "cabinet", "window", "door", "other",
]

KITTI360_CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck",
]

MAPPED_CLASS_NAMES = [
    "road",
    "building",
    "wall",
    "other",
    "vegetation",
    "sky",
    "person",
    "vehicle",
]

# Cross-dataset evaluation mapping requested by the user.
# KITTI-360 predictions are mapped to this 8-class evaluation space.
KITTI_TO_MAPPED = np.array([
    0,  # road -> road
    0,  # sidewalk -> road
    1,  # building -> building
    2,  # wall -> wall
    3,  # fence -> other
    3,  # pole -> other
    3,  # traffic light -> other
    3,  # traffic sign -> other
    4,  # vegetation -> vegetation
    3,  # terrain -> other
    5,  # sky -> sky
    6,  # person -> person
    6,  # rider -> person
    7,  # car -> vehicle
    7,  # truck -> vehicle
], dtype=np.int64)

# DIODE labels not listed here are ignored in mapped metrics.
DIODE_TO_MAPPED = np.array([
    2,   # wall -> wall
    -1,  # floor ignored
    -1,  # ceiling ignored
    -1,  # ground ignored
    0,   # road -> road
    5,   # sky -> sky
    1,   # building -> building
    4,   # tree -> vegetation
    4,   # vegetation -> vegetation
    -1,  # water ignored
    -1,  # mountain ignored
    7,   # vehicle -> vehicle
    6,   # person -> person
    -1,  # chair ignored
    -1,  # table ignored
    -1,  # sofa ignored
    -1,  # cabinet ignored
    -1,  # window ignored
    -1,  # door ignored
    3,   # other -> other
], dtype=np.int64)


def checkpoint_num_classes(state_dict: Dict[str, torch.Tensor]) -> int:
    """Infer semantic head output dimension from a checkpoint."""
    candidates = [
        (k, v) for k, v in state_dict.items()
        if k.startswith("head") and k.endswith("weight") and v.ndim == 2
    ]
    if not candidates:
        raise ValueError("Could not infer checkpoint class count from head.*.weight")
    # Multi-stage models have several heads; all should share num_classes.
    return int(candidates[0][1].shape[0])


def load_matching_weights(model: torch.nn.Module, ckpt_path: str) -> Tuple[int, List[str], List[str]]:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    checkpoint_state = checkpoint.get("state_dict", checkpoint)

    # Do not call model.state_dict() here. This repo's MultiLoss overrides
    # state_dict() and may KeyError when class weights are absent. For this
    # diagnostic we only need loadable model tensors, not loss-state tensors.
    model_state = {
        name: tensor
        for name, tensor in list(model.named_parameters()) + list(model.named_buffers())
    }

    loadable = {}
    skipped = []
    for key, value in checkpoint_state.items():
        if key not in model_state:
            skipped.append(f"{key}: not in model")
            continue
        if tuple(value.shape) != tuple(model_state[key].shape):
            skipped.append(
                f"{key}: checkpoint {tuple(value.shape)} != model {tuple(model_state[key].shape)}")
            continue
        loadable[key] = value

    torch.nn.Module.load_state_dict(model, loadable, strict=False)
    missing = [key for key in model_state.keys() if key not in loadable]
    return len(loadable), skipped, missing


def labels_to_atomic(labels: torch.Tensor) -> np.ndarray:
    arr = labels.detach().cpu().numpy()
    if arr.ndim > 1:
        arr = arr.argmax(axis=-1)
    return arr.astype(np.int64).reshape(-1)


def colorize_labels(labels: np.ndarray, palette: np.ndarray) -> np.ndarray:
    idx = np.clip(labels, 0, len(palette) - 1)
    return palette[idx]


def distinct_colors(n: int, seed: int = 13) -> np.ndarray:
    import colorsys

    if n <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    rng = np.random.RandomState(seed)
    hues = np.linspace(0, 1, n, endpoint=False)
    rng.shuffle(hues)
    return np.array([colorsys.hsv_to_rgb(h, 0.75, 0.95) for h in hues], dtype=np.float32)


def colorize_superpoints(super_index: torch.Tensor) -> np.ndarray:
    sp = super_index.detach().cpu().numpy().astype(np.int64)
    colors = distinct_colors(int(sp.max()) + 1 if sp.size else 0)
    return colors[sp]


def plot_cloud(ax, pos: np.ndarray, colors: np.ndarray, title: str, max_points: int) -> None:
    if pos.shape[0] > max_points:
        rng = np.random.RandomState(0)
        idx = rng.choice(pos.shape[0], max_points, replace=False)
        pos = pos[idx]
        colors = colors[idx]

    ax.scatter(pos[:, 0], pos[:, 1], c=colors, s=0.7, linewidths=0)
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def confusion_matrix(pred: np.ndarray, target: np.ndarray, num_classes: int) -> np.ndarray:
    valid = (target >= 0) & (target < num_classes) & (pred >= 0) & (pred < num_classes)
    if not np.any(valid):
        return np.zeros((num_classes, num_classes), dtype=np.int64)
    flat = target[valid] * num_classes + pred[valid]
    return np.bincount(flat, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def metrics_from_confusion(confmat: np.ndarray) -> Dict[str, float]:
    total = confmat.sum()
    oa = float(np.trace(confmat) / total) if total > 0 else float("nan")

    tp = np.diag(confmat).astype(np.float64)
    fp = confmat.sum(axis=0) - tp
    fn = confmat.sum(axis=1) - tp
    union = tp + fp + fn
    valid = union > 0
    miou = float(np.mean(tp[valid] / union[valid])) if np.any(valid) else float("nan")
    return {"miou": miou, "oa": oa}


def per_class_metrics_from_confusion(
        confmat: np.ndarray,
        kitti_names: List[str],
        diode_names: List[str],
) -> List[Dict[str, object]]:
    rows = []
    skipped_rows = []
    tp = np.diag(confmat).astype(np.float64)
    pred_count = confmat.sum(axis=0).astype(np.float64)
    target_count = confmat.sum(axis=1).astype(np.float64)
    fp = pred_count - tp
    fn = target_count - tp
    union = tp + fp + fn

    for class_index in range(confmat.shape[0]):
        iou = tp[class_index] / union[class_index] if union[class_index] > 0 else float("nan")
        recall = (
            tp[class_index] / target_count[class_index]
            if target_count[class_index] > 0 else float("nan"))
        precision = (
            tp[class_index] / pred_count[class_index]
            if pred_count[class_index] > 0 else float("nan"))
        rows.append({
            "class_index": class_index,
            "kitti_prediction_class": (
                kitti_names[class_index] if class_index < len(kitti_names) else ""),
            "diode_target_class": (
                diode_names[class_index] if class_index < len(diode_names) else ""),
            "iou": float(iou),
            "class_accuracy_recall": float(recall),
            "prediction_precision": float(precision),
            "true_positive_points": int(tp[class_index]),
            "diode_target_points": int(target_count[class_index]),
            "kitti_predicted_points": int(pred_count[class_index]),
            "false_positive_points": int(fp[class_index]),
            "false_negative_points": int(fn[class_index]),
        })
    return rows


def write_per_class_metrics(path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "class_index",
        "kitti_prediction_class",
        "diode_target_class",
        "iou",
        "class_accuracy_recall",
        "prediction_precision",
        "true_positive_points",
        "diode_target_points",
        "kitti_predicted_points",
        "false_positive_points",
        "false_negative_points",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mapped_confusion_matrix(pred_kitti: np.ndarray, target_diode: np.ndarray) -> np.ndarray:
    valid_pred = (pred_kitti >= 0) & (pred_kitti < len(KITTI_TO_MAPPED))
    valid_target = (target_diode >= 0) & (target_diode < len(DIODE_TO_MAPPED))
    valid = valid_pred & valid_target
    if not np.any(valid):
        return np.zeros((len(MAPPED_CLASS_NAMES), len(MAPPED_CLASS_NAMES)), dtype=np.int64)

    pred_mapped = KITTI_TO_MAPPED[pred_kitti[valid]]
    target_mapped = DIODE_TO_MAPPED[target_diode[valid]]
    valid_mapped = (pred_mapped >= 0) & (target_mapped >= 0)
    return confusion_matrix(
        pred_mapped[valid_mapped],
        target_mapped[valid_mapped],
        len(MAPPED_CLASS_NAMES))


def mapped_per_class_metrics_from_confusion(confmat: np.ndarray) -> List[Dict[str, object]]:
    rows = []
    tp = np.diag(confmat).astype(np.float64)
    pred_count = confmat.sum(axis=0).astype(np.float64)
    target_count = confmat.sum(axis=1).astype(np.float64)
    fp = pred_count - tp
    fn = target_count - tp
    union = tp + fp + fn

    for class_index, name in enumerate(MAPPED_CLASS_NAMES):
        iou = tp[class_index] / union[class_index] if union[class_index] > 0 else float("nan")
        recall = (
            tp[class_index] / target_count[class_index]
            if target_count[class_index] > 0 else float("nan"))
        precision = (
            tp[class_index] / pred_count[class_index]
            if pred_count[class_index] > 0 else float("nan"))
        rows.append({
            "mapped_class_index": class_index,
            "mapped_class": name,
            "iou": float(iou),
            "class_accuracy_recall": float(recall),
            "prediction_precision": float(precision),
            "true_positive_points": int(tp[class_index]),
            "diode_target_points": int(target_count[class_index]),
            "kitti_predicted_points": int(pred_count[class_index]),
            "false_positive_points": int(fp[class_index]),
            "false_negative_points": int(fn[class_index]),
        })
    return rows


def write_mapped_per_class_metrics(path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "mapped_class_index",
        "mapped_class",
        "iou",
        "class_accuracy_recall",
        "prediction_precision",
        "true_positive_points",
        "diode_target_points",
        "kitti_predicted_points",
        "false_positive_points",
        "false_negative_points",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_mapping_table(path: Path) -> None:
    rows = []
    for kitti_id, mapped_id in enumerate(KITTI_TO_MAPPED.tolist()):
        rows.append({
            "source": "kitti360_prediction",
            "source_index": kitti_id,
            "source_class": KITTI360_CLASS_NAMES[kitti_id],
            "mapped_index": mapped_id,
            "mapped_class": MAPPED_CLASS_NAMES[mapped_id],
            "used_for_metrics": True,
        })
    for diode_id, mapped_id in enumerate(DIODE_TO_MAPPED.tolist()):
        rows.append({
            "source": "diode_target",
            "source_index": diode_id,
            "source_class": DIODE_CLASS_NAMES[diode_id],
            "mapped_index": mapped_id if mapped_id >= 0 else "",
            "mapped_class": MAPPED_CLASS_NAMES[mapped_id] if mapped_id >= 0 else "ignored",
            "used_for_metrics": mapped_id >= 0,
        })

    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "source",
            "source_index",
            "source_class",
            "mapped_index",
            "mapped_class",
            "used_for_metrics",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compute_raw_index_metrics(
        pred_l0: np.ndarray,
        diode_y: np.ndarray,
        ckpt_classes: int,
        diode_classes: int,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Compute diagnostic metrics without semantic class remapping."""
    valid_target = (diode_y >= 0) & (diode_y < diode_classes)

    raw_classes = max(ckpt_classes, diode_classes)
    raw_conf = confusion_matrix(
        pred_l0[valid_target],
        diode_y[valid_target],
        raw_classes)
    raw = metrics_from_confusion(raw_conf)

    overlap_classes = min(ckpt_classes, diode_classes)
    overlap_target = valid_target & (diode_y < overlap_classes)
    overlap_pred = pred_l0 < overlap_classes
    overlap_conf = confusion_matrix(
        pred_l0[overlap_target & overlap_pred],
        diode_y[overlap_target & overlap_pred],
        overlap_classes)
    overlap = metrics_from_confusion(overlap_conf)

    metrics = {
        "raw_miou": raw["miou"],
        "raw_oa": raw["oa"],
        "overlap_miou": overlap["miou"],
        "overlap_oa": overlap["oa"],
        "metric_points": int(valid_target.sum()),
        "overlap_metric_points": int((overlap_target & overlap_pred).sum()),
    }
    return metrics, raw_conf, overlap_conf


def compute_scene_metrics(
        nag: NAG,
        pred_level1: torch.Tensor,
        ckpt_classes: int,
        diode_classes: int,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    l0 = nag[0].detach().cpu()
    pred_l0 = pred_level1.detach().cpu()[l0.super_index.cpu()].numpy().astype(np.int64)

    if getattr(l0, "y", None) is not None:
        diode_y = labels_to_atomic(l0.y)
    else:
        diode_y = np.full(pred_l0.shape[0], -1, dtype=np.int64)

    hist = np.bincount(pred_l0, minlength=ckpt_classes)
    metrics, raw_conf, overlap_conf = compute_raw_index_metrics(
        pred_l0=pred_l0,
        diode_y=diode_y,
        ckpt_classes=ckpt_classes,
        diode_classes=diode_classes)
    mapped_conf = mapped_confusion_matrix(pred_l0, diode_y)
    mapped = metrics_from_confusion(mapped_conf)
    metrics.update({
        "mapped_miou": mapped["miou"],
        "mapped_oa": mapped["oa"],
        "mapped_metric_points": int(mapped_conf.sum()),
    })
    metrics.update({f"pred_class_{i}": int(v) for i, v in enumerate(hist)})
    return metrics, raw_conf, overlap_conf, mapped_conf


def save_scene_plot(
        nag: NAG,
        pred_level1: torch.Tensor,
        cloud_id: str,
        out_path: Path,
        max_points: int,
        ckpt_classes: int,
        diode_classes: int,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    l0 = nag[0].detach().cpu()
    pos = l0.pos.numpy()
    rgb = l0.rgb.numpy() if getattr(l0, "rgb", None) is not None else np.full((pos.shape[0], 3), 0.7)
    rgb = np.clip(rgb, 0, 1)

    pred_l0 = pred_level1.detach().cpu()[l0.super_index.cpu()].numpy().astype(np.int64)
    pred_colors = colorize_labels(pred_l0, KITTI360_COLORS)
    sp_colors = colorize_superpoints(l0.super_index)

    if getattr(l0, "y", None) is not None:
        diode_y = labels_to_atomic(l0.y)
        diode_colors = colorize_labels(diode_y, DIODE_COLORS)
    else:
        diode_y = np.full(pos.shape[0], -1, dtype=np.int64)
        diode_colors = np.full((pos.shape[0], 3), 0.25)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=130)
    plot_cloud(axes[0, 0], pos, rgb, "DIODE RGB", max_points)
    plot_cloud(axes[0, 1], pos, diode_colors, "DIODE labels / pseudo-labels", max_points)
    plot_cloud(axes[1, 0], pos, pred_colors, "KITTI checkpoint prediction indices", max_points)
    plot_cloud(axes[1, 1], pos, sp_colors, "DIODE superpoints", max_points)
    fig.suptitle(cloud_id, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    hist = np.bincount(pred_l0, minlength=ckpt_classes)
    metrics, raw_conf, overlap_conf = compute_raw_index_metrics(
        pred_l0=pred_l0,
        diode_y=diode_y,
        ckpt_classes=ckpt_classes,
        diode_classes=diode_classes)
    mapped_conf = mapped_confusion_matrix(pred_l0, diode_y)
    mapped = metrics_from_confusion(mapped_conf)
    metrics.update({
        "mapped_miou": mapped["miou"],
        "mapped_oa": mapped["oa"],
        "mapped_metric_points": int(mapped_conf.sum()),
    })
    metrics.update({f"pred_class_{i}": int(v) for i, v in enumerate(hist)})
    return metrics, raw_conf, overlap_conf, mapped_conf


def parse_indices(scene_indices, total: int, num_scenes: int) -> List[int]:
    if scene_indices is not None:
        if isinstance(scene_indices, Iterable) and not isinstance(scene_indices, (str, bytes)):
            return [int(i) for i in scene_indices]
        return [int(scene_indices)]
    if num_scenes >= total:
        return list(range(total))
    step = total / num_scenes
    return [int(i * step) for i in range(num_scenes)]


def parse_image_indices(image_indices, metric_indices: List[int], num_images: int) -> List[int]:
    """Return dataset indices for which PNG visualizations should be saved."""
    if image_indices is not None:
        if isinstance(image_indices, Iterable) and not isinstance(image_indices, (str, bytes)):
            return [int(i) for i in image_indices]
        return [int(image_indices)]
    if num_images < 0 or num_images >= len(metric_indices):
        return list(metric_indices)
    if num_images == 0:
        return []
    if num_images == 1:
        return [metric_indices[0]]
    positions = np.linspace(0, len(metric_indices) - 1, num_images).round().astype(int)
    return [metric_indices[int(i)] for i in positions]


def parse_diode_stage(stage: str) -> Tuple[str, str]:
    """Return the base DIODE split and optional scene-type filter."""
    scene_type = "all"
    base_stage = stage
    for suffix in ("indoors", "outdoor"):
        token = f"_{suffix}"
        if stage.endswith(token):
            scene_type = suffix
            base_stage = stage[:-len(token)]
            break
    return base_stage, scene_type


def discover_diode_cloud_ids(data_dir: str, stage: str) -> List[str]:
    """Discover DIODE cloud ids without instantiating the dataset.

    This lets us restrict preprocessing before BaseDataset starts processing.
    """
    base_stage, scene_type = parse_diode_stage(stage)
    raw_root_name = "raw_ply_data" if base_stage in ("train", "test") else "raw_ply"
    split_dir = osp.join(data_dir, raw_root_name, base_stage)
    if not osp.isdir(split_dir):
        return []

    ply_files = sorted(glob.glob(osp.join(split_dir, "**", "*.ply"), recursive=True))
    ids = []
    for ply_path in ply_files:
        rel_path = osp.relpath(ply_path, split_dir)
        cloud_id = osp.splitext(rel_path)[0]
        if scene_type == "all" or scene_type in cloud_id:
            ids.append(cloud_id)
    return ids


@hydra.main(version_base="1.2", config_path=str(root / "configs"), config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    ckpt_path = cfg.get("diag_ckpt_path")
    if ckpt_path is None:
        raise ValueError("Pass +diag_ckpt_path=/path/to/kitti360_ezsp.ckpt")
    ckpt_path = str(ckpt_path)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)

    output_dir = Path(cfg.get("diag_output_dir", "outputs/kitti_on_diode_diag"))
    stage = str(cfg.get("diag_stage", "val"))
    num_scenes = int(cfg.get("diag_num_scenes", 5))
    num_metric_scenes = int(cfg.get("diag_num_metric_scenes", num_scenes))
    num_images = int(cfg.get("diag_num_images", min(num_scenes, 10)))
    max_points = int(cfg.get("diag_max_points", 70000))
    device = torch.device(str(cfg.get("diag_device", "cuda" if torch.cuda.is_available() else "cpu")))

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    ckpt_classes = checkpoint_num_classes(state_dict)

    with open_dict(cfg):
        cfg.model.num_classes = ckpt_classes
        cfg.model.train_cnn_from_scratch = True
        cfg.model.freeze_cnn = False
        cfg.train = False
        cfg.test = False

    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Checkpoint classes inferred from head: {ckpt_classes}")
    log.info(f"Instantiating DIODE datamodule <{cfg.datamodule._target_}>")
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model)
    loaded_count, skipped, missing = load_matching_weights(model, ckpt_path)
    model = model.to(device).eval()

    log.info(f"Loaded {loaded_count} tensors from checkpoint")
    log.info(f"Skipped {len(skipped)} checkpoint tensors; model kept {len(missing)} initialized tensors")
    (output_dir / "load_report.txt").write_text(
        "Loaded tensors: " + str(loaded_count) + "\n\n"
        "Skipped checkpoint tensors:\n" + "\n".join(skipped[:500]) + "\n\n"
        "Model tensors not loaded:\n" + "\n".join(missing[:500]) + "\n",
        encoding="utf-8")

    discovered_ids = discover_diode_cloud_ids(datamodule.hparams.data_dir, stage)
    selected_ids = None
    metric_dataset_indices = None
    image_dataset_indices = None
    if discovered_ids:
        requested_indices = parse_indices(
            cfg.get("diag_scene_indices", None),
            len(discovered_ids),
            num_metric_scenes)
        selected_ids = [
            discovered_ids[i]
            for i in requested_indices
            if 0 <= i < len(discovered_ids)
        ]
        if selected_ids:
            log.info(
                "Restricting DIODE preprocessing/inference to "
                f"{len(selected_ids)}/{len(discovered_ids)} discovered ids")
            datamodule.kwargs["active_learning_ids"] = selected_ids
            datamodule.kwargs["active_learning_stages"] = [parse_diode_stage(stage)[0]]
            datamodule.kwargs["active_learning_strict"] = True
            metric_dataset_indices = list(range(len(selected_ids)))
            image_selected_indices = parse_image_indices(
                cfg.get("diag_image_scene_indices", None),
                requested_indices,
                num_images)
            image_selected_ids = {
                discovered_ids[i]
                for i in image_selected_indices
                if 0 <= i < len(discovered_ids)
            }

    dataset = datamodule.dataset_class(
        datamodule.hparams.data_dir,
        stage=stage,
        transform=datamodule.test_transform,
        pre_transform=datamodule.pre_transform,
        on_device_transform=None,
        **datamodule.kwargs,
    )
    total = len(dataset)
    if total == 0:
        raise RuntimeError(f"No scenes found for stage={stage}")

    indices = metric_dataset_indices if metric_dataset_indices is not None else parse_indices(
        cfg.get("diag_scene_indices", None), total, num_metric_scenes)
    if selected_ids:
        image_dataset_indices = {
            i for i, cloud_id in enumerate(selected_ids)
            if cloud_id in image_selected_ids
        }
    else:
        image_dataset_indices = set(parse_image_indices(
            cfg.get("diag_image_scene_indices", None),
            indices,
            num_images))
    rows = []
    skipped_rows = []
    log.info(
        f"Running diagnostic on stage='{stage}', "
        f"metric scenes={len(indices)}, image scenes={len(image_dataset_indices)}")
    diode_classes = int(getattr(datamodule.hparams, "num_classes", 20))
    raw_classes = max(ckpt_classes, diode_classes)
    overlap_classes = min(ckpt_classes, diode_classes)
    raw_conf_total = np.zeros((raw_classes, raw_classes), dtype=np.int64)
    overlap_conf_total = np.zeros((overlap_classes, overlap_classes), dtype=np.int64)
    mapped_conf_total = np.zeros(
        (len(MAPPED_CLASS_NAMES), len(MAPPED_CLASS_NAMES)), dtype=np.int64)

    for out_idx, dataset_idx in enumerate(indices):
        if dataset_idx < 0 or dataset_idx >= total:
            log.warning(f"Skipping out-of-range scene index {dataset_idx}; total={total}")
            continue

        cloud_id = dataset.cloud_ids[dataset_idx]
        safe_id = str(cloud_id).replace("/", "_").replace("\\", "_")
        log.info(f"[{out_idx + 1}/{len(indices)}] {cloud_id}")

        nag = dataset[dataset_idx]
        if not isinstance(nag, NAG):
            log.warning(f"Skipping {cloud_id}: expected NAG, got {type(nag)}")
            continue

        batch = NAGBatch.from_nag_list([nag]).to(device)
        if datamodule.on_device_test_transform is not None:
            batch = datamodule.on_device_test_transform(batch)

        try:
            with torch.no_grad():
                output = model.forward(batch)
                pred_level1 = output.semantic_pred()
        except Exception as e:
            if "Data should contain the attribute 'rgb'" in str(e):
                log.warning(
                    f"Skipping {cloud_id}: missing rgb, likely an old empty-tile placeholder.")
                skipped_rows.append({
                    "scene_index": int(dataset_idx),
                    "cloud_id": cloud_id,
                    "reason": "missing_rgb_empty_tile_placeholder",
                })
                continue
            raise

        png_path = ""
        if dataset_idx in image_dataset_indices:
            png_path = output_dir / f"{out_idx:03d}_{safe_id}.png"
            metrics, raw_conf, overlap_conf, mapped_conf = save_scene_plot(
                batch,
                pred_level1,
                cloud_id,
                png_path,
                max_points,
                ckpt_classes=ckpt_classes,
                diode_classes=diode_classes)
        else:
            metrics, raw_conf, overlap_conf, mapped_conf = compute_scene_metrics(
                batch,
                pred_level1,
                ckpt_classes=ckpt_classes,
                diode_classes=diode_classes)
        raw_conf_total += raw_conf
        overlap_conf_total += overlap_conf
        mapped_conf_total += mapped_conf
        row = {
            "scene_index": int(dataset_idx),
            "cloud_id": cloud_id,
            "num_level0_points": int(batch[0].num_nodes),
            "num_level1_superpoints": int(batch[1].num_nodes),
            "png": str(png_path),
        }
        row.update(metrics)
        rows.append(row)

    csv_path = output_dir / "prediction_metrics.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    skipped_path = output_dir / "skipped_tiles.csv"
    if skipped_rows:
        with skipped_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["scene_index", "cloud_id", "reason"])
            writer.writeheader()
            writer.writerows(skipped_rows)

    raw_summary = metrics_from_confusion(raw_conf_total)
    overlap_summary = metrics_from_confusion(overlap_conf_total)
    mapped_summary = metrics_from_confusion(mapped_conf_total)

    class_metrics_path = output_dir / "class_metrics_raw_index.csv"
    class_metrics_overlap_path = output_dir / "class_metrics_overlap_index.csv"
    class_metrics_mapped_path = output_dir / "class_metrics_mapped.csv"
    mapping_table_path = output_dir / "class_mapping_kitti360_to_diode.csv"
    write_per_class_metrics(
        class_metrics_path,
        per_class_metrics_from_confusion(
            raw_conf_total,
            KITTI360_CLASS_NAMES,
            DIODE_CLASS_NAMES))
    write_per_class_metrics(
        class_metrics_overlap_path,
        per_class_metrics_from_confusion(
            overlap_conf_total,
            KITTI360_CLASS_NAMES,
            DIODE_CLASS_NAMES))
    write_mapped_per_class_metrics(
        class_metrics_mapped_path,
        mapped_per_class_metrics_from_confusion(mapped_conf_total))
    write_mapping_table(mapping_table_path)

    summary_path = output_dir / "summary_metrics.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "num_scenes",
            "num_skipped",
            "raw_miou",
            "raw_oa",
            "overlap_miou",
            "overlap_oa",
            "mapped_miou",
            "mapped_oa",
            "raw_metric_points",
            "overlap_metric_points",
            "mapped_metric_points",
            "checkpoint_classes",
            "diode_classes",
            "mapped_classes",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "num_scenes": len(rows),
            "num_skipped": len(skipped_rows),
            "raw_miou": raw_summary["miou"],
            "raw_oa": raw_summary["oa"],
            "overlap_miou": overlap_summary["miou"],
            "overlap_oa": overlap_summary["oa"],
            "mapped_miou": mapped_summary["miou"],
            "mapped_oa": mapped_summary["oa"],
            "raw_metric_points": int(raw_conf_total.sum()),
            "overlap_metric_points": int(overlap_conf_total.sum()),
            "mapped_metric_points": int(mapped_conf_total.sum()),
            "checkpoint_classes": ckpt_classes,
            "diode_classes": diode_classes,
            "mapped_classes": len(MAPPED_CLASS_NAMES),
        })

    print(f"\nSaved diagnostic images to: {output_dir}")
    print(f"Saved load report to: {output_dir / 'load_report.txt'}")
    print(f"Saved per-scene metrics CSV to: {csv_path}")
    print(f"Saved aggregate metrics CSV to: {summary_path}")
    print(f"Saved class metrics CSV to: {class_metrics_path}")
    print(f"Saved overlap class metrics CSV to: {class_metrics_overlap_path}")
    print(f"Saved mapped class metrics CSV to: {class_metrics_mapped_path}")
    print(f"Saved mapping table CSV to: {mapping_table_path}")
    if skipped_rows:
        print(f"Saved skipped tile report to: {skipped_path}")
    print(
        "Aggregate mapped metrics: "
        f"mIoU={mapped_summary['miou']:.4f}, OA={mapped_summary['oa']:.4f}")
    print(
        "Aggregate raw-index metrics: "
        f"mIoU={raw_summary['miou']:.4f}, OA={raw_summary['oa']:.4f}")
    print(
        "Aggregate overlap-index metrics: "
        f"mIoU={overlap_summary['miou']:.4f}, OA={overlap_summary['oa']:.4f}")


if __name__ == "__main__":
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    main()
