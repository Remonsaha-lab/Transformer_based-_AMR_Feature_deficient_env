import numpy as np
import pytest
import torch

from scripts.run_baselines import (
    AMR10_CLASSES,
    IGNORE_LABEL,
    NUM_CLASSES,
    MajorityPredictor,
    PointMLP,
    collate_point_clouds,
    remap_labels_to_amr10,
)
from scripts.evaluate_stage2 import confusion_matrix, metrics_from_confusion


def test_collate_point_clouds_variable_sizes():
    # 2 scans with different point counts (e.g. 10 points vs 25 points)
    scan1_feats = torch.randn(10, 6)
    scan1_lbls = torch.randint(0, 10, (10,))
    scan2_feats = torch.randn(25, 6)
    scan2_lbls = torch.randint(0, 10, (25,))

    batch = [(scan1_feats, scan1_lbls), (scan2_feats, scan2_lbls)]
    collated_feats, collated_lbls = collate_point_clouds(batch)

    assert collated_feats.shape == (35, 6)
    assert collated_lbls.shape == (35,)


def test_remap_labels_to_amr10():
    # Source labels: 0 (wall), 1 (floor), 4 (road), 20 (ignore)
    source = np.array([0, 1, 4, 20], dtype=np.int64)
    mapped = remap_labels_to_amr10(source)

    # Expected: 2 (building_wall), 0 (ground), 1 (road_path), 255 (ignore)
    assert mapped[0] == 2
    assert mapped[1] == 0
    assert mapped[2] == 1
    assert mapped[3] == IGNORE_LABEL


def test_majority_predictor_logic():
    majority = MajorityPredictor()
    majority.majority_class = 2  # building_wall
    preds = majority.predict(5)

    assert len(preds) == 5
    assert np.all(preds == 2)


def test_point_mlp_forward_and_backward():
    model = PointMLP(in_dim=6, hidden_dim=32, num_classes=NUM_CLASSES)
    x = torch.randn(20, 6, requires_grad=True)
    y = torch.randint(0, NUM_CLASSES, (20,))

    logits = model(x)
    assert logits.shape == (20, NUM_CLASSES)

    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()

    assert x.grad is not None
    assert torch.all(torch.isfinite(x.grad))


def test_baseline_metrics_integration():
    predictions = np.array([0, 1, 2, 2, 0])
    targets = np.array([0, 1, 2, 255, 1])

    matrix = confusion_matrix(predictions, targets, num_classes=NUM_CLASSES)
    metrics = metrics_from_confusion(matrix)

    assert metrics["valid_count"] == 4
    assert 0.0 <= metrics["macro_miou"] <= 1.0
    assert 0.0 <= metrics["overall_accuracy"] <= 1.0

