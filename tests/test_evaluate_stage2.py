import numpy as np
import pytest

from scripts.evaluate_stage2 import (
    confusion_matrix,
    metrics_from_confusion,
    validate_pair,
)


def test_ignore_labels_are_excluded_and_absent_classes_are_not_dynamic_zero():
    predictions = np.array([0, 1, 0, 2])
    targets = np.array([0, 1, 255, 1])

    predictions, targets = validate_pair(predictions, targets, num_classes=3)
    result = metrics_from_confusion(confusion_matrix(predictions, targets, 3))

    assert result["valid_count"] == 3
    assert result["ignored_count"] == 0
    assert result["present_class"] == [True, True, False]
    assert result["macro_miou"] == pytest.approx(0.75)
    assert result["fixed_closed_set_miou"] == pytest.approx(0.5)


def test_alignment_failure_is_explicit():
    with pytest.raises(ValueError, match="alignment failure"):
        validate_pair(np.array([0, 1]), np.array([0]), num_classes=2)


def test_invalid_prediction_is_rejected():
    with pytest.raises(ValueError, match="invalid class IDs"):
        validate_pair(np.array([0, 4]), np.array([0, 1]), num_classes=2)
