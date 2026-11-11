import pytest
import torch
import torch.nn as nn
from src.loss.focal import WeightedFocalLoss
from src.utils.loss import loss_with_target_histogram


def test_focal_loss_forward_and_backward():
    """Verify WeightedFocalLoss computes correct loss and backward gradients."""
    num_classes = 10
    batch_size = 32
    torch.manual_seed(42)

    logits = torch.randn(batch_size, num_classes, requires_grad=True)
    targets = torch.randint(0, num_classes, (batch_size,))

    loss_fn = WeightedFocalLoss(gamma=2.0, reduction="mean")
    loss = loss_fn(logits, targets)

    assert loss.dim() == 0
    assert loss.item() > 0.0
    assert torch.isfinite(loss)

    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_focal_loss_reductions():
    """Verify 'mean', 'sum', and 'none' reduction modes."""
    num_classes = 5
    batch_size = 16
    torch.manual_seed(42)

    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))

    fn_mean = WeightedFocalLoss(gamma=1.5, reduction="mean")
    fn_sum = WeightedFocalLoss(gamma=1.5, reduction="sum")
    fn_none = WeightedFocalLoss(gamma=1.5, reduction="none")

    loss_mean = fn_mean(logits, targets)
    loss_sum = fn_sum(logits, targets)
    loss_none = fn_none(logits, targets)

    assert loss_none.shape == (batch_size,)
    assert torch.allclose(loss_none.mean(), loss_mean, atol=1e-5)
    assert torch.allclose(loss_none.sum(), loss_sum, atol=1e-5)
    assert "WeightedFocalLoss" in repr(fn_mean)


def test_focal_loss_ignore_index():
    """Verify that ignored labels receive 0 loss and produce 0 gradient."""
    num_classes = 4
    ignore_index = 4
    logits = torch.randn(6, num_classes, requires_grad=True)
    targets = torch.tensor([0, 1, ignore_index, ignore_index, 2, ignore_index])

    loss_fn = WeightedFocalLoss(gamma=2.0, ignore_index=ignore_index, reduction="none")
    loss = loss_fn(logits, targets)

    assert loss[2].item() == 0.0
    assert loss[3].item() == 0.0
    assert loss[5].item() == 0.0
    assert loss[0].item() > 0.0

    loss.sum().backward()
    # Ignored indices should not receive gradient from loss
    assert torch.all(logits.grad[2] == 0.0)
    assert torch.all(logits.grad[3] == 0.0)
    assert torch.all(logits.grad[5] == 0.0)


def test_focal_modulation_effect():
    """Verify that gamma > 0 suppresses loss on easy (high confidence) predictions."""
    num_classes = 3
    # Easy prediction: very confident on class 0
    easy_logits = torch.tensor([[10.0, -5.0, -5.0]])
    # Hard prediction: ambiguous uniform logits
    hard_logits = torch.tensor([[0.0, 0.0, 0.0]])
    target = torch.tensor([0])

    ce_fn = WeightedFocalLoss(gamma=0.0, reduction="mean")
    focal_fn = WeightedFocalLoss(gamma=2.0, reduction="mean")

    easy_ce = ce_fn(easy_logits, target)
    easy_focal = focal_fn(easy_logits, target)

    hard_ce = ce_fn(hard_logits, target)
    hard_focal = focal_fn(hard_logits, target)

    # Focal loss should suppress easy examples far more than hard examples
    focal_suppression_easy = (easy_focal / easy_ce).item()
    focal_suppression_hard = (hard_focal / hard_ce).item()

    assert focal_suppression_easy < focal_suppression_hard
    assert focal_suppression_easy < 0.01  # Extremely suppressed


def test_focal_loss_with_target_histogram():
    """Verify compatibility with Superpoint Transformer's loss_with_target_histogram."""
    num_superpoints = 8
    num_classes = 10
    logits = torch.randn(num_superpoints, num_classes, requires_grad=True)

    # Simulated target histogram (superpoints contain points from multiple classes)
    y_hist = torch.zeros(num_superpoints, num_classes)
    y_hist[0, 1] = 50.0
    y_hist[0, 2] = 10.0
    y_hist[1, 0] = 100.0
    y_hist[2, 3] = 25.0
    y_hist[3, 4] = 80.0

    criterion = WeightedFocalLoss(gamma=2.0, ignore_index=num_classes)
    loss = loss_with_target_histogram(criterion, logits, y_hist)

    assert torch.isfinite(loss)
    assert loss.item() > 0.0

    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_class_balancing_modes():
    """Verify Median-Frequency and Effective-Number (Class-Balanced) formulas."""
    # Simulated highly imbalanced dataset (Wall dominant, Tree rare)
    counts = torch.tensor([5000.0, 2000.0, 50000.0, 1000.0, 50.0])  # Class 4 is rare (50 pts)
    num_classes = len(counts)

    # 1. Median frequency
    valid_mask = counts > 0
    freq = counts / counts.sum()
    median_freq = freq[valid_mask].median()
    median_weights = median_freq / freq
    median_weights /= median_weights.sum()

    assert median_weights[4] > median_weights[2]  # Rare class gets much higher weight than majority class
    assert torch.allclose(median_weights.sum(), torch.tensor(1.0))

    # 2. Effective number of samples (Class-Balanced)
    beta = 0.999
    effective_num = 1.0 - torch.pow(beta, counts)
    cb_weights = (1.0 - beta) / effective_num
    cb_weights /= cb_weights.sum()

    assert cb_weights[4] > cb_weights[2]  # Rare class gets higher weight
    assert torch.allclose(cb_weights.sum(), torch.tensor(1.0))


def test_multiloss_state_dict_unweighted():
    """Verify MultiLoss state_dict does not raise KeyError when criterion is unweighted."""
    from src.loss.multi import MultiLoss
    ce1 = nn.CrossEntropyLoss()
    ce2 = nn.CrossEntropyLoss()
    multi = MultiLoss([ce1, ce2], [1.0, 25.0])

    # Unweighted state_dict
    sd = multi.state_dict()
    assert isinstance(sd, dict)
    multi.load_state_dict(sd)

    # Weighted state_dict
    weights = torch.ones(10)
    multi.weight = weights
    sd_weighted = multi.state_dict()
    assert "weight" in sd_weighted
    multi.load_state_dict(sd_weighted)

