import numpy as np
import pytest

from scripts.load_diode import voxel_majority_voting


def test_single_point_voxel_high_confidence():
    labels = np.array([2], dtype=np.int32)
    confs = np.array([0.85], dtype=np.float32)
    idx_map = [[0]]

    v_lbl, v_cnf, v_cons, v_cnt = voxel_majority_voting(
        labels, confs, idx_map, min_confidence=0.7, min_consensus=0.6, ignore_label=20
    )

    assert v_lbl[0] == 2
    assert v_cnf[0] == pytest.approx(0.85)
    assert v_cons[0] == pytest.approx(1.0)
    assert v_cnt[0] == 1


def test_single_point_voxel_low_confidence_becomes_ignore():
    labels = np.array([2], dtype=np.int32)
    confs = np.array([0.55], dtype=np.float32)
    idx_map = [[0]]

    v_lbl, v_cnf, v_cons, v_cnt = voxel_majority_voting(
        labels, confs, idx_map, min_confidence=0.7, min_consensus=0.6, ignore_label=20
    )

    assert v_lbl[0] == 20
    assert v_cons[0] == pytest.approx(0.0)


def test_multi_point_voxel_majority_success():
    # 3 points: two '0' (ground), one '1' (road)
    labels = np.array([0, 0, 1], dtype=np.int32)
    confs = np.array([0.90, 0.80, 0.85], dtype=np.float32)
    idx_map = [[0, 1, 2]]

    v_lbl, v_cnf, v_cons, v_cnt = voxel_majority_voting(
        labels, confs, idx_map, min_confidence=0.7, min_consensus=0.6, ignore_label=20
    )

    assert v_lbl[0] == 0
    assert v_cons[0] == pytest.approx(2.0 / 3.0)
    assert v_cnf[0] == pytest.approx(0.85)
    assert v_cnt[0] == 3


def test_multi_point_voxel_low_consensus_becomes_ignore():
    # 3 points: three different labels, consensus = 1/3 = 0.33 < 0.6
    labels = np.array([0, 1, 2], dtype=np.int32)
    confs = np.array([0.90, 0.85, 0.80], dtype=np.float32)
    idx_map = [[0, 1, 2]]

    v_lbl, v_cnf, v_cons, v_cnt = voxel_majority_voting(
        labels, confs, idx_map, min_confidence=0.7, min_consensus=0.6, ignore_label=20
    )

    assert v_lbl[0] == 20
    assert v_cons[0] == pytest.approx(1.0 / 3.0)


def test_multi_point_voxel_with_unconfident_point():
    # 2 points with same label, but one has low confidence (0.45 < 0.7)
    # 1 confident point out of 2 total points gives consensus = 1/2 = 0.5 < 0.6
    labels = np.array([0, 0], dtype=np.int32)
    confs = np.array([0.90, 0.45], dtype=np.float32)
    idx_map = [[0, 1]]

    v_lbl, v_cnf, v_cons, v_cnt = voxel_majority_voting(
        labels, confs, idx_map, min_confidence=0.7, min_consensus=0.6, ignore_label=20
    )

    assert v_lbl[0] == 20
    assert v_cons[0] == pytest.approx(0.5)


def test_multi_point_voxel_all_unconfident():
    labels = np.array([0, 1], dtype=np.int32)
    confs = np.array([0.30, 0.40], dtype=np.float32)
    idx_map = [[0, 1]]

    v_lbl, v_cnf, v_cons, v_cnt = voxel_majority_voting(
        labels, confs, idx_map, min_confidence=0.7, min_consensus=0.6, ignore_label=20
    )

    assert v_lbl[0] == 20
    assert v_cons[0] == pytest.approx(0.0)

