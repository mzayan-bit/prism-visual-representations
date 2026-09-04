"""Unit tests for temporal metrics, consistency, and motion sensitivity."""

from prism.temporal.metrics import (
    compute_motion_sensitivity,
    compute_temporal_consistency,
    compute_temporal_drift_curve,
    compute_video_classification_metrics,
)


def test_temporal_consistency_static_sequence() -> None:
    # Identical features across all timesteps
    static_feats = [[1.0, 2.0, 3.0] for _ in range(4)]
    summary = compute_temporal_consistency(static_feats)

    assert abs(summary.mean_adjacent_distance - 0.0) < 1e-6
    assert abs(summary.median_adjacent_distance - 0.0) < 1e-6
    assert abs(summary.max_temporal_jump - 0.0) < 1e-6
    assert abs(summary.mean_adjacent_cosine_similarity - 1.0) < 1e-6


def test_temporal_consistency_dynamic_sequence() -> None:
    feats = [
        [1.0, 0.0],
        [0.8, 0.6],  # cos = 0.8, dist = sqrt(0.04 + 0.36) = sqrt(0.4) ≈ 0.632
        [0.0, 1.0],  # cos = 0.6, dist = sqrt(0.64 + 0.16) = sqrt(0.8) ≈ 0.894
    ]
    summary = compute_temporal_consistency(feats)
    assert summary.mean_adjacent_distance > 0.0
    assert summary.max_temporal_jump > 0.8
    assert summary.timestep_of_max_jump == 2
    assert summary.mean_adjacent_cosine_similarity < 1.0


def test_temporal_drift_curve() -> None:
    feats = [
        [2.0, 0.0],
        [1.5, 0.5],
        [1.0, 1.0],
    ]
    curve = compute_temporal_drift_curve(feats)
    assert len(curve) == 3

    # Anchor frame (t=0)
    assert abs(curve[0]["euclidean_drift"] - 0.0) < 1e-6
    assert abs(curve[0]["cosine_distance"] - 0.0) < 1e-6
    assert abs(curve[0]["norm_ratio"] - 1.0) < 1e-6

    # Later frames drift increases
    assert curve[1]["euclidean_drift"] > 0.0
    assert curve[2]["euclidean_drift"] > curve[1]["euclidean_drift"]


def test_motion_sensitivity_calculation() -> None:
    feats = [
        [0.0, 0.0],
        [0.2, 0.0],
        [0.4, 0.0],
        [0.6, 0.0],
    ]
    positions = [
        (0.1, 0.5),
        (0.2, 0.5),
        (0.3, 0.5),
        (0.4, 0.5),
    ]
    res = compute_motion_sensitivity(feats, positions)
    assert len(res["paired_deltas"]) == 3
    assert abs(res["mean_motion_displacement"] - 0.1) < 1e-5
    assert abs(res["mean_representation_drift"] - 0.2) < 1e-5
    assert abs(res["sensitivity_ratio"] - 2.0) < 1e-2


def test_video_classification_metrics() -> None:
    preds = [0, 1, 2, 2, 3]
    targets = [0, 1, 2, 0, 3]
    metrics = compute_video_classification_metrics(preds, targets)
    assert abs(metrics["accuracy"] - 0.8) < 1e-6
    assert metrics["total_samples"] == 5.0
    assert metrics["correct_samples"] == 4.0
