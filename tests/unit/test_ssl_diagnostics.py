"""Unit tests for SSL collapse diagnostics and alignment descriptors."""

from prism.ssl.diagnostics import compute_collapse_diagnostics


def test_healthy_representation_diagnostics() -> None:
    """Test diagnostics for diverse, healthy representations."""
    # 10 samples with distinct, diverse values across 4 dimensions
    reps = [
        [float(i) * 0.5, float(i % 3) * 1.2, float(i % 5) * 0.8, -float(i) * 0.3]
        for i in range(10)
    ]

    summary = compute_collapse_diagnostics(reps)
    assert summary.total_dimensions == 4
    assert summary.near_zero_variance_dimensions == 0
    assert summary.near_zero_variance_fraction == 0.0
    assert summary.mean_feature_std > 0.1
    assert summary.is_collapsed is False
    assert len(summary.warnings) == 0


def test_collapsed_representation_detection() -> None:
    """Test that constant / zero-variance representations trigger collapse detection."""
    # 10 samples with identical constant features
    collapsed_reps = [[1.0, 2.0, 3.0, 4.0] for _ in range(10)]

    summary = compute_collapse_diagnostics(collapsed_reps)
    assert summary.total_dimensions == 4
    assert summary.near_zero_variance_dimensions == 4
    assert summary.near_zero_variance_fraction == 1.0
    assert summary.mean_feature_std == 0.0
    assert summary.is_collapsed is True
    assert len(summary.warnings) > 0
