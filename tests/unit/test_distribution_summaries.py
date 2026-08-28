"""Unit tests for FeatureDistributionSummary and stability comparisons."""

import math

import pytest

from prism.representations.summaries import (
    FeatureDistributionSummary,
    compare_distribution_summaries,
    compute_distribution_summary,
)


@pytest.mark.unit
def test_compute_distribution_summary_vector() -> None:
    """Verify statistical summary computation on 2D vector representation."""
    # 2 samples, 4 features
    tensor = [
        [1.0, 2.0, 0.0, 5.0],
        [3.0, 4.0, 0.0, 1.0],
    ]
    summary = compute_distribution_summary(tensor)

    # 8 elements: [1, 2, 0, 5, 3, 4, 0, 1], sum = 16, mean = 2.0
    assert summary.sample_count == 8
    assert summary.tensor_shape == (2, 4)
    assert summary.mean == pytest.approx(2.0)
    assert summary.min_value == pytest.approx(0.0)
    assert summary.max_value == pytest.approx(5.0)
    assert summary.zero_fraction == pytest.approx(2.0 / 8.0)  # two 0.0 values
    assert summary.is_finite is True
    assert summary.channel_means is None  # only for 4D tensors


@pytest.mark.unit
def test_compute_distribution_summary_spatial_4d() -> None:
    """Verify channel-wise statistics for 4D spatial feature tensor [N, C, H, W]."""
    # 1 sample, 2 channels, 2x2 image
    tensor = [
        [
            [[2.0, 2.0], [2.0, 2.0]],  # Channel 0: all 2.0 (mean=2, var=0)
            [[10.0, 20.0], [30.0, 40.0]],  # Channel 1: mean=25.0, var=125.0
        ]
    ]
    summary = compute_distribution_summary(tensor)

    assert summary.sample_count == 8
    assert summary.tensor_shape == (1, 2, 2, 2)
    assert summary.channel_means is not None
    assert summary.channel_variances is not None
    assert len(summary.channel_means) == 2
    assert summary.channel_means[0] == pytest.approx(2.0)
    assert summary.channel_means[1] == pytest.approx(25.0)
    assert summary.channel_variances[0] == pytest.approx(0.0)
    assert summary.channel_variances[1] == pytest.approx(125.0)


@pytest.mark.unit
def test_compute_distribution_summary_non_finite() -> None:
    """Verify detection of NaN and Inf in representation tensors."""
    tensor_nan = [[1.0, float("nan")], [3.0, 4.0]]
    summary_nan = compute_distribution_summary(tensor_nan)
    assert summary_nan.is_finite is False
    assert math.isnan(summary_nan.mean)

    tensor_inf = [[1.0, float("inf")], [3.0, 4.0]]
    summary_inf = compute_distribution_summary(tensor_inf)
    assert summary_inf.is_finite is False


@pytest.mark.unit
def test_compare_distribution_summaries() -> None:
    """Verify stability shift measurement across two representation summaries."""
    sum_a = FeatureDistributionSummary(
        mean=0.0,
        variance=1.0,
        std_dev=1.0,
        min_value=-2.0,
        max_value=2.0,
        zero_fraction=0.1,
        is_finite=True,
        sample_count=100,
        tensor_shape=(10, 10),
    )

    sum_b = FeatureDistributionSummary(
        mean=0.5,
        variance=1.44,
        std_dev=1.2,
        min_value=-1.5,
        max_value=3.0,
        zero_fraction=0.3,
        is_finite=True,
        sample_count=100,
        tensor_shape=(10, 10),
    )

    diff = compare_distribution_summaries(sum_a, sum_b)
    assert diff["mean_shift"] == pytest.approx(0.5)
    assert diff["std_shift"] == pytest.approx(0.2)
    assert diff["range_delta"] == pytest.approx(0.5)
    assert diff["zero_fraction_delta"] == pytest.approx(0.2)


@pytest.mark.unit
def test_feature_distribution_summary_json_roundtrip() -> None:
    """Verify JSON serialization and deserialization of FeatureDistributionSummary."""
    summary = FeatureDistributionSummary(
        mean=1.234,
        variance=0.567,
        std_dev=0.753,
        min_value=-3.0,
        max_value=4.5,
        zero_fraction=0.05,
        is_finite=True,
        sample_count=128,
        tensor_shape=(8, 16),
    )
    json_str = summary.to_json()
    reconstructed = FeatureDistributionSummary.from_json(json_str)
    assert reconstructed == summary
