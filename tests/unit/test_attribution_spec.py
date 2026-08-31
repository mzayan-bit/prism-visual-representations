"""Unit tests for attribution specification, statistics, and normalization."""

import pytest

from prism.core.errors import ValidationError
from prism.explainability.attribution import (
    AttributionMethod,
    AttributionNormalizationPolicy,
    AttributionSpecification,
    ChannelReductionPolicy,
    compute_attribution_statistics,
    normalize_attribution_map,
    reduce_channels,
)


def test_attribution_specification_defaults_and_fingerprint() -> None:
    """Test default specification values, validation, and fingerprint determinism."""
    spec = AttributionSpecification(method=AttributionMethod.INPUT_GRADIENT)
    assert spec.method == AttributionMethod.INPUT_GRADIENT
    assert spec.channel_reduction == ChannelReductionPolicy.ABS_MAX
    assert spec.normalization == AttributionNormalizationPolicy.MIN_MAX_ABSOLUTE

    fp1 = spec.fingerprint()
    fp2 = spec.fingerprint()
    assert fp1 == fp2
    assert len(fp1) == 64  # SHA-256


def test_attribution_specification_invalid_parameters() -> None:
    """Test that invalid window size or stride raises ValidationError."""
    with pytest.raises(ValidationError, match="positive"):
        AttributionSpecification(
            method=AttributionMethod.OCCLUSION_SENSITIVITY,
            occlusion_window_size=(0, 2),
        )

    with pytest.raises(ValidationError, match="positive"):
        AttributionSpecification(
            method=AttributionMethod.OCCLUSION_SENSITIVITY,
            occlusion_stride=(2, -1),
        )


def test_channel_reduction_policies() -> None:
    """Test 3D channel reduction policies (abs_max, abs_mean, l2_channel_norm)."""
    # Tensor shape [2, 2, 2]:
    # Channel 0 = [[3, -4], [0, 1]], Channel 1 = [[4, 3], [0, -2]]
    tensor_3d = [
        [[3.0, -4.0], [0.0, 1.0]],
        [[4.0, 3.0], [0.0, -2.0]],
    ]

    # L2: sqrt(3^2 + 4^2) = 5.0, sqrt((-4)^2 + 3^2) = 5.0, sqrt(1 + 4) = sqrt(5)
    l2_map = reduce_channels(tensor_3d, ChannelReductionPolicy.L2_CHANNEL_NORM)
    assert pytest.approx(l2_map[0][0]) == 5.0
    assert pytest.approx(l2_map[0][1]) == 5.0
    assert pytest.approx(l2_map[1][0]) == 0.0
    assert pytest.approx(l2_map[1][1]) == 5.0**0.5

    # Abs Max: max(|3|, |4|) = 4.0, max(|-4|, |3|) = 4.0
    max_map = reduce_channels(tensor_3d, ChannelReductionPolicy.ABS_MAX)
    assert pytest.approx(max_map[0][0]) == 4.0
    assert pytest.approx(max_map[0][1]) == 4.0

    # Abs Mean: (|3| + |4|)/2 = 3.5, (|-4| + |3|)/2 = 3.5
    mean_map = reduce_channels(tensor_3d, ChannelReductionPolicy.ABS_MEAN)
    assert pytest.approx(mean_map[0][0]) == 3.5
    assert pytest.approx(mean_map[0][1]) == 3.5


def test_normalize_attribution_map() -> None:
    """Test min-max, abs-sum, and signed min-max normalization policies."""
    raw_map = [[-2.0, 2.0], [0.0, 6.0]]

    # Min-max absolute: max(|raw|) = 6.0 -> absolute map divided by 6.0
    min_max = normalize_attribution_map(
        raw_map, AttributionNormalizationPolicy.MIN_MAX_ABSOLUTE
    )
    assert pytest.approx(min_max[0][0]) == 2.0 / 6.0
    assert pytest.approx(min_max[0][1]) == 2.0 / 6.0
    assert pytest.approx(min_max[1][0]) == 0.0
    assert pytest.approx(min_max[1][1]) == 1.0

    # Abs sum normalize: sum(|-2| + 2 + 0 + 6) = 10.0 -> elements divided by 10.0
    abs_sum = normalize_attribution_map(
        raw_map, AttributionNormalizationPolicy.ABS_SUM_NORMALIZE
    )
    assert pytest.approx(abs_sum[0][0]) == 0.2
    assert pytest.approx(abs_sum[1][1]) == 0.6

    # Signed min-max: max(|min|, |max|) = 6.0 -> divided by 6.0
    signed = normalize_attribution_map(
        raw_map, AttributionNormalizationPolicy.SIGNED_MIN_MAX
    )
    assert pytest.approx(signed[0][0]) == -2.0 / 6.0
    assert pytest.approx(signed[1][1]) == 1.0


def test_compute_attribution_statistics() -> None:
    """Test entropy, concentration, top-10% mass, and center of mass calculations."""
    # 4x4 matrix with a single point of mass at (1, 2)
    map_2d = [[0.0 for _ in range(4)] for _ in range(4)]
    map_2d[1][2] = 10.0

    stats = compute_attribution_statistics(map_2d)
    assert stats.is_finite is True
    assert pytest.approx(stats.total_absolute_mass) == 10.0
    assert pytest.approx(stats.top_10_percent_mass_fraction) == 1.0
    assert pytest.approx(stats.center_of_mass_row) == 1.0
    assert pytest.approx(stats.center_of_mass_col) == 2.0
    assert pytest.approx(stats.spatial_entropy) == 0.0  # single point -> zero entropy
    assert stats.concentration_score > 0.90
