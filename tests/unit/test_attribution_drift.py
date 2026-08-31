"""Unit tests for attribution drift under input corruptions."""

import pytest

from prism.explainability.attribution import (
    AttributionMethod,
    AttributionNormalizationPolicy,
    AttributionResult,
    AttributionSpecification,
    compute_attribution_statistics,
)
from prism.explainability.drift import compute_attribution_drift


def _make_dummy_result(
    method: AttributionMethod,
    map_2d: list[list[float]],
    target_class: int = 0,
    predicted_class: int = 0,
) -> AttributionResult:
    stats = compute_attribution_statistics(map_2d)
    h, w = len(map_2d), len(map_2d[0])
    return AttributionResult(
        sample_id="drift_sample",
        model_id="drift_model",
        architecture="cnn",
        method=method,
        specification=AttributionSpecification(
            method=method, normalization=AttributionNormalizationPolicy.NONE
        ),
        target_class=target_class,
        predicted_class=predicted_class,
        true_class=0,
        target_score=1.0,
        predicted_score=1.0,
        source_image_shape=[3, h, w],
        attribution_shape=[h, w],
        raw_attribution_map=map_2d,
        normalized_attribution_map=map_2d,
        statistics=stats,
        positive_mass=stats.total_absolute_mass,
        negative_mass=0.0,
        absolute_mass=stats.total_absolute_mass,
        method_metadata={},
        warnings=[],
    )


def test_compute_attribution_drift_stable() -> None:
    """Test attribution drift when clean and corrupted maps are identical."""
    map_clean = [[1.0, 0.0], [0.0, 1.0]]
    clean_res = _make_dummy_result(
        AttributionMethod.INPUT_GRADIENT, map_clean, predicted_class=0
    )
    corr_res = _make_dummy_result(
        AttributionMethod.INPUT_GRADIENT, map_clean, predicted_class=0
    )

    drift = compute_attribution_drift(
        clean_result=clean_res,
        corrupted_result=corr_res,
        corruption_type="gaussian_noise",
        corruption_severity=0.15,
        representation_drift_distance=0.05,
    )

    assert pytest.approx(drift.attribution_cosine_similarity) == 1.0
    assert pytest.approx(drift.top_10_percent_mask_overlap) == 1.0
    assert pytest.approx(drift.center_of_mass_displacement) == 0.0
    assert drift.prediction_preserved is True
    assert drift.representation_drift_distance == 0.05


def test_compute_attribution_drift_flipped_prediction() -> None:
    """Test attribution drift when corruption causes a prediction flip."""
    map_clean = [[1.0, 0.0], [0.0, 1.0]]
    map_corr = [[0.0, 1.0], [1.0, 0.0]]

    clean_res = _make_dummy_result(
        AttributionMethod.INPUT_GRADIENT, map_clean, predicted_class=0
    )
    corr_res = _make_dummy_result(
        AttributionMethod.INPUT_GRADIENT, map_corr, predicted_class=1
    )

    drift = compute_attribution_drift(
        clean_result=clean_res,
        corrupted_result=corr_res,
        corruption_type="gaussian_noise",
        corruption_severity=0.15,
    )

    assert pytest.approx(drift.attribution_cosine_similarity) == 0.0
    assert drift.prediction_preserved is False
    assert drift.clean_predicted_class == 0
    assert drift.corrupted_predicted_class == 1
