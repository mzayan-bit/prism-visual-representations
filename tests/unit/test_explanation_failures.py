"""Unit tests for explanation failure taxonomy and diagnostic flagging."""

from prism.explainability.attribution import (
    AttributionMethod,
    AttributionNormalizationPolicy,
    AttributionResult,
    AttributionSpecification,
    compute_attribution_statistics,
)
from prism.explainability.comparison import compare_attributions
from prism.explainability.drift import compute_attribution_drift
from prism.explainability.failures import (
    ExplanationFailureCategory,
    flag_explanation_failures,
)


def _make_dummy_result(
    method: AttributionMethod,
    map_2d: list[list[float]],
    predicted_class: int = 0,
) -> AttributionResult:
    stats = compute_attribution_statistics(map_2d)
    h, w = len(map_2d), len(map_2d[0])
    return AttributionResult(
        sample_id="fail_sample",
        model_id="fail_model",
        architecture="cnn",
        method=method,
        specification=AttributionSpecification(
            method=method, normalization=AttributionNormalizationPolicy.NONE
        ),
        target_class=0,
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


def test_flag_low_attribution_signal() -> None:
    """Test flagging near-zero attribution map as low_attribution_signal."""
    zero_map = [[0.0 for _ in range(4)] for _ in range(4)]
    res = _make_dummy_result(AttributionMethod.INPUT_GRADIENT, zero_map)

    flags = flag_explanation_failures(attribution_result=res)
    assert any(
        f.category == ExplanationFailureCategory.LOW_ATTRIBUTION_SIGNAL for f in flags
    )


def test_flag_method_disagreement() -> None:
    """Test flagging when methods have severe spatial divergence."""
    res_a = _make_dummy_result(
        AttributionMethod.INPUT_GRADIENT, [[1.0, 0.0], [0.0, 0.0]]
    )
    res_b = _make_dummy_result(
        AttributionMethod.OCCLUSION_SENSITIVITY, [[0.0, 0.0], [0.0, 1.0]]
    )

    report = compare_attributions([res_a, res_b])
    flags = flag_explanation_failures(comparison_report=report)
    assert any(
        f.category == ExplanationFailureCategory.METHOD_DISAGREEMENT for f in flags
    )


def test_flag_prediction_flip_with_stable_attribution() -> None:
    """Test flagging when prediction flips while attribution remains stable."""
    map_same = [[1.0, 0.0], [0.0, 1.0]]
    res_clean = _make_dummy_result(
        AttributionMethod.INPUT_GRADIENT, map_same, predicted_class=0
    )
    res_corr = _make_dummy_result(
        AttributionMethod.INPUT_GRADIENT, map_same, predicted_class=1
    )

    drift = compute_attribution_drift(
        clean_result=res_clean,
        corrupted_result=res_corr,
        corruption_type="gaussian_noise",
        corruption_severity=0.15,
    )

    flags = flag_explanation_failures(drift_summary=drift)
    assert any(
        f.category == ExplanationFailureCategory.PREDICTION_FLIP_WITH_STABLE_ATTRIBUTION
        for f in flags
    )
