"""Unit tests for cross-method attribution agreement and comparison reports."""

import pytest

from prism.explainability.attribution import (
    AttributionMethod,
    AttributionNormalizationPolicy,
    AttributionResult,
    AttributionSpecification,
    compute_attribution_statistics,
)
from prism.explainability.comparison import (
    compare_attributions,
    compute_map_cosine_similarity,
    compute_top_percent_overlap,
    create_top_percent_mask,
)


def _make_dummy_result(
    method: AttributionMethod,
    map_2d: list[list[float]],
) -> AttributionResult:
    stats = compute_attribution_statistics(map_2d)
    h, w = len(map_2d), len(map_2d[0])
    return AttributionResult(
        sample_id="test_sample",
        model_id="test_model",
        architecture="cnn",
        method=method,
        specification=AttributionSpecification(
            method=method, normalization=AttributionNormalizationPolicy.NONE
        ),
        target_class=0,
        predicted_class=0,
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


def test_compute_map_cosine_similarity() -> None:
    """Test vector cosine similarity between 2D maps."""
    map_a = [[1.0, 0.0], [0.0, 1.0]]
    map_b = [[1.0, 0.0], [0.0, 1.0]]
    map_c = [[0.0, 1.0], [1.0, 0.0]]

    assert pytest.approx(compute_map_cosine_similarity(map_a, map_b)) == 1.0
    assert pytest.approx(compute_map_cosine_similarity(map_a, map_c)) == 0.0


def test_create_top_percent_mask_and_overlap() -> None:
    """Test top-percentile binary mask and Jaccard overlap."""
    map_a = [
        [10.0, 0.0],
        [0.0, 0.0],
    ]
    map_b = [
        [5.0, 0.0],
        [0.0, 0.0],
    ]
    map_c = [
        [0.0, 0.0],
        [0.0, 10.0],
    ]

    mask_a = create_top_percent_mask(map_a, top_percent=0.25)
    assert mask_a[0][0] == 1
    assert mask_a[1][1] == 0

    assert pytest.approx(compute_top_percent_overlap(map_a, map_b, 0.25)) == 1.0
    assert pytest.approx(compute_top_percent_overlap(map_a, map_c, 0.25)) == 0.0


def test_compare_attributions_report() -> None:
    """Test generation of AttributionComparisonReport across multiple methods."""
    res_ig = _make_dummy_result(
        AttributionMethod.INPUT_GRADIENT, [[1.0, 0.0], [0.0, 1.0]]
    )
    res_gxi = _make_dummy_result(
        AttributionMethod.GRADIENT_X_INPUT, [[1.0, 0.0], [0.0, 1.0]]
    )
    res_occ = _make_dummy_result(
        AttributionMethod.OCCLUSION_SENSITIVITY, [[0.0, 1.0], [1.0, 0.0]]
    )

    report = compare_attributions([res_ig, res_gxi, res_occ])
    assert report.sample_id == "test_sample"
    assert len(report.results) == 3
    assert len(report.pairwise_agreements) == 3  # (IG, GxI), (IG, Occ), (GxI, Occ)
    assert report.mean_cross_method_agreement >= 0.0
    sim_val = report.cosine_similarity_matrix["input_gradient"]["gradient_x_input"]
    assert sim_val is not None
    assert pytest.approx(sim_val) == 1.0
