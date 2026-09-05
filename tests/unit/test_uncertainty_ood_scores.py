"""Unit tests for out-of-distribution scoring functions and polarity normalization."""

from __future__ import annotations

import pytest

from prism.uncertainty.contracts import (
    OODScoreResult,
    PredictiveDistribution,
)
from prism.uncertainty.enums import OODCategory, OODScoreMethod
from prism.uncertainty.ood_scores import (
    compute_class_centroid_ood_score,
    compute_energy_ood_score,
    compute_entropy_ood_score,
    compute_knn_ood_score,
    compute_max_softmax_ood_score,
    score_ood_sample,
)


def test_max_softmax_ood_score() -> None:
    """Verify MSP OOD score is 1 - max(p) with higher score meaning more OOD."""
    dist_confident = PredictiveDistribution(
        sample_id="s1",
        logits=[5.0, 0.0],
        probabilities=[0.95, 0.05],
        predicted_class=0,
        true_class=0,
        max_probability=0.95,
        entropy=0.1,
        normalized_entropy=0.1,
        logit_margin=5.0,
        probability_margin=0.9,
        is_correct=True,
        is_finite=True,
    )
    assert compute_max_softmax_ood_score(dist_confident) == pytest.approx(
        0.05, abs=1e-4
    )

    dist_uncertain = PredictiveDistribution(
        sample_id="s2",
        logits=[0.0, 0.0],
        probabilities=[0.5, 0.5],
        predicted_class=0,
        true_class=0,
        max_probability=0.5,
        entropy=0.69,
        normalized_entropy=1.0,
        logit_margin=0.0,
        probability_margin=0.0,
        is_correct=True,
        is_finite=True,
    )
    assert compute_max_softmax_ood_score(dist_uncertain) == pytest.approx(0.5, abs=1e-4)


def test_entropy_ood_score() -> None:
    """Verify predictive entropy OOD score."""
    dist = PredictiveDistribution(
        sample_id="s3",
        logits=[0.0, 0.0, 0.0],
        probabilities=[1 / 3, 1 / 3, 1 / 3],
        predicted_class=0,
        true_class=0,
        max_probability=1 / 3,
        entropy=1.09,
        normalized_entropy=1.0,
        logit_margin=0.0,
        probability_margin=0.0,
        is_correct=True,
        is_finite=True,
    )
    assert compute_entropy_ood_score(dist) == 1.0


def test_class_centroid_ood_score() -> None:
    """Verify Euclidean distance to class centroids."""
    centroids = {
        "0": [0.0, 0.0],
        "1": [10.0, 10.0],
    }

    # Point at origin is at centroid 0 -> dist 0
    rep_origin = [0.0, 0.0]
    dist_0, c_id_0 = compute_class_centroid_ood_score(rep_origin, centroids)
    assert dist_0 == pytest.approx(0.0, abs=1e-4)
    assert c_id_0 == "0"

    # Point at [3, 4] is distance 5 from origin
    rep_3_4 = [3.0, 4.0]
    dist_3_4, c_id_3_4 = compute_class_centroid_ood_score(rep_3_4, centroids)
    assert dist_3_4 == pytest.approx(5.0, abs=1e-4)
    assert c_id_3_4 == "0"


def test_knn_ood_score() -> None:
    """Verify k-nearest-neighbor Euclidean distance in representation space."""
    reference_reps = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [10.0, 10.0],
    ]

    # Point near cluster [0,0]
    rep_near = [0.1, 0.1]
    score_k3 = compute_knn_ood_score(rep_near, reference_reps, k=3)
    assert score_k3 < 1.5

    # Point far out
    rep_far = [50.0, 50.0]
    score_far = compute_knn_ood_score(rep_far, reference_reps, k=3)
    assert score_far > 50.0


def test_energy_ood_score() -> None:
    """Verify Helmholtz free energy OOD score (-T * ln sum exp(z/T))."""
    id_logits = [10.0, 2.0, 1.0]
    id_energy = compute_energy_ood_score(id_logits, temperature=1.0)

    ood_logits = [-5.0, -6.0, -5.5]
    ood_energy = compute_energy_ood_score(ood_logits, temperature=1.0)

    assert ood_energy > id_energy


def test_score_ood_sample() -> None:
    """Verify score_ood_sample helper returns structured OODScoreResult."""
    dist = PredictiveDistribution(
        sample_id="s1",
        logits=[4.0, 1.0],
        probabilities=[0.95, 0.05],
        predicted_class=0,
        true_class=0,
        max_probability=0.95,
        entropy=0.1,
        normalized_entropy=0.1,
        logit_margin=3.0,
        probability_margin=0.9,
        is_correct=True,
        is_finite=True,
    )

    res = score_ood_sample(
        sample_id="s1",
        category=OODCategory.IN_DISTRIBUTION,
        distribution=dist,
        score_method=OODScoreMethod.MAX_SOFTMAX_PROBABILITY,
    )

    assert isinstance(res, OODScoreResult)
    assert res.normalized_ood_score == pytest.approx(0.05, abs=1e-4)
