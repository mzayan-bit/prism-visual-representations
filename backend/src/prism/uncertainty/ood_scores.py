"""Out-of-Distribution (OOD) scoring methods and sample evaluation."""

from __future__ import annotations

import math
from collections.abc import Sequence

from prism.core.errors import ValidationError
from prism.representations.geometry import DistanceMetric, compute_distance
from prism.uncertainty.contracts import (
    OODReferenceSet,
    OODScoreResult,
    PredictiveDistribution,
)
from prism.uncertainty.enums import OODCategory, OODScoreMethod


def compute_max_softmax_ood_score(distribution: PredictiveDistribution) -> float:
    """Compute OOD novelty score based on Maximum Softmax Probability (MSP).

    Normalized polarity: higher score = more OOD-like.
    score = 1.0 - max_i p_i in [0.0, 1.0 - 1/K].
    """
    return max(0.0, min(1.0, 1.0 - distribution.max_probability))


def compute_entropy_ood_score(distribution: PredictiveDistribution) -> float:
    """Compute OOD novelty score based on normalized predictive entropy.

    Normalized polarity: higher score = more OOD-like.
    score = H(p) / ln(K) in [0.0, 1.0].
    """
    return distribution.normalized_entropy


def compute_class_centroid_ood_score(
    representation: list[float],
    class_centroids: dict[str, list[float]],
    metric: DistanceMetric | str = DistanceMetric.EUCLIDEAN,
) -> tuple[float, str]:
    """Compute representation distance to the nearest class centroid.

    Parameters
    ----------
    representation : list[float]
        Extracted sample feature vector [D].
    class_centroids : dict[str, list[float]]
        Mapping of class IDs to centroid vectors {class_id: [D]}.
    metric : DistanceMetric | str
        Distance metric to use.

    Returns
    -------
    tuple[float, str]
        (minimum_distance_to_nearest_centroid, nearest_class_id).
    """
    if not class_centroids:
        raise ValidationError(
            "class_centroids cannot be empty for centroid OOD scoring."
        )

    min_dist = float("inf")
    nearest_class = ""

    for c_id, centroid in class_centroids.items():
        dist = compute_distance(representation, centroid, metric=metric)
        if dist < min_dist:
            min_dist = dist
            nearest_class = c_id

    return min_dist, nearest_class


def compute_knn_ood_score(
    representation: list[float],
    reference_representations: Sequence[list[float]],
    k: int = 5,
    metric: DistanceMetric | str = DistanceMetric.EUCLIDEAN,
) -> float:
    """Compute mean distance to the k nearest reference representations.

    Parameters
    ----------
    representation : list[float]
        Extracted sample feature vector [D].
    reference_representations : Sequence[list[float]]
        In-distribution reference feature representations.
    k : int
        Number of nearest neighbors to aggregate (k >= 1).
    metric : DistanceMetric | str
        Distance metric.

    Returns
    -------
    float
        Mean distance to k nearest neighbors. Higher = more OOD-like.
    """
    if not reference_representations:
        raise ValidationError(
            "reference_representations cannot be empty for kNN OOD scoring."
        )
    if k < 1:
        raise ValidationError(f"k must be at least 1, got {k}.")

    effective_k = min(k, len(reference_representations))
    distances = [
        compute_distance(representation, ref_vec, metric=metric)
        for ref_vec in reference_representations
    ]
    sorted_distances = sorted(distances)
    top_k_distances = sorted_distances[:effective_k]

    return sum(top_k_distances) / float(effective_k)


def compute_energy_ood_score(logits: list[float], temperature: float = 1.0) -> float:
    """Compute Free Energy OOD novelty score: E(x) = -T * ln(sum exp(z_i / T)).

    For OOD scoring where higher score = more OOD-like:
    ID samples have high logits -> large sum_exp -> strongly negative energy.
    OOD samples have low/flat logits -> small sum_exp -> higher energy (closer to 0).
    score = -T * ln(sum exp(z_i / T)).
    """
    if not logits:
        raise ValidationError("Logits list cannot be empty for Energy score.")
    if temperature <= 0.0:
        raise ValidationError(f"Temperature must be positive, got {temperature}.")

    max_z = max(logits)
    sum_exp = sum(math.exp((z - max_z) / temperature) for z in logits)
    log_sum_exp = (max_z / temperature) + math.log(max(1e-15, sum_exp))

    # Free energy E(x) = -T * log_sum_exp
    energy = -temperature * log_sum_exp
    return energy


def score_ood_sample(
    sample_id: str,
    category: OODCategory,
    distribution: PredictiveDistribution,
    score_method: OODScoreMethod,
    representation: list[float] | None = None,
    reference_set: OODReferenceSet | None = None,
    reference_vectors: Sequence[list[float]] | None = None,
    k: int = 5,
) -> OODScoreResult:
    """Compute individual sample OOD score result with consistent polarity metadata.

    Guarantees:
    - `normalized_ood_score` always follows 'higher = more OOD-like'.
    - Returns structured `OODScoreResult` with full diagnostics.
    """
    nearest_c_id: str | None = None
    c_dist: float | None = None
    knn_dist: float | None = None

    if score_method == OODScoreMethod.MAX_SOFTMAX_PROBABILITY:
        raw = compute_max_softmax_ood_score(distribution)
        norm_score = raw

    elif score_method == OODScoreMethod.PREDICTIVE_ENTROPY:
        raw = compute_entropy_ood_score(distribution)
        norm_score = raw

    elif score_method == OODScoreMethod.NEAREST_CLASS_CENTROID_DISTANCE:
        if representation is None or reference_set is None:
            raise ValidationError(
                "Centroid OOD score requires both representation and reference_set."
            )
        dist, c_id = compute_class_centroid_ood_score(
            representation=representation,
            class_centroids=reference_set.class_centroids,
            metric=reference_set.distance_metric,
        )
        raw = dist
        norm_score = dist
        c_dist = dist
        nearest_c_id = c_id

    elif score_method == OODScoreMethod.KNN_REPRESENTATION_DISTANCE:
        if representation is None:
            raise ValidationError(
                "kNN distance OOD score requires representation vector."
            )
        ref_vecs = reference_vectors
        if ref_vecs is None and reference_set is not None:
            # If reference set is provided without explicit matrix, use centroids
            ref_vecs = list(reference_set.class_centroids.values())
        if not ref_vecs:
            raise ValidationError("kNN distance scoring requires reference vectors.")

        metric_name = (
            reference_set.distance_metric if reference_set else DistanceMetric.EUCLIDEAN
        )
        k_val = compute_knn_ood_score(
            representation=representation,
            reference_representations=ref_vecs,
            k=k,
            metric=metric_name,
        )
        raw = k_val
        norm_score = k_val
        knn_dist = k_val

    elif score_method == OODScoreMethod.ENERGY_SCORE:
        raw = compute_energy_ood_score(distribution.logits)
        norm_score = raw

    else:
        raise ValidationError(f"Unsupported OODScoreMethod: {score_method}")

    return OODScoreResult(
        sample_id=sample_id,
        category=category,
        score_method=score_method,
        raw_score=raw,
        normalized_ood_score=norm_score,
        score_direction="higher_is_more_ood",
        predicted_class=distribution.predicted_class,
        confidence=distribution.max_probability,
        entropy=distribution.entropy,
        nearest_centroid_class=nearest_c_id,
        centroid_distance=c_dist,
        knn_distance=knn_dist,
    )
