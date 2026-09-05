"""Representation geometry vs predictive confidence and entropy relationships."""

from __future__ import annotations

import math
from collections.abc import Sequence

from prism.core.errors import ValidationError
from prism.representations.geometry import compute_distance
from prism.uncertainty.contracts import (
    OODReferenceSet,
    PredictiveDistribution,
    RepresentationConfidenceRelationship,
)


def _compute_pearson_correlation(
    x: Sequence[float], y: Sequence[float]
) -> float | None:
    """Compute Pearson linear correlation coefficient."""
    n = len(x)
    if n < 3 or len(y) != n:
        return None

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y, strict=False))
    var_x = sum((xi - mean_x) ** 2 for xi in x)
    var_y = sum((yi - mean_y) ** 2 for yi in y)

    den = math.sqrt(var_x * var_y)
    if den < 1e-12:
        return 0.0
    return max(-1.0, min(1.0, cov / den))


def _compute_stats(values: Sequence[float]) -> tuple[float, float, float]:
    """Compute (mean, median, std_dev) for a sequence of floats."""
    if not values:
        return 0.0, 0.0, 0.0
    n = len(values)
    mean_val = sum(values) / n
    sorted_v = sorted(values)
    median_val = (
        sorted_v[n // 2]
        if n % 2 == 1
        else (sorted_v[n // 2 - 1] + sorted_v[n // 2]) / 2.0
    )
    variance = sum((v - mean_val) ** 2 for v in values) / n
    std_val = math.sqrt(variance)
    return mean_val, median_val, std_val


def compute_representation_confidence_relationships(
    distributions: Sequence[PredictiveDistribution],
    representations: Sequence[Sequence[float]],
    reference_set: OODReferenceSet,
    k_neighbors: int = 5,
    reference_vectors: Sequence[list[float]] | None = None,
) -> RepresentationConfidenceRelationship:
    """Compute relationships between representation geometry and uncertainty.

    Parameters
    ----------
    distributions : Sequence[PredictiveDistribution]
        Model predictive distributions for evaluation samples.
    representations : Sequence[Sequence[float]]
        Feature vectors corresponding to the evaluation samples.
    reference_set : OODReferenceSet
        In-distribution reference set with centroids and vectors.
    k_neighbors : int
        Number of nearest neighbors for kNN distance.
    reference_vectors : Sequence[list[float]] | None
        Optional explicit reference representation vectors.

    Returns
    -------
    RepresentationConfidenceRelationship
        Summary metrics connecting centroid / kNN distances to uncertainty.
    """
    if len(distributions) != len(representations):
        raise ValidationError(
            f"Distributions count ({len(distributions)}) "
            f"does not match representations count ({len(representations)})."
        )
    if not distributions:
        raise ValidationError("Distributions cannot be empty.")

    confidences: list[float] = []
    centroid_dists: list[float] = []
    knn_dists: list[float] = []

    correct_centroid_dists: list[float] = []
    correct_knn_dists: list[float] = []

    error_centroid_dists: list[float] = []
    error_knn_dists: list[float] = []

    ref_vecs = reference_vectors or list(reference_set.class_centroids.values())
    k_eff = max(1, min(k_neighbors, len(ref_vecs)))

    for d, rep in zip(distributions, representations, strict=False):
        conf = d.max_probability
        confidences.append(conf)
        rep_list = list(rep)

        # Centroid distance
        min_c_dist = min(
            compute_distance(rep_list, c, reference_set.distance_metric)
            for c in reference_set.class_centroids.values()
        )
        centroid_dists.append(min_c_dist)

        # kNN distance
        neighbor_dists = [
            compute_distance(rep_list, ref_vec, reference_set.distance_metric)
            for ref_vec in ref_vecs
        ]
        neighbor_dists.sort()
        knn_dist = sum(neighbor_dists[:k_eff]) / k_eff
        knn_dists.append(knn_dist)

        is_corr = d.is_correct if d.is_correct is not None else False
        if is_corr:
            correct_centroid_dists.append(min_c_dist)
            correct_knn_dists.append(knn_dist)
        else:
            error_centroid_dists.append(min_c_dist)
            error_knn_dists.append(knn_dist)

    # Correlations
    r_cent_conf = _compute_pearson_correlation(centroid_dists, confidences)
    r_knn_conf = _compute_pearson_correlation(knn_dists, confidences)

    # Descriptive grouping stats
    mean_corr_c, _, _ = _compute_stats(correct_centroid_dists)
    mean_err_c, _, _ = _compute_stats(error_centroid_dists)
    mean_corr_k, _, _ = _compute_stats(correct_knn_dists)
    mean_err_k, _, _ = _compute_stats(error_knn_dists)

    return RepresentationConfidenceRelationship(
        centroid_distance_pearson_correlation=r_cent_conf,
        knn_distance_pearson_correlation=r_knn_conf,
        correct_mean_centroid_distance=mean_corr_c,
        incorrect_mean_centroid_distance=mean_err_c,
        correct_mean_knn_distance=mean_corr_k,
        incorrect_mean_knn_distance=mean_err_k,
    )
