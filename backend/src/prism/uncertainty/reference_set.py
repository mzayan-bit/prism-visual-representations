"""In-distribution reference set construction and centroid analysis."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

from prism.core.errors import ValidationError
from prism.representations.geometry import DistanceMetric, compute_distance
from prism.uncertainty.contracts import OODReferenceSet


def _compute_vector_fingerprint(vectors: Sequence[Sequence[float]]) -> str:
    """Compute deterministic SHA-256 fingerprint for feature vectors."""
    hasher = hashlib.sha256()
    for vec in vectors:
        for val in vec:
            hasher.update(round(val * 100000.0).to_bytes(8, "big", signed=True))
    return hasher.hexdigest()[:16]


def compute_class_centroids(
    vectors: Sequence[Sequence[float]],
    labels: Sequence[int],
) -> dict[str, list[float]]:
    """Compute the empirical geometric mean centroid for each class.

    Parameters
    ----------
    vectors : Sequence[Sequence[float]]
        Feature vectors of shape (N, D).
    labels : Sequence[int]
        Class label for each feature vector.

    Returns
    -------
    dict[str, list[float]]
        Mapping of class label string -> D-dimensional centroid vector.
    """
    if len(vectors) != len(labels):
        raise ValidationError(
            f"Vectors count ({len(vectors)}) does not match labels ({len(labels)})."
        )
    if not vectors:
        return {}

    dim = len(vectors[0])
    sums: dict[int, list[float]] = {}
    counts: dict[int, int] = {}

    for vec, lbl in zip(vectors, labels, strict=False):
        if len(vec) != dim:
            raise ValidationError(
                f"Vector dimension mismatch: expected {dim}, got {len(vec)}."
            )
        if lbl not in sums:
            sums[lbl] = [0.0] * dim
            counts[lbl] = 0
        for d in range(dim):
            sums[lbl][d] += vec[d]
        counts[lbl] += 1

    centroids: dict[str, list[float]] = {}
    for lbl, sum_vec in sums.items():
        cnt = counts[lbl]
        centroids[str(lbl)] = [val / cnt for val in sum_vec]

    return centroids


def compute_intra_class_radii(
    vectors: Sequence[Sequence[float]],
    labels: Sequence[int],
    centroids: dict[str, list[float]],
    distance_metric: DistanceMetric = DistanceMetric.EUCLIDEAN,
    percentile: float = 0.90,
) -> dict[str, float]:
    """Compute empirical intra-class percentile boundary radius.

    Parameters
    ----------
    vectors : Sequence[Sequence[float]]
        Reference feature vectors.
    labels : Sequence[int]
        Class labels.
    centroids : dict[str, list[float]]
        Computed class centroids {class_id_str: centroid_vec}.
    distance_metric : DistanceMetric
        Distance metric for geometry.
    percentile : float
        Percentile for radius (e.g. 0.90 for 90th percentile).

    Returns
    -------
    dict[str, float]
        Mapping of class label string -> boundary radius.
    """
    class_dists: dict[str, list[float]] = {lbl_str: [] for lbl_str in centroids}

    for vec, lbl in zip(vectors, labels, strict=False):
        lbl_str = str(lbl)
        if lbl_str in centroids:
            c = centroids[lbl_str]
            d = compute_distance(list(vec), c, distance_metric)
            class_dists[lbl_str].append(d)

    radii: dict[str, float] = {}
    for lbl_str, dist_list in class_dists.items():
        if not dist_list:
            radii[lbl_str] = 0.0
            continue
        sorted_dists = sorted(dist_list)
        idx = math.ceil(percentile * len(sorted_dists)) - 1
        idx = max(0, min(idx, len(sorted_dists) - 1))
        radii[lbl_str] = sorted_dists[idx]

    return radii


def build_ood_reference_set(
    sample_ids: Sequence[str],
    vectors: Sequence[Sequence[float]],
    labels: Sequence[int],
    source_experiment: str = "exp_id_reference",
    representation_layer: str = "backbone.encoder",
    distance_metric: DistanceMetric = DistanceMetric.EUCLIDEAN,
    normalization_policy: str = "l2",
) -> OODReferenceSet:
    """Build calibrated in-distribution reference set for OOD scoring.

    Parameters
    ----------
    sample_ids : Sequence[str]
        IDs of reference samples.
    vectors : Sequence[Sequence[float]]
        Reference representation feature vectors.
    labels : Sequence[int]
        Ground-truth labels of reference samples.
    source_experiment : str
        Originating experiment identifier.
    representation_layer : str
        Name of layer representations were extracted from.
    distance_metric : DistanceMetric
        Distance metric to use for comparisons.
    normalization_policy : str
        Feature normalization applied (e.g. 'none', 'l2', 'standardized').

    Returns
    -------
    OODReferenceSet
        Contract capturing reference samples, centroids, and fingerprint.
    """
    if len(sample_ids) != len(vectors) or len(vectors) != len(labels):
        raise ValidationError(
            f"Counts mismatch: sample_ids ({len(sample_ids)}), "
            f"vectors ({len(vectors)}), labels ({len(labels)})."
        )
    if not vectors:
        raise ValidationError("Cannot build reference set from empty vectors.")

    centroids = compute_class_centroids(vectors, labels)
    radii = compute_intra_class_radii(
        vectors=vectors,
        labels=labels,
        centroids=centroids,
        distance_metric=distance_metric,
        percentile=0.90,
    )
    fingerprint = _compute_vector_fingerprint(vectors)

    return OODReferenceSet(
        source_experiment=source_experiment,
        representation_layer=representation_layer,
        sample_ids=list(sample_ids),
        labels=list(labels),
        class_centroids=centroids,
        intra_class_radii=radii,
        normalization_policy=normalization_policy,
        distance_metric=distance_metric.value,
        fingerprint=fingerprint,
    )
