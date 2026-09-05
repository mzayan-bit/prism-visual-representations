"""Uncertainty and Out-of-Distribution failure case detection and taxonomy."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from prism.representations.geometry import compute_distance
from prism.uncertainty.contracts import (
    OODReferenceSet,
    OODSample,
    PredictionFlipUncertainty,
    PredictiveDistribution,
)
from prism.uncertainty.enums import OODCategory, UncertaintyFailureType


@dataclass(frozen=True)
class UncertaintyFailureRecord:
    """Detected failure event in uncertainty or OOD representation analysis."""

    failure_type: UncertaintyFailureType
    sample_id: str
    description: str
    confidence: float
    entropy: float
    predicted_class: int
    true_class: int | None
    centroid_distance: float | None = None
    knn_distance: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize failure record to JSON-compatible dictionary."""
        return {
            "failure_type": self.failure_type.value,
            "sample_id": self.sample_id,
            "description": self.description,
            "confidence": round(self.confidence, 4),
            "entropy": round(self.entropy, 4),
            "predicted_class": self.predicted_class,
            "true_class": self.true_class,
            "centroid_distance": (
                round(self.centroid_distance, 4)
                if self.centroid_distance is not None
                else None
            ),
            "knn_distance": (
                round(self.knn_distance, 4) if self.knn_distance is not None else None
            ),
            "metadata": self.metadata,
        }


def detect_uncertainty_failures(
    distributions: Sequence[PredictiveDistribution],
    representations: Sequence[Sequence[float]] | None = None,
    reference_set: OODReferenceSet | None = None,
    ood_samples: Sequence[OODSample] | None = None,
    ood_distributions: Sequence[PredictiveDistribution] | None = None,
    ood_representations: Sequence[Sequence[float]] | None = None,
    prediction_flips: Sequence[PredictionFlipUncertainty] | None = None,
    high_conf_threshold: float = 0.80,
    low_conf_threshold: float = 0.40,
) -> list[UncertaintyFailureRecord]:
    """Detect empirical uncertainty, calibration, and OOD failure cases.

    Parameters
    ----------
    distributions : Sequence[PredictiveDistribution]
        In-distribution evaluation sample distributions.
    representations : Sequence[Sequence[float]] | None
        In-distribution feature vectors.
    reference_set : OODReferenceSet | None
        In-distribution reference set with centroids.
    ood_samples : Sequence[OODSample] | None
        OOD sample definitions.
    ood_distributions : Sequence[PredictiveDistribution] | None
        Model predictive distributions on OOD samples.
    ood_representations : Sequence[Sequence[float]] | None
        Feature vectors on OOD samples.
    prediction_flips : Sequence[PredictionFlipUncertainty] | None
        Corruption prediction flip events.
    high_conf_threshold : float
        Threshold for high-confidence classification (default 0.80).
    low_conf_threshold : float
        Threshold for low-confidence classification (default 0.40).

    Returns
    -------
    list[UncertaintyFailureRecord]
        Identified failure records across all taxonomy categories.
    """
    failures: list[UncertaintyFailureRecord] = []

    # 1. In-distribution failures
    for idx, d in enumerate(distributions):
        rep = representations[idx] if representations is not None else None
        c_dist: float | None = None
        if rep is not None and reference_set is not None:
            c_dist = min(
                compute_distance(list(rep), c, reference_set.distance_metric)
                for c in reference_set.class_centroids.values()
            )

        is_corr = d.is_correct if d.is_correct is not None else False

        # HIGH_CONFIDENCE_ERROR
        if not is_corr and d.max_probability >= high_conf_threshold:
            failures.append(
                UncertaintyFailureRecord(
                    failure_type=UncertaintyFailureType.HIGH_CONFIDENCE_ERROR,
                    sample_id=d.sample_id,
                    description=(
                        f"Sample incorrectly predicted as class {d.predicted_class} "
                        f"(true: {d.true_class}) with high confidence "
                        f"{d.max_probability:.2f}."
                    ),
                    confidence=d.max_probability,
                    entropy=d.entropy,
                    predicted_class=d.predicted_class,
                    true_class=d.true_class,
                    centroid_distance=c_dist,
                )
            )

        # LOW_CONFIDENCE_CORRECT
        if is_corr and d.max_probability <= low_conf_threshold:
            failures.append(
                UncertaintyFailureRecord(
                    failure_type=UncertaintyFailureType.LOW_CONFIDENCE_CORRECT,
                    sample_id=d.sample_id,
                    description=(
                        f"Sample correctly predicted as class {d.predicted_class} "
                        f"with unusually low confidence {d.max_probability:.2f}."
                    ),
                    confidence=d.max_probability,
                    entropy=d.entropy,
                    predicted_class=d.predicted_class,
                    true_class=d.true_class,
                    centroid_distance=c_dist,
                )
            )

        # ID_REPRESENTATION_OUTLIER
        if (
            c_dist is not None
            and d.true_class is not None
            and reference_set is not None
            and str(d.true_class) in reference_set.intra_class_radii
        ):
            max_r = reference_set.intra_class_radii[str(d.true_class)] * 1.5
            if c_dist > max_r:
                failures.append(
                    UncertaintyFailureRecord(
                        failure_type=UncertaintyFailureType.ID_REPRESENTATION_OUTLIER,
                        sample_id=d.sample_id,
                        description=(
                            f"In-distribution sample for class {d.true_class} is "
                            f"located far from class centroid (dist={c_dist:.2f} "
                            f"> boundary {max_r:.2f})."
                        ),
                        confidence=d.max_probability,
                        entropy=d.entropy,
                        predicted_class=d.predicted_class,
                        true_class=d.true_class,
                        centroid_distance=c_dist,
                    )
                )

    # 2. OOD failures
    if ood_samples is not None and ood_distributions is not None:
        for idx, (s, d) in enumerate(zip(ood_samples, ood_distributions, strict=False)):
            rep = ood_representations[idx] if ood_representations is not None else None
            c_dist = None
            nearest_class: str | None = None
            if rep is not None and reference_set is not None:
                dists = [
                    (
                        lbl_str,
                        compute_distance(list(rep), c, reference_set.distance_metric),
                    )
                    for lbl_str, c in reference_set.class_centroids.items()
                ]
                dists.sort(key=lambda t: t[1])
                nearest_class, c_dist = dists[0]

            # HIGH_CONFIDENCE_OOD
            if (
                s.category in (OODCategory.OUT_OF_DISTRIBUTION, OODCategory.NEAR_OOD)
                and d.max_probability >= high_conf_threshold
            ):
                failures.append(
                    UncertaintyFailureRecord(
                        failure_type=UncertaintyFailureType.HIGH_CONFIDENCE_OOD,
                        sample_id=s.sample_id,
                        description=(
                            f"OOD sample ({s.category.value}) predicted as "
                            f"class {d.predicted_class} with overconfident "
                            f"probability {d.max_probability:.2f}."
                        ),
                        confidence=d.max_probability,
                        entropy=d.entropy,
                        predicted_class=d.predicted_class,
                        true_class=None,
                        centroid_distance=c_dist,
                        metadata={"ood_category": s.category.value},
                    )
                )

            # OOD_NEAR_KNOWN_STRUCTURE
            if (
                s.category in (OODCategory.OUT_OF_DISTRIBUTION, OODCategory.NEAR_OOD)
                and c_dist is not None
                and nearest_class is not None
                and reference_set is not None
                and nearest_class in reference_set.intra_class_radii
            ):
                boundary_r = reference_set.intra_class_radii[nearest_class]
                if c_dist <= boundary_r:
                    failures.append(
                        UncertaintyFailureRecord(
                            failure_type=UncertaintyFailureType.OOD_NEAR_KNOWN_STRUCTURE,
                            sample_id=s.sample_id,
                            description=(
                                f"OOD sample falls within representation boundary "
                                f"of class {nearest_class} (dist={c_dist:.2f} "
                                f"<= {boundary_r:.2f})."
                            ),
                            confidence=d.max_probability,
                            entropy=d.entropy,
                            predicted_class=d.predicted_class,
                            true_class=None,
                            centroid_distance=c_dist,
                            metadata={"nearest_class": nearest_class},
                        )
                    )

    # 3. Corruption failures: CORRUPTION_OVERCONFIDENCE
    if prediction_flips is not None:
        for flip in prediction_flips:
            if flip.corrupted_confidence >= high_conf_threshold:
                failures.append(
                    UncertaintyFailureRecord(
                        failure_type=UncertaintyFailureType.CORRUPTION_OVERCONFIDENCE,
                        sample_id=flip.sample_id,
                        description=(
                            f"Under corruption '{flip.corruption_type}' "
                            f"(sev={flip.severity}), prediction flipped to "
                            f"wrong class {flip.corrupted_prediction} "
                            f"with high confidence {flip.corrupted_confidence:.2f}."
                        ),
                        confidence=flip.corrupted_confidence,
                        entropy=flip.corrupted_entropy,
                        predicted_class=flip.corrupted_prediction,
                        true_class=None,
                        metadata={
                            "corruption_type": flip.corruption_type,
                            "severity": flip.severity,
                            "representation_drift": flip.representation_drift,
                        },
                    )
                )

    return failures
