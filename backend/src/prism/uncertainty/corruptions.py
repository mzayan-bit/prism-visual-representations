"""Corruption uncertainty tracking and prediction flip dynamics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from prism.core.errors import ValidationError
from prism.representations.geometry import DistanceMetric, compute_distance
from prism.uncertainty.calibration import compute_calibration_report
from prism.uncertainty.contracts import (
    CorruptionUncertaintyCurve,
    OODReferenceSet,
    PredictionFlipUncertainty,
    PredictiveDistribution,
)


def _compute_linear_slope(
    x_values: Sequence[float], y_values: Sequence[float]
) -> float:
    """Compute ordinary least-squares slope for a sequence of points."""
    n = len(x_values)
    if n < 2 or len(y_values) != n:
        return 0.0

    mean_x = sum(x_values) / n
    mean_y = sum(y_values) / n

    num = sum(
        (x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values, strict=False)
    )
    den = sum((x - mean_x) ** 2 for x in x_values)

    if abs(den) < 1e-12:
        return 0.0
    return num / den


def _check_monotonic_increasing(
    values: Sequence[float], tolerance: float = 0.01
) -> bool:
    """Check if sequence is monotonically non-decreasing within tolerance."""
    return all(values[i] >= values[i - 1] - tolerance for i in range(1, len(values)))


def _check_monotonic_decreasing(
    values: Sequence[float], tolerance: float = 0.01
) -> bool:
    """Check if sequence is monotonically non-increasing within tolerance."""
    return all(values[i] <= values[i - 1] + tolerance for i in range(1, len(values)))


def evaluate_corruption_uncertainty(
    clean_distributions: Sequence[PredictiveDistribution],
    clean_representations: Sequence[Sequence[float]] | None,
    corrupted_distributions_by_severity: Mapping[int, Sequence[PredictiveDistribution]],
    corrupted_representations_by_severity: (
        Mapping[int, Sequence[Sequence[float]]] | None
    ) = None,
    reference_set: OODReferenceSet | None = None,
    corruption_name: str = "gaussian_noise",
    bin_count: int = 10,
) -> tuple[CorruptionUncertaintyCurve, list[PredictionFlipUncertainty]]:
    """Evaluate uncertainty trajectory and flip dynamics across corruptions.

    Parameters
    ----------
    clean_distributions : Sequence[PredictiveDistribution]
        Predictive distributions on clean (severity 0) inputs.
    clean_representations : Sequence[Sequence[float]] | None
        Clean feature representations of shape (N, D).
    corrupted_distributions_by_severity : dict[int, Sequence[PredictiveDistribution]]
        Mapping of severity (1..5) -> predictive distributions.
    corrupted_representations_by_severity : (
        dict[int, Sequence[Sequence[float]]] | None
    )
        Mapping of severity (1..5) -> feature representations.
    reference_set : OODReferenceSet | None
        Optional reference set to compute centroid distance trajectory.
    corruption_name : str
        Identifier of corruption type being studied.
    bin_count : int
        Number of bins for calibration analysis at each severity.

    Returns
    -------
    tuple[CorruptionUncertaintyCurve, list[PredictionFlipUncertainty]]
        Summary curve and list of per-sample prediction flip events.
    """
    if not clean_distributions:
        raise ValidationError("Clean distributions cannot be empty.")

    n_samples = len(clean_distributions)
    sorted_severities = sorted(corrupted_distributions_by_severity.keys())
    all_severities = [0, *sorted_severities]

    accuracies: list[float] = []
    mean_confidences: list[float] = []
    mean_entropies: list[float] = []
    eces: list[float] = []
    mean_drifts: list[float] = []
    mean_centroid_dists: list[float] = []

    # Severity 0 (Clean)
    clean_report = compute_calibration_report(clean_distributions, bin_count=bin_count)
    accuracies.append(clean_report.accuracy)
    mean_confidences.append(clean_report.mean_confidence)
    mean_entropies.append(clean_report.mean_predictive_entropy)
    eces.append(clean_report.ece)
    mean_drifts.append(0.0)

    if reference_set is not None and clean_representations is not None:
        c_dists = []
        for rep in clean_representations:
            d_min = min(
                compute_distance(list(rep), c, reference_set.distance_metric)
                for c in reference_set.class_centroids.values()
            )
            c_dists.append(d_min)
        mean_centroid_dists.append(sum(c_dists) / len(c_dists) if c_dists else 0.0)
    else:
        mean_centroid_dists.append(0.0)

    flips: list[PredictionFlipUncertainty] = []

    # Severities 1..K
    for sev in sorted_severities:
        corr_dists = corrupted_distributions_by_severity[sev]
        if len(corr_dists) != n_samples:
            raise ValidationError(
                f"Severity {sev} sample count ({len(corr_dists)}) "
                f"does not match clean ({n_samples})."
            )

        report = compute_calibration_report(corr_dists, bin_count=bin_count)
        accuracies.append(report.accuracy)
        mean_confidences.append(report.mean_confidence)
        mean_entropies.append(report.mean_predictive_entropy)
        eces.append(report.ece)

        # Representation drift
        corr_reps = (
            corrupted_representations_by_severity.get(sev)
            if corrupted_representations_by_severity
            else None
        )
        if clean_representations is not None and corr_reps is not None:
            drifts = [
                compute_distance(list(h_clean), list(h_corr), DistanceMetric.EUCLIDEAN)
                for h_clean, h_corr in zip(
                    clean_representations, corr_reps, strict=False
                )
            ]
            mean_drifts.append(sum(drifts) / len(drifts) if drifts else 0.0)
        else:
            mean_drifts.append(0.0)

        # Centroid distance
        if reference_set is not None and corr_reps is not None:
            c_dists = [
                min(
                    compute_distance(list(rep), c, reference_set.distance_metric)
                    for c in reference_set.class_centroids.values()
                )
                for rep in corr_reps
            ]
            mean_centroid_dists.append(sum(c_dists) / len(c_dists) if c_dists else 0.0)
        else:
            mean_centroid_dists.append(0.0)

        # Flip detection compared to clean
        for idx in range(n_samples):
            d_clean = clean_distributions[idx]
            d_corr = corr_dists[idx]
            if d_corr.predicted_class != d_clean.predicted_class:
                drift_val = 0.0
                if clean_representations is not None and corr_reps is not None:
                    drift_val = compute_distance(
                        list(clean_representations[idx]),
                        list(corr_reps[idx]),
                        DistanceMetric.EUCLIDEAN,
                    )
                flips.append(
                    PredictionFlipUncertainty(
                        sample_id=d_clean.sample_id,
                        corruption_type=corruption_name,
                        severity=sev,
                        clean_prediction=d_clean.predicted_class,
                        corrupted_prediction=d_corr.predicted_class,
                        clean_confidence=d_clean.max_probability,
                        corrupted_confidence=d_corr.max_probability,
                        clean_entropy=d_clean.entropy,
                        corrupted_entropy=d_corr.entropy,
                        representation_drift=drift_val,
                    )
                )

    x_sevs = [float(s) for s in all_severities]
    conf_slope = _compute_linear_slope(x_sevs, mean_confidences)
    ent_slope = _compute_linear_slope(x_sevs, mean_entropies)
    is_ent_monotonic = _check_monotonic_increasing(mean_entropies)

    curve = CorruptionUncertaintyCurve(
        corruption_type=corruption_name,
        severities=all_severities,
        accuracies=accuracies,
        mean_confidences=mean_confidences,
        mean_entropies=mean_entropies,
        eces=eces,
        mean_representation_drifts=mean_drifts,
        mean_ood_scores=mean_centroid_dists,
        confidence_slope=conf_slope,
        entropy_slope=ent_slope,
        is_monotonic_entropy=is_ent_monotonic,
    )

    return curve, flips
