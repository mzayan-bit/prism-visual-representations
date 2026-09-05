"""Post-hoc scalar temperature scaling calibration optimization and evaluation."""

from __future__ import annotations

import math
from collections.abc import Sequence

from prism.core.errors import ValidationError
from prism.uncertainty.calibration import (
    compute_calibration_report,
    compute_negative_log_likelihood,
)
from prism.uncertainty.contracts import (
    CalibrationReport,
    PredictiveDistribution,
    TemperatureScalingResult,
)
from prism.uncertainty.probabilities import (
    batch_predictive_distributions,
    compute_predictive_distribution,
    compute_stable_softmax,
)


def apply_temperature_scaling(logits: list[float], temperature: float) -> list[float]:
    """Compute calibrated softmax probabilities with scalar temperature T > 0.

    Parameters
    ----------
    logits : list[float]
        Raw model logits [K].
    temperature : float
        Scalar temperature parameter T > 0.

    Returns
    -------
    list[float]
        Calibrated softmax probability distribution [K].
    """
    if temperature <= 0.0 or math.isnan(temperature) or math.isinf(temperature):
        raise ValidationError(
            f"Temperature T must be strictly positive and finite: {temperature}"
        )

    scaled = [z / temperature for z in logits]
    return compute_stable_softmax(scaled)


def fit_temperature_scaling(
    val_logits: Sequence[list[float]],
    val_targets: Sequence[int],
    search_range: tuple[float, float] = (0.05, 10.0),
    coarse_steps: int = 100,
    fine_steps: int = 50,
) -> TemperatureScalingResult:
    """Optimize scalar temperature T* on validation data to minimize NLL.

    Parameters
    ----------
    val_logits : Sequence[list[float]]
        Logits extracted on validation partition.
    val_targets : Sequence[int]
        Ground-truth labels for validation partition.
    search_range : tuple[float, float]
        Bounds (min_T, max_T) with min_T > 0.
    coarse_steps : int
        Number of steps in coarse grid search.
    fine_steps : int
        Number of steps in fine refinement search.

    Returns
    -------
    TemperatureScalingResult
        Optimal temperature parameter and validation metrics before/after.
    """
    if not val_logits or not val_targets:
        raise ValidationError(
            "Validation logits and targets cannot be empty for temperature fitting."
        )
    if len(val_logits) != len(val_targets):
        raise ValidationError(
            "Mismatch between validation logits count and targets count."
        )

    min_t, max_t = search_range
    if min_t <= 0.0 or max_t <= min_t:
        raise ValidationError(f"Invalid temperature search range: ({min_t}, {max_t}).")

    warnings: list[str] = []
    if len(val_logits) < 20:
        warnings.append(
            f"Small validation set (N={len(val_logits)}). "
            "Temperature fitting may overfit."
        )

    # 1. Baseline uncalibrated distributions (T = 1.0)
    sample_ids = [f"val_{i}" for i in range(len(val_logits))]
    uncal_dists = batch_predictive_distributions(
        sample_ids=sample_ids,
        logits_matrix=val_logits,
        true_classes=val_targets,
        temperature=1.0,
    )
    val_nll_before = compute_negative_log_likelihood(uncal_dists)
    uncal_report = compute_calibration_report(uncal_dists, bin_count=10)
    ece_before = uncal_report.ece

    # Objective function: validation NLL for candidate T
    def eval_t(t: float) -> float:
        total_nll = 0.0
        for row, y in zip(val_logits, val_targets, strict=True):
            probs = apply_temperature_scaling(row, t)
            p_target = max(1e-15, probs[y])
            total_nll -= math.log(p_target)
        return total_nll / float(len(val_logits))

    # 2. Coarse grid search
    best_t = 1.0
    best_nll = val_nll_before
    iterations = 0

    coarse_step_size = (max_t - min_t) / float(coarse_steps)
    for i in range(coarse_steps + 1):
        cand_t = min_t + float(i) * coarse_step_size
        if cand_t <= 0.0:
            continue
        nll = eval_t(cand_t)
        iterations += 1
        if nll < best_nll:
            best_nll = nll
            best_t = cand_t

    # 3. Fine refinement search around best_t
    fine_radius = coarse_step_size
    fine_min = max(min_t, best_t - fine_radius)
    fine_max = min(max_t, best_t + fine_radius)
    fine_step_size = (fine_max - fine_min) / float(fine_steps)

    for i in range(fine_steps + 1):
        cand_t = fine_min + float(i) * fine_step_size
        if cand_t <= 0.0:
            continue
        nll = eval_t(cand_t)
        iterations += 1
        if nll < best_nll:
            best_nll = nll
            best_t = cand_t

    # 4. Compute calibrated validation distributions and ECE after
    cal_dists = batch_predictive_distributions(
        sample_ids=sample_ids,
        logits_matrix=val_logits,
        true_classes=val_targets,
        temperature=best_t,
    )
    val_nll_after = compute_negative_log_likelihood(cal_dists)
    cal_report = compute_calibration_report(cal_dists, bin_count=10)
    ece_after = cal_report.ece

    return TemperatureScalingResult(
        fitted_temperature=best_t,
        validation_nll_before=val_nll_before,
        validation_nll_after=val_nll_after,
        ece_before=ece_before,
        ece_after=ece_after,
        search_range=[min_t, max_t],
        fitting_method="deterministic_coarse_fine_1d_grid_search",
        iterations=iterations,
        warnings=warnings,
    )


def evaluate_calibrated_predictions(
    test_distributions: Sequence[PredictiveDistribution],
    fitted_temperature: float,
    bin_count: int = 10,
) -> CalibrationReport:
    """Apply fitted temperature to test distributions and compute calibrated report.

    Guarantees:
    - Classification accuracy is invariant under temperature scaling.
    - All logits are scaled by T > 0 without mutating source distributions.
    """
    if fitted_temperature <= 0.0:
        raise ValidationError(f"Invalid fitted temperature: {fitted_temperature}")

    calibrated_dists: list[PredictiveDistribution] = []
    for d in test_distributions:
        cal_d = compute_predictive_distribution(
            sample_id=d.sample_id,
            logits=d.logits,
            true_class=d.true_class,
            temperature=fitted_temperature,
        )
        calibrated_dists.append(cal_d)

    report = compute_calibration_report(calibrated_dists, bin_count=bin_count)
    return report
