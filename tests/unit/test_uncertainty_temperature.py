"""Unit tests for temperature scaling calibration and accuracy invariance."""

from __future__ import annotations

import pytest

from prism.core.errors import ValidationError
from prism.uncertainty.calibration import compute_calibration_report
from prism.uncertainty.contracts import TemperatureScalingResult
from prism.uncertainty.probabilities import batch_predictive_distributions
from prism.uncertainty.temperature import (
    apply_temperature_scaling,
    evaluate_calibrated_predictions,
    fit_temperature_scaling,
)


def test_apply_temperature_scaling() -> None:
    """Verify temperature scaling computes probabilities and rejects non-positive T."""
    logits = [2.0, -4.0, 6.0]
    probs = apply_temperature_scaling(logits, temperature=2.0)
    assert abs(sum(probs) - 1.0) < 1e-6
    assert probs[2] > probs[0] > probs[1]

    # Non-positive temperature validation
    with pytest.raises(ValidationError, match="strictly positive"):
        apply_temperature_scaling(logits, temperature=0.0)
    with pytest.raises(ValidationError, match="strictly positive"):
        apply_temperature_scaling(logits, temperature=-0.5)


def test_fit_temperature_scaling_overconfident() -> None:
    """Verify 1D grid search finds T > 1.0 on overconfident logits, reducing NLL."""
    # Overconfident model: logits have huge scale, but some are incorrect
    val_logits = [
        [10.0, 0.0, 0.0],  # true 0 (correct)
        [8.0, 1.0, 0.0],  # true 1 (incorrect overconfident)
        [0.0, 9.0, 0.0],  # true 1 (correct)
        [0.0, 0.0, 7.0],  # true 2 (correct)
        [6.0, 0.0, 1.0],  # true 2 (incorrect overconfident)
    ]
    val_labels = [0, 1, 1, 2, 2]

    res = fit_temperature_scaling(
        val_logits,
        val_labels,
        search_range=(0.05, 10.0),
        coarse_steps=50,
        fine_steps=20,
    )

    assert isinstance(res, TemperatureScalingResult)
    assert res.fitted_temperature > 1.0  # Needs softening
    assert res.validation_nll_after < res.validation_nll_before


def test_fit_temperature_scaling_underconfident() -> None:
    """Verify 1D grid search finds T < 1.0 on underconfident validation logits."""
    # Underconfident model: logits have tiny scale, but all are correct
    val_logits = [
        [0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0],
        [0.0, 0.0, 0.1],
        [0.1, 0.0, 0.0],
    ]
    val_labels = [0, 1, 2, 0]

    res = fit_temperature_scaling(
        val_logits,
        val_labels,
        search_range=(0.05, 10.0),
        coarse_steps=50,
        fine_steps=20,
    )

    assert res.fitted_temperature < 1.0  # Needs sharpening
    assert res.validation_nll_after < res.validation_nll_before


def test_temperature_scaling_accuracy_invariance() -> None:
    """Verify invariant: argmax_k(z_k / T) == argmax_k(z_k) for all T > 0."""
    test_logits = [
        [2.5, 1.2, -0.4],
        [-1.0, 4.2, 0.5],
        [0.3, 0.2, 3.1],
        [5.0, 4.9, 1.0],
    ]
    test_labels = [0, 1, 2, 0]
    sample_ids = ["t0", "t1", "t2", "t3"]

    uncal_dists = batch_predictive_distributions(
        sample_ids=sample_ids,
        logits_matrix=test_logits,
        true_classes=test_labels,
        temperature=1.0,
    )
    uncal_report = compute_calibration_report(uncal_dists)

    cal_report = evaluate_calibrated_predictions(uncal_dists, fitted_temperature=2.45)

    # Classification accuracy MUST be identical
    assert uncal_report.accuracy == cal_report.accuracy

    # Softened confidence under T > 1.0
    assert cal_report.mean_confidence < uncal_report.mean_confidence
