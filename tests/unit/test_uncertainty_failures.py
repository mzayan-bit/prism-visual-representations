"""Unit tests for uncertainty failure case detection and taxonomy categorization."""

from __future__ import annotations

from prism.uncertainty.contracts import (
    OODSample,
    PredictionFlipUncertainty,
    PredictiveDistribution,
)
from prism.uncertainty.enums import OODCategory, UncertaintyFailureType
from prism.uncertainty.failures import detect_uncertainty_failures


def test_detect_high_confidence_error() -> None:
    """Verify detection of confident mistakes."""
    # Incorrect prediction with 0.95 confidence
    dists = [
        PredictiveDistribution(
            sample_id="err_1",
            logits=[5.0, 0.0],
            probabilities=[0.95, 0.05],
            predicted_class=0,
            true_class=1,  # mistake!
            max_probability=0.95,
            entropy=0.1,
            normalized_entropy=0.1,
            logit_margin=5.0,
            probability_margin=0.9,
            is_correct=False,
            is_finite=True,
        )
    ]

    failures = detect_uncertainty_failures(
        distributions=dists, high_conf_threshold=0.80
    )
    assert len(failures) == 1
    assert failures[0].failure_type == UncertaintyFailureType.HIGH_CONFIDENCE_ERROR
    assert failures[0].sample_id == "err_1"


def test_detect_low_confidence_correct() -> None:
    """Verify detection of hesitant correct predictions."""
    # Correct prediction with only 0.35 confidence
    dists = [
        PredictiveDistribution(
            sample_id="hesitant_1",
            logits=[0.0, 0.0, 0.0],
            probabilities=[0.35, 0.33, 0.32],
            predicted_class=0,
            true_class=0,  # correct!
            max_probability=0.35,
            entropy=1.09,
            normalized_entropy=0.99,
            logit_margin=0.0,
            probability_margin=0.02,
            is_correct=True,
            is_finite=True,
        )
    ]

    failures = detect_uncertainty_failures(distributions=dists, low_conf_threshold=0.40)
    assert len(failures) == 1
    assert failures[0].failure_type == UncertaintyFailureType.LOW_CONFIDENCE_CORRECT
    assert failures[0].sample_id == "hesitant_1"


def test_detect_high_confidence_ood() -> None:
    """Verify detection of overconfident predictions on OOD inputs."""
    ood_samples = [
        OODSample(
            sample_id="ood_1",
            source_dataset_identity="synth_ood",
            category=OODCategory.OUT_OF_DISTRIBUTION,
            image=[[[0.0]]],
        )
    ]
    ood_dists = [
        PredictiveDistribution(
            sample_id="ood_1",
            logits=[4.0, 0.0],
            probabilities=[0.92, 0.08],
            predicted_class=0,
            true_class=None,
            max_probability=0.92,
            entropy=0.1,
            normalized_entropy=0.1,
            logit_margin=4.0,
            probability_margin=0.84,
            is_correct=None,
            is_finite=True,
        )
    ]

    failures = detect_uncertainty_failures(
        distributions=[],
        ood_samples=ood_samples,
        ood_distributions=ood_dists,
        high_conf_threshold=0.80,
    )

    assert any(
        f.failure_type == UncertaintyFailureType.HIGH_CONFIDENCE_OOD for f in failures
    )


def test_detect_corruption_overconfidence() -> None:
    """Verify detection of high-confidence corruption prediction flips."""
    flips = [
        PredictionFlipUncertainty(
            sample_id="flip_1",
            corruption_type="gaussian_noise",
            severity=5,
            clean_prediction=0,
            corrupted_prediction=1,
            clean_confidence=0.9,
            corrupted_confidence=0.88,  # Overconfident corrupted flip!
            clean_entropy=0.1,
            corrupted_entropy=0.2,
            representation_drift=1.5,
        )
    ]

    failures = detect_uncertainty_failures(
        distributions=[],
        prediction_flips=flips,
        high_conf_threshold=0.80,
    )

    assert any(
        f.failure_type == UncertaintyFailureType.CORRUPTION_OVERCONFIDENCE
        for f in failures
    )
