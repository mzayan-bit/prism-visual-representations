"""Unit tests for corruption uncertainty curves and prediction flip dynamics."""

from __future__ import annotations

from prism.uncertainty.contracts import (
    CorruptionUncertaintyCurve,
    PredictiveDistribution,
)
from prism.uncertainty.corruptions import evaluate_corruption_uncertainty


def test_evaluate_corruption_uncertainty() -> None:
    """Verify evaluation of accuracy decay, confidence slope, and flips."""
    clean_dists = [
        PredictiveDistribution(
            sample_id="s1",
            logits=[4.0, 0.0, 0.0],
            probabilities=[0.9, 0.05, 0.05],
            predicted_class=0,
            true_class=0,
            max_probability=0.9,
            entropy=0.1,
            normalized_entropy=0.09,
            logit_margin=4.0,
            probability_margin=0.85,
            is_correct=True,
            is_finite=True,
        ),
        PredictiveDistribution(
            sample_id="s2",
            logits=[0.0, 3.5, 0.0],
            probabilities=[0.1, 0.85, 0.05],
            predicted_class=1,
            true_class=1,
            max_probability=0.85,
            entropy=0.15,
            normalized_entropy=0.13,
            logit_margin=3.5,
            probability_margin=0.75,
            is_correct=True,
            is_finite=True,
        ),
    ]
    clean_reps = [[1.0, 0.0], [0.0, 1.0]]

    # Severity 1 to 3 corrupted distributions showing degradation
    corrupted_dists = {
        1: [
            PredictiveDistribution(
                sample_id="s1",
                logits=[3.0, 0.5, 0.5],
                probabilities=[0.8, 0.1, 0.1],
                predicted_class=0,
                true_class=0,
                max_probability=0.8,
                entropy=0.3,
                normalized_entropy=0.27,
                logit_margin=2.5,
                probability_margin=0.7,
                is_correct=True,
                is_finite=True,
            ),
            PredictiveDistribution(
                sample_id="s2",
                logits=[0.5, 2.5, 0.5],
                probabilities=[0.2, 0.7, 0.1],
                predicted_class=1,
                true_class=1,
                max_probability=0.7,
                entropy=0.4,
                normalized_entropy=0.36,
                logit_margin=2.0,
                probability_margin=0.5,
                is_correct=True,
                is_finite=True,
            ),
        ],
        2: [
            PredictiveDistribution(
                sample_id="s1",
                logits=[2.0, 1.0, 1.0],
                probabilities=[0.6, 0.2, 0.2],
                predicted_class=0,
                true_class=0,
                max_probability=0.6,
                entropy=0.6,
                normalized_entropy=0.54,
                logit_margin=1.0,
                probability_margin=0.4,
                is_correct=True,
                is_finite=True,
            ),
            PredictiveDistribution(
                sample_id="s2",
                logits=[1.0, 1.5, 0.5],
                probabilities=[0.4, 0.5, 0.1],
                predicted_class=1,
                true_class=1,
                max_probability=0.5,
                entropy=0.7,
                normalized_entropy=0.63,
                logit_margin=0.5,
                probability_margin=0.1,
                is_correct=True,
                is_finite=True,
            ),
        ],
        3: [
            # Sample 2 flips prediction from 1 to 0!
            PredictiveDistribution(
                sample_id="s1",
                logits=[1.0, 0.8, 0.8],
                probabilities=[0.4, 0.3, 0.3],
                predicted_class=0,
                true_class=0,
                max_probability=0.4,
                entropy=0.9,
                normalized_entropy=0.81,
                logit_margin=0.2,
                probability_margin=0.1,
                is_correct=True,
                is_finite=True,
            ),
            PredictiveDistribution(
                sample_id="s2",
                logits=[2.0, 1.0, 0.5],
                probabilities=[0.6, 0.3, 0.1],
                predicted_class=0,
                true_class=1,
                max_probability=0.6,
                entropy=0.65,
                normalized_entropy=0.59,
                logit_margin=1.0,
                probability_margin=0.3,
                is_correct=False,
                is_finite=True,
            ),
        ],
    }

    corrupted_reps = {
        1: [[1.1, 0.1], [0.1, 1.1]],
        2: [[1.3, 0.3], [0.4, 0.8]],
        3: [[1.5, 0.6], [0.8, 0.4]],
    }

    curve, flips = evaluate_corruption_uncertainty(
        clean_distributions=clean_dists,
        clean_representations=clean_reps,
        corrupted_distributions_by_severity=corrupted_dists,
        corrupted_representations_by_severity=corrupted_reps,
        corruption_name="gaussian_noise",
    )

    assert isinstance(curve, CorruptionUncertaintyCurve)
    assert curve.corruption_type == "gaussian_noise"
    assert curve.severities == [0, 1, 2, 3]
    assert curve.confidence_slope < 0  # Confidence drops with severity
    assert curve.entropy_slope > 0  # Entropy rises with severity
    assert len(curve.accuracies) == 4

    # Prediction flip was detected on severity 3 for s2
    assert len(flips) >= 1
    flip_s2 = next(f for f in flips if f.sample_id == "s2" and f.severity == 3)
    assert flip_s2.clean_prediction == 1
    assert flip_s2.corrupted_prediction == 0
