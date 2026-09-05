"""Unit tests for calibration curves, reliability bins, ECE, MCE, Brier, and NLL."""

from __future__ import annotations

import pytest

from prism.uncertainty.calibration import (
    compute_brier_score,
    compute_calibration_report,
    compute_expected_calibration_error,
    compute_maximum_calibration_error,
    compute_negative_log_likelihood,
    compute_reliability_bins,
)
from prism.uncertainty.contracts import (
    CalibrationReport,
    CalibrationSample,
    PredictiveDistribution,
)
from prism.uncertainty.enums import BinningStrategy


def test_reliability_bins_equal_width() -> None:
    """Verify equal-width partitioning of confidence samples into disjoint bins."""
    samples = [
        CalibrationSample(
            sample_id="s1",
            confidence=0.15,
            is_correct=True,
            predicted_class=0,
            true_class=0,
        ),
        CalibrationSample(
            sample_id="s2",
            confidence=0.18,
            is_correct=False,
            predicted_class=0,
            true_class=1,
        ),
        CalibrationSample(
            sample_id="s3",
            confidence=0.85,
            is_correct=True,
            predicted_class=1,
            true_class=1,
        ),
        CalibrationSample(
            sample_id="s4",
            confidence=0.92,
            is_correct=True,
            predicted_class=1,
            true_class=1,
        ),
    ]

    bins = compute_reliability_bins(
        samples, bin_count=10, strategy=BinningStrategy.EQUAL_WIDTH
    )
    assert len(bins) == 10

    # Bin 1: [0.1, 0.2]
    bin_1 = bins[1]
    assert bin_1.sample_count == 2
    assert bin_1.mean_confidence == pytest.approx(0.165, abs=1e-4)
    assert bin_1.empirical_accuracy == pytest.approx(0.5, abs=1e-4)
    assert bin_1.calibration_gap == pytest.approx(0.335, abs=1e-4)

    # Bin 8: [0.8, 0.9]
    bin_8 = bins[8]
    assert bin_8.sample_count == 1
    assert bin_8.mean_confidence == pytest.approx(0.85, abs=1e-4)
    assert bin_8.empirical_accuracy == pytest.approx(1.0, abs=1e-4)
    assert bin_8.calibration_gap == pytest.approx(0.15, abs=1e-4)

    # Empty bins
    bin_0 = bins[0]
    assert bin_0.sample_count == 0
    assert bin_0.mean_confidence == 0.0
    assert bin_0.empirical_accuracy == 0.0
    assert bin_0.calibration_gap == 0.0


def test_reliability_bins_equal_frequency() -> None:
    """Verify equal-frequency (quantile) binning distributes samples evenly."""
    samples = [
        CalibrationSample(
            sample_id=f"s_{i}",
            confidence=i / 10.0 + 0.05,
            is_correct=(i % 2 == 0),
            predicted_class=0,
            true_class=0 if i % 2 == 0 else 1,
        )
        for i in range(10)
    ]

    bins = compute_reliability_bins(
        samples, bin_count=5, strategy=BinningStrategy.EQUAL_FREQUENCY
    )
    assert len(bins) == 5
    assert all(b.sample_count == 2 for b in bins)


def test_ece_and_mce_computation() -> None:
    """Verify ECE and MCE metric calculations."""
    samples = [
        # 10 samples with conf 0.8, all correct (acc = 1.0) -> gap = 0.2
        *[
            CalibrationSample(
                sample_id=f"c_{i}",
                confidence=0.8,
                is_correct=True,
                predicted_class=0,
                true_class=0,
            )
            for i in range(10)
        ],
        # 10 samples with conf 0.6, all incorrect (acc = 0.0) -> gap = 0.6
        *[
            CalibrationSample(
                sample_id=f"e_{i}",
                confidence=0.6,
                is_correct=False,
                predicted_class=0,
                true_class=1,
            )
            for i in range(10)
        ],
    ]

    bins = compute_reliability_bins(
        samples, bin_count=10, strategy=BinningStrategy.EQUAL_WIDTH
    )
    ece = compute_expected_calibration_error(bins, total_sample_count=20)
    mce = compute_maximum_calibration_error(bins)

    # Weighted ECE: (10/20)*0.2 + (10/20)*0.6 = 0.1 + 0.3 = 0.4
    assert ece == pytest.approx(0.4, rel=1e-4)
    assert mce == pytest.approx(0.6, rel=1e-4)


def test_brier_and_nll_scores() -> None:
    """Verify multiclass Brier score and Negative Log Likelihood."""
    dists = [
        # Perfect confident prediction on class 0
        PredictiveDistribution(
            sample_id="s1",
            logits=[2.0, -2.0],
            probabilities=[0.9, 0.1],
            predicted_class=0,
            true_class=0,
            max_probability=0.9,
            entropy=0.3,
            normalized_entropy=0.3,
            logit_margin=4.0,
            probability_margin=0.8,
            is_correct=True,
            is_finite=True,
        ),
        # Uncertain prediction on class 1
        PredictiveDistribution(
            sample_id="s2",
            logits=[0.0, 0.5],
            probabilities=[0.4, 0.6],
            predicted_class=1,
            true_class=1,
            max_probability=0.6,
            entropy=0.67,
            normalized_entropy=0.9,
            logit_margin=0.5,
            probability_margin=0.2,
            is_correct=True,
            is_finite=True,
        ),
    ]

    brier = compute_brier_score(dists)
    assert brier >= 0.0

    nll = compute_negative_log_likelihood(dists)
    assert nll >= 0.0


def test_full_calibration_report() -> None:
    """Verify end-to-end CalibrationReport generation."""
    dists = [
        PredictiveDistribution(
            sample_id=f"s_{i}",
            logits=[3.0, 1.0, 0.0] if i != 2 else [0.0, 3.0, 1.0],
            probabilities=[0.7, 0.2, 0.1] if i != 2 else [0.1, 0.7, 0.2],
            predicted_class=0 if i != 2 else 1,
            true_class=0,
            max_probability=0.7,
            entropy=0.8,
            normalized_entropy=0.72,
            logit_margin=2.0,
            probability_margin=0.5,
            is_correct=(i != 2),
            is_finite=True,
        )
        for i in range(5)
    ]

    report = compute_calibration_report(
        distributions=dists,
        bin_count=5,
        class_names={0: "Square", 1: "Circle", 2: "Triangle"},
    )

    assert isinstance(report, CalibrationReport)
    assert report.sample_count == 5
    assert report.accuracy == pytest.approx(0.8, rel=1e-4)
    assert report.ece >= 0.0
    assert report.brier_score >= 0.0
    assert report.nll >= 0.0
    assert len(report.reliability_bins) == 5
