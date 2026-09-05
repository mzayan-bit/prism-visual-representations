"""Calibration metrics, reliability diagram binning, ECE, MCE, Brier score, and NLL."""

from __future__ import annotations

import math
from collections.abc import Sequence

from prism.core.errors import ValidationError
from prism.uncertainty.contracts import (
    CalibrationReport,
    CalibrationSample,
    ClassCalibrationSummary,
    ConfidenceSubsetSummary,
    PredictiveDistribution,
    ReliabilityBin,
)
from prism.uncertainty.enums import BinningStrategy


def compute_reliability_bins(
    samples: Sequence[CalibrationSample],
    bin_count: int = 10,
    strategy: BinningStrategy = BinningStrategy.EQUAL_WIDTH,
) -> list[ReliabilityBin]:
    """Partition calibration samples into confidence bins and compute empirical metrics.

    Parameters
    ----------
    samples : Sequence[CalibrationSample]
        Evaluated calibration samples.
    bin_count : int
        Number of confidence bins B >= 1.
    strategy : BinningStrategy
        Binning strategy (EQUAL_WIDTH or EQUAL_FREQUENCY).

    Returns
    -------
    list[ReliabilityBin]
        List of B reliability bins.
    """
    if bin_count < 1:
        raise ValidationError(f"bin_count must be at least 1, got {bin_count}.")

    if not samples:
        # Return empty bins covering [0, 1]
        bins: list[ReliabilityBin] = []
        for b in range(bin_count):
            l_b = float(b) / float(bin_count)
            u_b = float(b + 1) / float(bin_count)
            bins.append(
                ReliabilityBin(
                    bin_index=b,
                    lower_bound=l_b,
                    upper_bound=u_b,
                    sample_count=0,
                    mean_confidence=0.0,
                    empirical_accuracy=0.0,
                    calibration_gap=0.0,
                )
            )
        return bins

    bins = []

    if strategy == BinningStrategy.EQUAL_WIDTH:
        for b in range(bin_count):
            l_b = float(b) / float(bin_count)
            u_b = float(b + 1) / float(bin_count)

            # Sample falls into bin if l_b <= conf < u_b (or <= u_b for last bin)
            is_last = b == bin_count - 1
            bin_samples = [
                s
                for s in samples
                if (
                    l_b <= s.confidence <= u_b if is_last else l_b <= s.confidence < u_b
                )
            ]

            count = len(bin_samples)
            if count > 0:
                mean_conf = sum(s.confidence for s in bin_samples) / float(count)
                emp_acc = sum(1.0 for s in bin_samples if s.is_correct) / float(count)
                gap = abs(emp_acc - mean_conf)
            else:
                mean_conf = 0.0
                emp_acc = 0.0
                gap = 0.0

            bins.append(
                ReliabilityBin(
                    bin_index=b,
                    lower_bound=l_b,
                    upper_bound=u_b,
                    sample_count=count,
                    mean_confidence=mean_conf,
                    empirical_accuracy=emp_acc,
                    calibration_gap=gap,
                )
            )

    elif strategy == BinningStrategy.EQUAL_FREQUENCY:
        sorted_samples = sorted(samples, key=lambda s: s.confidence)
        n = len(sorted_samples)

        for b in range(bin_count):
            start_idx = math.floor(float(b * n) / float(bin_count))
            end_idx = math.floor(float((b + 1) * n) / float(bin_count))
            bin_samples = sorted_samples[start_idx:end_idx]

            count = len(bin_samples)
            if count > 0:
                l_b = bin_samples[0].confidence
                u_b = bin_samples[-1].confidence
                mean_conf = sum(s.confidence for s in bin_samples) / float(count)
                emp_acc = sum(1.0 for s in bin_samples if s.is_correct) / float(count)
                gap = abs(emp_acc - mean_conf)
            else:
                l_b = float(b) / float(bin_count)
                u_b = float(b + 1) / float(bin_count)
                mean_conf = 0.0
                emp_acc = 0.0
                gap = 0.0

            bins.append(
                ReliabilityBin(
                    bin_index=b,
                    lower_bound=l_b,
                    upper_bound=u_b,
                    sample_count=count,
                    mean_confidence=mean_conf,
                    empirical_accuracy=emp_acc,
                    calibration_gap=gap,
                )
            )

    return bins


def compute_expected_calibration_error(
    bins: Sequence[ReliabilityBin], total_sample_count: int
) -> float:
    """Compute Expected Calibration Error (ECE) = sum_b (n_b / N) * |acc_b - conf_b|.

    Parameters
    ----------
    bins : Sequence[ReliabilityBin]
        Reliability diagram bins.
    total_sample_count : int
        Total sample count N.

    Returns
    -------
    float
        ECE score in [0.0, 1.0].
    """
    if total_sample_count <= 0:
        return 0.0

    ece = 0.0
    for b in bins:
        if b.sample_count > 0:
            weight = float(b.sample_count) / float(total_sample_count)
            ece += weight * b.calibration_gap

    return max(0.0, min(1.0, ece))


def compute_maximum_calibration_error(bins: Sequence[ReliabilityBin]) -> float:
    """Compute Maximum Calibration Error (MCE) = max_{b: n_b > 0} |acc_b - conf_b|.

    Parameters
    ----------
    bins : Sequence[ReliabilityBin]
        Reliability diagram bins.

    Returns
    -------
    float
        MCE score in [0.0, 1.0].
    """
    non_empty_gaps = [b.calibration_gap for b in bins if b.sample_count > 0]
    if not non_empty_gaps:
        return 0.0
    return max(non_empty_gaps)


def compute_brier_score(
    predictions: Sequence[PredictiveDistribution],
) -> float:
    """Compute multiclass Brier score: (1/N) sum_n sum_k (p_nk - y_nk)^2.

    Parameters
    ----------
    predictions : Sequence[PredictiveDistribution]
        Sample predictive distributions with ground-truth labels.

    Returns
    -------
    float
        Mean multiclass Brier score (non-negative).
    """
    if not predictions:
        return 0.0

    valid_count = 0
    total_brier = 0.0

    for pred in predictions:
        if pred.true_class is None:
            continue
        valid_count += 1
        k = len(pred.probabilities)
        target = pred.true_class

        sample_sq_error = 0.0
        for c in range(k):
            indicator = 1.0 if c == target else 0.0
            diff = pred.probabilities[c] - indicator
            sample_sq_error += diff * diff

        total_brier += sample_sq_error

    if valid_count == 0:
        return 0.0

    return total_brier / float(valid_count)


def compute_negative_log_likelihood(
    predictions: Sequence[PredictiveDistribution], eps: float = 1e-15
) -> float:
    """Compute mean negative log-likelihood (NLL): -(1/N) sum_n ln(p_n,y_n + eps).

    Parameters
    ----------
    predictions : Sequence[PredictiveDistribution]
        Sample predictive distributions with ground-truth labels.
    eps : float
        Numerical stabilizer for logarithm.

    Returns
    -------
    float
        Mean NLL score (non-negative).
    """
    if not predictions:
        return 0.0

    valid_count = 0
    total_nll = 0.0

    for pred in predictions:
        if pred.true_class is None:
            continue
        valid_count += 1
        target = pred.true_class
        if 0 <= target < len(pred.probabilities):
            p_target = max(eps, pred.probabilities[target])
            total_nll -= math.log(p_target)
        else:
            total_nll -= math.log(eps)

    if valid_count == 0:
        return 0.0

    return total_nll / float(valid_count)


def compute_confidence_subset_summary(
    distributions: Sequence[PredictiveDistribution],
) -> ConfidenceSubsetSummary:
    """Compute descriptive statistics for a subset of predictions."""
    if not distributions:
        return ConfidenceSubsetSummary(
            sample_count=0,
            mean_max_probability=0.0,
            median_max_probability=0.0,
            mean_entropy=0.0,
            mean_normalized_entropy=0.0,
        )

    n = len(distributions)
    confs = [d.max_probability for d in distributions]
    entropies = [d.entropy for d in distributions]
    norm_entropies = [d.normalized_entropy for d in distributions]

    mean_conf = sum(confs) / float(n)
    sorted_confs = sorted(confs)
    median_conf = (
        sorted_confs[n // 2]
        if n % 2 == 1
        else (sorted_confs[n // 2 - 1] + sorted_confs[n // 2]) / 2.0
    )

    mean_ent = sum(entropies) / float(n)
    mean_norm_ent = sum(norm_entropies) / float(n)

    return ConfidenceSubsetSummary(
        sample_count=n,
        mean_max_probability=mean_conf,
        median_max_probability=median_conf,
        mean_entropy=mean_ent,
        mean_normalized_entropy=mean_norm_ent,
    )


def compute_calibration_report(
    distributions: Sequence[PredictiveDistribution],
    bin_count: int = 10,
    strategy: BinningStrategy = BinningStrategy.EQUAL_WIDTH,
    class_names: dict[int, str] | None = None,
) -> CalibrationReport:
    """Construct a comprehensive CalibrationReport from sample predictive distributions.

    Parameters
    ----------
    distributions : Sequence[PredictiveDistribution]
        List of predictive distributions for all evaluated samples.
    bin_count : int
        Number of reliability diagram bins.
    strategy : BinningStrategy
        Binning partition strategy.
    class_names : dict[int, str] | None
        Optional mapping of class indices to human-readable names.

    Returns
    -------
    CalibrationReport
        Structured calibration report with all metrics and reliability bins.
    """
    if not distributions:
        raise ValidationError(
            "Cannot compute CalibrationReport on empty distributions list."
        )

    n = len(distributions)
    warnings: list[str] = []

    if n < 30:
        warnings.append(
            f"Small evaluation set (N={n}). ECE/MCE metrics may have "
            "high sampling variance."
        )

    # 1. Convert to CalibrationSample list
    cal_samples: list[CalibrationSample] = []
    correct_dists: list[PredictiveDistribution] = []
    error_dists: list[PredictiveDistribution] = []

    for d in distributions:
        is_corr = d.is_correct if d.is_correct is not None else False
        cal_samples.append(
            CalibrationSample(
                sample_id=d.sample_id,
                confidence=d.max_probability,
                is_correct=is_corr,
                predicted_class=d.predicted_class,
                true_class=d.true_class if d.true_class is not None else 0,
                probabilities=d.probabilities,
            )
        )
        if is_corr:
            correct_dists.append(d)
        else:
            error_dists.append(d)

    # 2. Overall Accuracy and Mean Confidence
    overall_acc = sum(1.0 for s in cal_samples if s.is_correct) / float(n)
    mean_conf = sum(s.confidence for s in cal_samples) / float(n)

    # 3. Reliability Bins and ECE / MCE
    bins = compute_reliability_bins(cal_samples, bin_count=bin_count, strategy=strategy)
    ece = compute_expected_calibration_error(bins, total_sample_count=n)
    mce = compute_maximum_calibration_error(bins)

    # 4. Brier and NLL
    brier = compute_brier_score(distributions)
    nll = compute_negative_log_likelihood(distributions)

    # 5. Entropy summaries
    mean_entropy = sum(d.entropy for d in distributions) / float(n)
    mean_norm_entropy = sum(d.normalized_entropy for d in distributions) / float(n)

    # 6. Correct vs Incorrect Subset Summaries
    err_summary = compute_confidence_subset_summary(error_dists)
    corr_summary = compute_confidence_subset_summary(correct_dists)

    # 7. Class-conditional calibration
    class_groups: dict[int, list[PredictiveDistribution]] = {}
    for d in distributions:
        if d.true_class is not None:
            class_groups.setdefault(d.true_class, []).append(d)

    class_summaries: list[ClassCalibrationSummary] = []
    names = class_names or {}

    for c_id, c_dists in sorted(class_groups.items()):
        c_count = len(c_dists)
        c_name = names.get(c_id, f"class_{c_id}")
        c_acc = sum(1.0 for d in c_dists if d.is_correct) / float(c_count)
        c_conf = sum(d.max_probability for d in c_dists) / float(c_count)
        c_ent = sum(d.entropy for d in c_dists) / float(c_count)

        c_warning: str | None = None
        c_ece: float | None = None

        if c_count < 10:
            c_warning = f"Class sample count ({c_count}) < 10; ECE not estimated."
        else:
            c_cal_samples = [
                CalibrationSample(
                    sample_id=d.sample_id,
                    confidence=d.max_probability,
                    is_correct=bool(d.is_correct),
                    predicted_class=d.predicted_class,
                    true_class=d.true_class or 0,
                )
                for d in c_dists
            ]
            c_bins = compute_reliability_bins(c_cal_samples, bin_count=5)
            c_ece = compute_expected_calibration_error(
                c_bins, total_sample_count=c_count
            )

        class_summaries.append(
            ClassCalibrationSummary(
                class_id=c_id,
                class_name=c_name,
                sample_count=c_count,
                accuracy=c_acc,
                mean_confidence=c_conf,
                mean_entropy=c_ent,
                ece=c_ece,
                warning=c_warning,
            )
        )

    return CalibrationReport(
        sample_count=n,
        accuracy=overall_acc,
        mean_confidence=mean_conf,
        ece=ece,
        mce=mce,
        brier_score=brier,
        nll=nll,
        mean_predictive_entropy=mean_entropy,
        mean_normalized_entropy=mean_norm_entropy,
        binning_strategy=strategy,
        bin_count=bin_count,
        reliability_bins=bins,
        error_subset_summary=err_summary,
        correct_subset_summary=corr_summary,
        class_conditional_summaries=class_summaries,
        warnings=warnings,
    )
