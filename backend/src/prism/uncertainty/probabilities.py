"""Stable softmax, predictive distribution, Shannon entropy, and margin calculations."""

from __future__ import annotations

import math
from collections.abc import Sequence

from prism.core.errors import NumericalInstabilityError, ValidationError
from prism.uncertainty.contracts import PredictiveDistribution


def compute_stable_softmax(logits: list[float]) -> list[float]:
    """Compute numerically stable softmax probability distribution from logits.

    Parameters
    ----------
    logits : list[float]
        Raw model logit vector [K].

    Returns
    -------
    list[float]
        Probability distribution vector [K] summing to 1.0.
    """
    if not logits:
        raise ValidationError("Logits list cannot be empty.")

    max_z = max(logits)
    if math.isnan(max_z) or math.isinf(max_z):
        raise NumericalInstabilityError(
            f"Non-finite logit encountered in softmax: {logits}"
        )

    exps = [math.exp(z - max_z) for z in logits]
    sum_exps = sum(exps)

    if sum_exps <= 0.0 or math.isnan(sum_exps) or math.isinf(sum_exps):
        raise NumericalInstabilityError(
            f"Numerical overflow or underflow in softmax sum: sum_exps={sum_exps}"
        )

    probs = [e / sum_exps for e in exps]

    # Ensure precise unit normalization despite floating-point rounding
    sum_p = sum(probs)
    if sum_p > 0.0 and abs(sum_p - 1.0) > 1e-9:
        probs = [p / sum_p for p in probs]

    return probs


def compute_predictive_entropy(probabilities: list[float], eps: float = 1e-15) -> float:
    """Compute predictive Shannon entropy H(p) = -sum p_i * ln(p_i + eps) in nats.

    Parameters
    ----------
    probabilities : list[float]
        Probability distribution vector.
    eps : float
        Small positive epsilon to prevent log(0).

    Returns
    -------
    float
        Predictive Shannon entropy in nats (non-negative).
    """
    if not probabilities:
        return 0.0

    entropy = 0.0
    for p in probabilities:
        if p > 0.0:
            entropy -= p * math.log(max(p, eps))

    return max(0.0, entropy)


def compute_normalized_entropy(probabilities: list[float], eps: float = 1e-15) -> float:
    """Compute normalized Shannon entropy H(p) / ln(K) bounded in [0.0, 1.0].

    Parameters
    ----------
    probabilities : list[float]
        Probability distribution vector over K classes.
    eps : float
        Numerical stabilizer for logarithm.

    Returns
    -------
    float
        Normalized entropy in [0.0, 1.0]. Returns 0.0 when K=1.
    """
    k = len(probabilities)
    if k <= 1:
        return 0.0

    raw_entropy = compute_predictive_entropy(probabilities, eps=eps)
    max_entropy = math.log(k)

    if max_entropy <= 0.0:
        return 0.0

    return max(0.0, min(1.0, raw_entropy / max_entropy))


def compute_logit_margin(logits: list[float]) -> float:
    """Compute difference between highest and second-highest logit."""
    if len(logits) < 2:
        return 0.0

    sorted_logits = sorted(logits, reverse=True)
    return max(0.0, sorted_logits[0] - sorted_logits[1])


def compute_probability_margin(probabilities: list[float]) -> float:
    """Compute difference between highest and second-highest probability."""
    if len(probabilities) < 2:
        return 0.0

    sorted_probs = sorted(probabilities, reverse=True)
    return max(0.0, min(1.0, sorted_probs[0] - sorted_probs[1]))


def compute_predictive_distribution(
    sample_id: str,
    logits: list[float],
    true_class: int | None = None,
    temperature: float = 1.0,
    eps: float = 1e-15,
) -> PredictiveDistribution:
    """Construct a strongly typed PredictiveDistribution from sample logits.

    Parameters
    ----------
    sample_id : str
        Unique sample identifier.
    logits : list[float]
        Model output logits [K].
    true_class : int | None
        Ground truth integer label if available.
    temperature : float
        Positive temperature parameter T > 0 (default 1.0).
    eps : float
        Epsilon for entropy calculation.

    Returns
    -------
    PredictiveDistribution
        Validated predictive distribution record.
    """
    if temperature <= 0.0 or math.isnan(temperature) or math.isinf(temperature):
        raise ValidationError(
            f"Temperature must be strictly positive and finite: {temperature}"
        )

    scaled_logits = [z / temperature for z in logits]
    probs = compute_stable_softmax(scaled_logits)

    predicted_idx = max(range(len(probs)), key=lambda i: probs[i])
    max_prob = probs[predicted_idx]
    entropy = compute_predictive_entropy(probs, eps=eps)
    norm_entropy = compute_normalized_entropy(probs, eps=eps)
    logit_marg = compute_logit_margin(logits)
    prob_marg = compute_probability_margin(probs)

    is_corr: bool | None = None
    if true_class is not None:
        is_corr = predicted_idx == true_class

    is_fin = all(math.isfinite(z) for z in logits) and all(
        math.isfinite(p) for p in probs
    )

    return PredictiveDistribution(
        sample_id=sample_id,
        logits=logits,
        probabilities=probs,
        predicted_class=predicted_idx,
        true_class=true_class,
        max_probability=max_prob,
        entropy=entropy,
        normalized_entropy=norm_entropy,
        logit_margin=logit_marg,
        probability_margin=prob_marg,
        is_correct=is_corr,
        is_finite=is_fin,
    )


def batch_predictive_distributions(
    sample_ids: Sequence[str],
    logits_matrix: Sequence[list[float]],
    true_classes: Sequence[int | None] | None = None,
    temperature: float = 1.0,
) -> list[PredictiveDistribution]:
    """Batch construct PredictiveDistribution instances for multiple samples."""
    if len(sample_ids) != len(logits_matrix):
        raise ValidationError(
            f"Sample IDs ({len(sample_ids)}) != logits ({len(logits_matrix)})."
        )

    targets = true_classes if true_classes is not None else [None] * len(sample_ids)
    if len(targets) != len(sample_ids):
        raise ValidationError(
            f"Targets ({len(targets)}) != samples ({len(sample_ids)})."
        )

    results: list[PredictiveDistribution] = []
    for s_id, row, y in zip(sample_ids, logits_matrix, targets, strict=True):
        dist = compute_predictive_distribution(
            sample_id=s_id,
            logits=row,
            true_class=y,
            temperature=temperature,
        )
        results.append(dist)

    return results
