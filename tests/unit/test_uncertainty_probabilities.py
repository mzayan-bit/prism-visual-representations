"""Unit tests for numerical probability foundations and uncertainty distributions."""

from __future__ import annotations

import math

import pytest

from prism.core.errors import ValidationError
from prism.uncertainty.contracts import PredictiveDistribution
from prism.uncertainty.probabilities import (
    batch_predictive_distributions,
    compute_logit_margin,
    compute_normalized_entropy,
    compute_predictive_distribution,
    compute_predictive_entropy,
    compute_probability_margin,
    compute_stable_softmax,
)


def test_stable_softmax_sum_and_shift_invariance() -> None:
    """Verify softmax sums to 1.0 and is invariant to constant additive shifts."""
    logits = [2.0, 1.0, 0.1, -1.5]
    probs = compute_stable_softmax(logits)

    assert len(probs) == len(logits)
    assert abs(sum(probs) - 1.0) < 1e-6
    assert all(0.0 <= p <= 1.0 for p in probs)

    # Shift invariance: softmax(z + c) == softmax(z)
    shift = 1000.0
    shifted_logits = [z + shift for z in logits]
    shifted_probs = compute_stable_softmax(shifted_logits)

    for p1, p2 in zip(probs, shifted_probs, strict=True):
        assert abs(p1 - p2) < 1e-6


def test_stable_softmax_extreme_logits() -> None:
    """Verify numerical stability on extreme logit values without overflow."""
    extreme_logits = [1000.0, 999.0, 500.0, -500.0]
    probs = compute_stable_softmax(extreme_logits)

    assert abs(sum(probs) - 1.0) < 1e-6
    assert probs[0] > probs[1] > probs[2] >= 0.0


def test_stable_softmax_empty() -> None:
    """Verify error on empty logits."""
    with pytest.raises(ValidationError, match="cannot be empty"):
        compute_stable_softmax([])


def test_predictive_distribution_temperature() -> None:
    """Verify temperature scaling effects on predictive distributions."""
    logits = [3.0, 1.0, 0.0]

    # High temperature -> uniform
    high_t = compute_predictive_distribution("sample_high", logits, temperature=100.0)
    for p in high_t.probabilities:
        assert abs(p - 1.0 / 3.0) < 0.02

    # Low temperature -> sharp
    low_t = compute_predictive_distribution("sample_low", logits, temperature=0.1)
    assert low_t.probabilities[0] > 0.99
    assert low_t.probabilities[1] < 0.01

    # Invalid non-positive temperature
    with pytest.raises(ValidationError, match="strictly positive"):
        compute_predictive_distribution("sample_bad", logits, temperature=0.0)
    with pytest.raises(ValidationError, match="strictly positive"):
        compute_predictive_distribution("sample_bad", logits, temperature=-1.0)


def test_predictive_entropy_bounds() -> None:
    """Verify predictive entropy is 0 for one-hot and ln(K) for uniform."""
    # One-hot distribution -> H(p) = 0
    one_hot = [1.0, 0.0, 0.0, 0.0]
    assert compute_predictive_entropy(one_hot) == pytest.approx(0.0, abs=1e-5)

    # Uniform distribution -> H(p) = ln(K)
    num_classes = 4
    uniform = [1.0 / num_classes] * num_classes
    expected_entropy = math.log(num_classes)
    assert compute_predictive_entropy(uniform) == pytest.approx(
        expected_entropy, rel=1e-4
    )


def test_normalized_entropy_range() -> None:
    """Verify normalized entropy is in [0.0, 1.0]."""
    # 2 classes
    assert compute_normalized_entropy([1.0, 0.0]) == pytest.approx(0.0, abs=1e-5)
    assert compute_normalized_entropy([0.5, 0.5]) == pytest.approx(1.0, rel=1e-4)

    # Single class edge case
    assert compute_normalized_entropy([1.0]) == 0.0


def test_margins() -> None:
    """Verify logit and probability top-1 vs top-2 margins."""
    logits = [10.0, 7.0, 3.0]
    assert compute_logit_margin(logits) == pytest.approx(3.0, rel=1e-5)

    probs = [0.8, 0.15, 0.05]
    assert compute_probability_margin(probs) == pytest.approx(0.65, rel=1e-5)

    # Single class
    assert compute_logit_margin([5.0]) == 0.0
    assert compute_probability_margin([1.0]) == 0.0


def test_batch_predictive_distributions() -> None:
    """Verify batch processing into PredictiveDistribution items."""
    sample_ids = ["s0", "s1", "s2"]
    logits_matrix = [
        [4.0, 1.0, 0.0],
        [1.0, 5.0, 2.0],
        [2.0, 2.0, 2.0],
    ]
    true_classes = [0, 1, 2]

    dists = batch_predictive_distributions(
        sample_ids=sample_ids,
        logits_matrix=logits_matrix,
        true_classes=true_classes,
        temperature=1.0,
    )

    assert isinstance(dists, list)
    assert len(dists) == 3

    d0 = dists[0]
    assert isinstance(d0, PredictiveDistribution)
    assert d0.sample_id == "s0"
    assert d0.predicted_class == 0
    assert d0.true_class == 0
    assert d0.max_probability > 0.9
    assert d0.entropy >= 0.0

    d1 = dists[1]
    assert d1.predicted_class == 1
    assert d1.true_class == 1

    d2 = dists[2]
    assert d2.entropy == pytest.approx(math.log(3), rel=1e-4)
