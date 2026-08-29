"""Unit tests for attention representation contracts, entropy, and summaries."""

import math

import pytest

from prism.core.errors import SerializationError, ValidationError
from prism.representations.attention import (
    AttentionHeadSummary,
    AttentionTensorSummary,
    compare_attention_summaries,
    compute_attention_entropy,
    compute_diagonal_attention_mass,
    summarize_attention_weights,
)


@pytest.mark.unit
def test_summarize_attention_weights_uniform_distribution() -> None:
    """Verify summarize_attention_weights on uniform matrix gives log(L)."""
    # N=1, H=2, L=4, L=4 with uniform weights 1/4 = 0.25
    uniform_weights = [[[[0.25 for _ in range(4)] for _ in range(4)] for _ in range(2)]]
    summary = summarize_attention_weights(uniform_weights)

    assert isinstance(summary, AttentionTensorSummary)
    assert summary.batch_size == 1
    assert summary.num_heads == 2
    assert summary.seq_len == 4
    assert summary.is_row_normalized is True
    assert summary.is_finite is True
    assert summary.min_value == pytest.approx(0.25, abs=1e-6)
    assert summary.max_value == pytest.approx(0.25, abs=1e-6)
    assert summary.mean_value == pytest.approx(0.25, abs=1e-6)
    assert summary.mean_diagonal_mass == pytest.approx(0.25, abs=1e-6)

    # For uniform distribution of size 4, entropy = log(4) = 2 * log(2) ~ 1.386294
    expected_entropy = math.log(4.0)
    assert summary.mean_entropy == pytest.approx(expected_entropy, abs=1e-5)
    assert len(summary.head_summaries) == 2
    for h in summary.head_summaries:
        assert isinstance(h, AttentionHeadSummary)
        assert h.entropy == pytest.approx(expected_entropy, abs=1e-5)
        assert h.diagonal_mass == pytest.approx(0.25, abs=1e-6)
        assert h.is_row_normalized is True


@pytest.mark.unit
def test_summarize_attention_weights_peaked_distribution() -> None:
    """Verify summarize_attention_weights on one-hot matrix gives zero entropy."""
    # N=1, H=1, L=3, L=3 with identity diagonal (deterministic hard attention)
    identity_weights = [
        [
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ]
    ]
    summary = summarize_attention_weights(identity_weights)

    assert summary.mean_entropy == pytest.approx(0.0, abs=1e-6)
    assert summary.min_value == 0.0
    assert summary.max_value == 1.0
    assert summary.mean_diagonal_mass == pytest.approx(1.0, abs=1e-6)
    assert summary.zero_fraction == pytest.approx(6.0 / 9.0, abs=1e-6)


@pytest.mark.unit
def test_compute_attention_entropy_and_diagonal_mass() -> None:
    """Verify compute_attention_entropy and compute_diagonal_attention_mass."""
    # 1 sample, 1 head, 2 queries: row 0 uniform [0.5, 0.5], row 1 one-hot [0.0, 1.0]
    weights = [[[[0.5, 0.5], [0.0, 1.0]]]]
    entropies = compute_attention_entropy(weights)

    assert len(entropies) == 1 and len(entropies[0]) == 1 and len(entropies[0][0]) == 2
    assert entropies[0][0][0] == pytest.approx(math.log(2.0), abs=1e-5)
    assert entropies[0][0][1] == pytest.approx(0.0, abs=1e-5)

    diag_mass = compute_diagonal_attention_mass(weights)
    # Diagonal: weights[0][0][0][0] = 0.5, weights[0][0][1][1] = 1.0 -> avg 0.75
    assert diag_mass == pytest.approx(0.75, abs=1e-6)


@pytest.mark.unit
def test_compare_attention_summaries() -> None:
    """Verify compare_attention_summaries computes correct metric shifts."""
    uniform_weights = [[[[0.5, 0.5], [0.5, 0.5]]]]
    peaked_weights = [[[[1.0, 0.0], [0.0, 1.0]]]]

    summary_a = summarize_attention_weights(uniform_weights)
    summary_b = summarize_attention_weights(peaked_weights)

    comparison = compare_attention_summaries(summary_a, summary_b)

    # Entropy should decrease from log(2) to 0.0
    assert comparison["mean_entropy_delta"] == pytest.approx(-math.log(2.0), abs=1e-5)
    # Diagonal mass should increase from 0.5 to 1.0
    assert comparison["diagonal_mass_delta"] == pytest.approx(0.5, abs=1e-6)
    assert comparison["max_value_delta"] == pytest.approx(0.5, abs=1e-6)

    # Rejection if head count mismatch
    multihead_weights = [[[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]]]]
    summary_multi = summarize_attention_weights(multihead_weights)
    with pytest.raises(ValidationError, match="different head counts"):
        compare_attention_summaries(summary_a, summary_multi)


@pytest.mark.unit
def test_summarize_attention_weights_normalization_detection() -> None:
    """Verify is_row_normalized becomes False when rows do not sum to 1.0."""
    unnormalized_weights = [
        [
            [
                [0.5, 0.5, 0.5],
                [0.1, 0.1, 0.1],
                [1.0, 0.0, 0.0],
            ]
        ]
    ]
    summary = summarize_attention_weights(unnormalized_weights, tolerance=1e-4)
    assert summary.is_row_normalized is False
    assert summary.head_summaries[0].is_row_normalized is False


@pytest.mark.unit
def test_attention_summary_serialization_roundtrip() -> None:
    """Verify serialization and restoration for attention summaries."""
    weights = [[[[0.5, 0.5], [0.8, 0.2]]]]
    summary = summarize_attention_weights(weights)

    # Dictionary roundtrip
    data_dict = summary.to_dict()
    restored_dict = AttentionTensorSummary.from_dict(data_dict)
    assert restored_dict == summary

    # JSON string roundtrip
    json_str = summary.to_json(indent=2)
    restored_json = AttentionTensorSummary.from_json(json_str)
    assert restored_json == summary

    # Rejection of invalid JSON
    with pytest.raises(SerializationError, match="Invalid JSON string"):
        AttentionTensorSummary.from_json("invalid json {")
