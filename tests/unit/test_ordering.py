"""Unit tests for deterministic data ordering and fingerprints."""

import pytest

from prism.core.enums import OrderingStrategy
from prism.data.ordering import (
    OrderingSpecification,
    compute_ordering_fingerprint,
    compute_sample_order,
)


@pytest.fixture
def sample_ids() -> list[str]:
    return [f"sample_{i:04d}" for i in range(20)]


@pytest.mark.unit
def test_ordering_specification_valid() -> None:
    """Verify OrderingSpecification validation and defaults."""
    spec = OrderingSpecification(strategy="sequential")
    assert spec.strategy == OrderingStrategy.SEQUENTIAL
    assert spec.seed == 42
    assert spec.epoch == 0

    spec_shuffle = OrderingSpecification(
        strategy=OrderingStrategy.EPOCH_AWARE_SHUFFLE,
        seed=100,
        epoch=3,
    )
    assert spec_shuffle.strategy == OrderingStrategy.EPOCH_AWARE_SHUFFLE
    assert spec_shuffle.epoch == 3


@pytest.mark.unit
def test_sequential_ordering_preserves_manifest_order(
    sample_ids: list[str],
) -> None:
    """Verify SEQUENTIAL strategy returns identity index list."""
    order = compute_sample_order(sample_ids, strategy=OrderingStrategy.SEQUENTIAL)
    assert order == list(range(len(sample_ids)))


@pytest.mark.unit
def test_fixed_shuffle_deterministic(sample_ids: list[str]) -> None:
    """Verify FIXED_SHUFFLE is deterministic with same seed and diverges."""
    order1 = compute_sample_order(
        sample_ids, strategy=OrderingStrategy.FIXED_SHUFFLE, seed=42
    )
    order2 = compute_sample_order(
        sample_ids, strategy=OrderingStrategy.FIXED_SHUFFLE, seed=42
    )
    order_diff = compute_sample_order(
        sample_ids, strategy=OrderingStrategy.FIXED_SHUFFLE, seed=999
    )

    assert order1 == order2
    assert order1 != list(range(len(sample_ids)))
    assert order1 != order_diff
    assert set(order1) == set(range(len(sample_ids)))


@pytest.mark.unit
def test_epoch_aware_shuffle(sample_ids: list[str]) -> None:
    """Verify EPOCH_AWARE_SHUFFLE produces reproducible per-epoch orders."""
    epoch0_a = compute_sample_order(
        sample_ids,
        strategy=OrderingStrategy.EPOCH_AWARE_SHUFFLE,
        seed=42,
        epoch=0,
    )
    epoch0_b = compute_sample_order(
        sample_ids,
        strategy=OrderingStrategy.EPOCH_AWARE_SHUFFLE,
        seed=42,
        epoch=0,
    )
    epoch1 = compute_sample_order(
        sample_ids,
        strategy=OrderingStrategy.EPOCH_AWARE_SHUFFLE,
        seed=42,
        epoch=1,
    )

    assert epoch0_a == epoch0_b
    assert epoch0_a != epoch1
    assert set(epoch0_a) == set(range(len(sample_ids)))
    assert set(epoch1) == set(range(len(sample_ids)))


@pytest.mark.unit
def test_ordering_fingerprints(sample_ids: list[str]) -> None:
    """Verify ordering fingerprint sensitivity to strategy, seed, and epoch."""
    fp_seq = compute_ordering_fingerprint(
        sample_ids, strategy=OrderingStrategy.SEQUENTIAL
    )
    fp_shuffle1 = compute_ordering_fingerprint(
        sample_ids, strategy=OrderingStrategy.FIXED_SHUFFLE, seed=42
    )
    fp_shuffle2 = compute_ordering_fingerprint(
        sample_ids, strategy=OrderingStrategy.FIXED_SHUFFLE, seed=42
    )
    fp_shuffle_diff_seed = compute_ordering_fingerprint(
        sample_ids, strategy=OrderingStrategy.FIXED_SHUFFLE, seed=999
    )
    fp_epoch0 = compute_ordering_fingerprint(
        sample_ids,
        strategy=OrderingStrategy.EPOCH_AWARE_SHUFFLE,
        seed=42,
        epoch=0,
    )
    fp_epoch1 = compute_ordering_fingerprint(
        sample_ids,
        strategy=OrderingStrategy.EPOCH_AWARE_SHUFFLE,
        seed=42,
        epoch=1,
    )

    assert fp_shuffle1 == fp_shuffle2
    assert fp_seq != fp_shuffle1
    assert fp_shuffle1 != fp_shuffle_diff_seed
    assert fp_epoch0 != fp_epoch1
