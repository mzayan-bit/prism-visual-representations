"""Unit tests for batch loading and batch traceability."""

import pytest

from prism.core.enums import OrderingStrategy
from prism.core.errors import ValidationError
from prism.data.batching import DeterministicBatchLoader, MaterializedBatch
from prism.data.materialized import MaterializedDataset, MaterializedSample


@pytest.fixture
def test_dataset() -> MaterializedDataset:
    samples = [
        MaterializedSample(
            sample_id=f"ds-synth/train/{i:04d}",
            source_split="train",
            source_index=i,
            data=[float(i)],
            target=i % 2,
        )
        for i in range(10)
    ]
    return MaterializedDataset(dataset_id="ds-synthetic", samples=samples)


@pytest.mark.unit
def test_batch_loader_full_and_partial_batches(
    test_dataset: MaterializedDataset,
) -> None:
    """Verify batch loader with drop_last=False produces correct partial final batch."""
    loader = DeterministicBatchLoader(
        dataset=test_dataset,
        batch_size=4,
        drop_last=False,
    )

    assert len(loader) == 3  # 4 + 4 + 2

    batches = list(loader)
    assert len(batches) == 3

    assert batches[0].batch_size == 4
    assert batches[0].batch_index == 0
    assert len(batches[0].sample_ids) == 4

    assert batches[1].batch_size == 4
    assert batches[1].batch_index == 1

    assert batches[2].batch_size == 2
    assert batches[2].batch_index == 2
    assert batches[2].sample_ids == ["ds-synth/train/0008", "ds-synth/train/0009"]


@pytest.mark.unit
def test_batch_loader_drop_last(test_dataset: MaterializedDataset) -> None:
    """Verify batch loader with drop_last=True removes partial final batch."""
    loader = DeterministicBatchLoader(
        dataset=test_dataset,
        batch_size=4,
        drop_last=True,
    )

    assert len(loader) == 2  # Only two full batches of 4
    batches = list(loader)
    assert len(batches) == 2
    assert batches[0].batch_size == 4
    assert batches[1].batch_size == 4


@pytest.mark.unit
def test_batch_loader_preserves_sample_traceability(
    test_dataset: MaterializedDataset,
) -> None:
    """Verify every batch contains exact traceable canonical sample IDs."""
    loader = DeterministicBatchLoader(
        dataset=test_dataset,
        batch_size=5,
        ordering_strategy=OrderingStrategy.FIXED_SHUFFLE,
        seed=42,
    )

    batches = list(loader)
    all_batch_ids = []
    for b in batches:
        assert isinstance(b, MaterializedBatch)
        all_batch_ids.extend(b.sample_ids)

    assert len(all_batch_ids) == 10
    assert set(all_batch_ids) == set(test_dataset.sample_ids)


@pytest.mark.unit
def test_batch_loader_rejects_invalid_batch_size(
    test_dataset: MaterializedDataset,
) -> None:
    """Verify batch loader rejects batch_size <= 0."""
    with pytest.raises(ValidationError, match="batch_size must be positive"):
        DeterministicBatchLoader(dataset=test_dataset, batch_size=0)
