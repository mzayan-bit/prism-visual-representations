"""Unit tests for DataPreparer and DataRuntimeContext."""

import pytest

from prism.core.enums import OrderingStrategy
from prism.data.context import DataRuntimeContext
from prism.data.preparer import DataPreparer
from prism.data.synthetic import SyntheticVisionAdapter


@pytest.mark.unit
def test_data_preparer_flow() -> None:
    """Verify DataPreparer creates dataset, loader, and DataRuntimeContext."""
    adapter = SyntheticVisionAdapter(num_train=50, num_test=10, num_classes=2)
    canonical = adapter.get_canonical_manifest()
    partition = adapter.get_default_partition(seed=42)
    subsets = adapter.get_nested_subsets(seed=42)

    preparer = DataPreparer()

    dataset, loader, context = preparer.prepare(
        adapter=adapter,
        canonical_manifest=canonical,
        partition_manifest=partition,
        subset_manifest=subsets[0.10],
        batch_size=2,
        ordering_strategy=OrderingStrategy.SEQUENTIAL,
        seed=42,
    )

    assert len(dataset) == 4  # 10% of 40 train samples
    assert len(loader) == 2
    assert isinstance(context, DataRuntimeContext)
    assert context.dataset_id == adapter.dataset_id
    assert context.resolved_sample_count == 4
    assert context.canonical_manifest_fingerprint == canonical.compute_fingerprint()
    assert context.partition_manifest_fingerprint == partition.compute_fingerprint()
    assert context.subset_manifest_fingerprint == subsets[0.10].compute_fingerprint()


@pytest.mark.unit
def test_data_runtime_context_serialization_round_trip() -> None:
    """Verify DataRuntimeContext serializes and deserializes cleanly."""
    context = DataRuntimeContext(
        dataset_id="ds-cifar10",
        canonical_manifest_fingerprint="abc123canonical",
        partition_manifest_fingerprint="def456partition",
        subset_manifest_fingerprint="ghi789subset",
        resolved_sample_count=4500,
        ordering_strategy="fixed_shuffle",
        ordering_fingerprint="jkl012ordering",
        batch_size=64,
        drop_last=False,
        backend_name="in_memory",
        metadata={"seed": 42},
    )

    # Dict round trip
    dumped_dict = context.to_dict()
    restored_dict = DataRuntimeContext.from_dict(dumped_dict)
    assert context.compute_fingerprint() == restored_dict.compute_fingerprint()

    # JSON round trip
    json_str = context.to_json()
    restored_json = DataRuntimeContext.from_json(json_str)
    assert context.compute_fingerprint() == restored_json.compute_fingerprint()
