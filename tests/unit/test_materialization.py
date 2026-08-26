"""Unit tests for dataset materialization and MaterializedDataset."""

import pytest

from prism.core.errors import (
    DatasetMaterializationError,
    ValidationError,
)
from prism.data.adapters import CIFAR10Adapter
from prism.data.materialized import MaterializedDataset, MaterializedSample
from prism.data.materializer import DatasetMaterializer
from prism.data.synthetic import SyntheticVisionAdapter


@pytest.fixture
def synthetic_adapter() -> SyntheticVisionAdapter:
    return SyntheticVisionAdapter(num_train=20, num_test=10, num_classes=2)


@pytest.fixture
def sample_list() -> list[MaterializedSample]:
    return [
        MaterializedSample(
            sample_id=f"ds-synth/train/{i:04d}",
            source_split="train",
            source_index=i,
            data=[float(i), float(i + 1)],
            target=i % 2,
        )
        for i in range(5)
    ]


@pytest.mark.unit
def test_materialized_dataset_basic_access(
    sample_list: list[MaterializedSample],
) -> None:
    """Verify MaterializedDataset length, indexing, lookups, and properties."""
    ds = MaterializedDataset(
        dataset_id="ds-synthetic",
        samples=sample_list,
        split_name="train",
    )

    assert len(ds) == 5
    assert ds.sample_ids == [f"ds-synth/train/{i:04d}" for i in range(5)]
    assert ds.targets == [0, 1, 0, 1, 0]

    sample0 = ds[0]
    assert sample0.sample_id == "ds-synth/train/0000"
    assert sample0.data == [0.0, 1.0]

    sample_lookup = ds.get_sample("ds-synth/train/0002")
    assert sample_lookup.source_index == 2
    assert sample_lookup.target == 0

    with pytest.raises(KeyError):
        ds.get_sample("nonexistent_id")

    with pytest.raises(IndexError):
        _ = ds[99]


@pytest.mark.unit
def test_materialized_dataset_slicing(
    sample_list: list[MaterializedSample],
) -> None:
    """Verify MaterializedDataset supports slicing into child datasets."""
    ds = MaterializedDataset(
        dataset_id="ds-synthetic",
        samples=sample_list,
    )
    sliced = ds[1:4]
    assert isinstance(sliced, MaterializedDataset)
    assert len(sliced) == 3
    assert sliced.sample_ids == [
        "ds-synth/train/0001",
        "ds-synth/train/0002",
        "ds-synth/train/0003",
    ]


@pytest.mark.unit
def test_materialized_dataset_rejects_duplicate_ids() -> None:
    """Verify MaterializedDataset rejects duplicate sample IDs."""
    duplicate_samples = [
        MaterializedSample(
            sample_id="ds-synth/train/0001",
            source_split="train",
            source_index=0,
            data=None,
        ),
        MaterializedSample(
            sample_id="ds-synth/train/0001",  # duplicate ID
            source_split="train",
            source_index=1,
            data=None,
        ),
    ]
    with pytest.raises(ValidationError, match="Duplicate sample_id"):
        MaterializedDataset(
            dataset_id="ds-synthetic",
            samples=duplicate_samples,
        )


@pytest.mark.unit
def test_materializer_resolves_exact_samples(
    synthetic_adapter: SyntheticVisionAdapter,
) -> None:
    """Verify DatasetMaterializer resolves canonical samples with exact identities."""
    materializer = DatasetMaterializer()
    canonical = synthetic_adapter.get_canonical_manifest()
    partition = synthetic_adapter.get_default_partition(seed=42)
    subsets = synthetic_adapter.get_nested_subsets(seed=42)

    # 1. Materialize full partition split (train)
    ds_train = materializer.materialize(
        adapter=synthetic_adapter,
        canonical_manifest=canonical,
        partition_manifest=partition,
        split_name="train",
    )
    assert len(ds_train) == partition.get_split("train").num_samples
    assert ds_train.sample_ids == partition.get_split("train").sample_ids

    # 2. Materialize subset (10%)
    ds_subset = materializer.materialize(
        adapter=synthetic_adapter,
        canonical_manifest=canonical,
        partition_manifest=partition,
        subset_manifest=subsets[0.10],
    )
    assert len(ds_subset) == len(subsets[0.10].sample_ids)
    assert ds_subset.sample_ids == subsets[0.10].sample_ids


@pytest.mark.unit
def test_materializer_rejects_mismatched_adapter(
    synthetic_adapter: SyntheticVisionAdapter,
) -> None:
    """Verify materializer rejects adapter that does not match canonical manifest."""
    materializer = DatasetMaterializer()
    cifar_adapter = CIFAR10Adapter()
    canonical = synthetic_adapter.get_canonical_manifest()

    with pytest.raises(DatasetMaterializationError, match="does not match"):
        materializer.materialize(
            adapter=cifar_adapter,
            canonical_manifest=canonical,
        )
