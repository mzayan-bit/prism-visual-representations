"""Unit tests for CIFAR-10 and CIFAR-100 benchmark adapters."""

import pytest

from prism.core.errors import ConfigurationError
from prism.data.adapters import CIFAR10Adapter, CIFAR100Adapter
from prism.data.manifests import DatasetManifest
from prism.data.partitions import PartitionManifest
from prism.data.samples import CanonicalSampleManifest


@pytest.mark.unit
def test_cifar10_adapter_contracts() -> None:
    """Verify CIFAR-10 metadata, canonical universe, and partition contracts."""
    adapter = CIFAR10Adapter()

    assert adapter.dataset_id == "ds-cifar10"
    assert adapter.name == "CIFAR-10"
    assert adapter.num_classes == 10
    assert len(adapter.classes) == 10
    assert "airplane" in adapter.classes

    manifest = adapter.get_dataset_manifest()
    assert isinstance(manifest, DatasetManifest)
    assert manifest.num_classes == 10
    assert len(manifest.splits) == 2

    canonical = adapter.get_canonical_manifest()
    assert isinstance(canonical, CanonicalSampleManifest)
    assert canonical.num_samples == 60000
    assert len(canonical.filter_by_source_split("train")) == 50000
    assert len(canonical.filter_by_source_split("test")) == 10000

    partition = adapter.get_default_partition(seed=42, val_ratio=0.1)
    assert isinstance(partition, PartitionManifest)
    assert partition.total_samples == 60000
    assert partition.get_split("train").num_samples == 45000
    assert partition.get_split("val").num_samples == 5000
    assert partition.get_split("test").num_samples == 10000


@pytest.mark.unit
def test_cifar100_adapter_contracts() -> None:
    """Verify CIFAR-100 metadata, canonical universe, and partition contracts."""
    adapter = CIFAR100Adapter()

    assert adapter.dataset_id == "ds-cifar100"
    assert adapter.name == "CIFAR-100"
    assert adapter.num_classes == 100
    assert len(adapter.classes) == 100
    assert "apple" in adapter.classes

    manifest = adapter.get_dataset_manifest()
    assert isinstance(manifest, DatasetManifest)
    assert manifest.num_classes == 100

    canonical = adapter.get_canonical_manifest()
    assert isinstance(canonical, CanonicalSampleManifest)
    assert canonical.num_samples == 60000
    assert len(canonical.filter_by_source_split("train")) == 50000
    assert len(canonical.filter_by_source_split("test")) == 10000

    partition = adapter.get_default_partition(seed=42, val_ratio=0.1)
    assert isinstance(partition, PartitionManifest)
    assert partition.total_samples == 60000
    assert partition.get_split("train").num_samples == 45000
    assert partition.get_split("val").num_samples == 5000
    assert partition.get_split("test").num_samples == 10000


@pytest.mark.unit
def test_cifar_nested_subsets_generation() -> None:
    """Verify CIFAR adapter generates nested subsets for low-data regimes."""
    adapter = CIFAR10Adapter()
    subsets = adapter.get_nested_subsets(budget_ratios=(0.01, 0.10, 1.00), seed=42)

    assert 0.01 in subsets
    assert 0.10 in subsets
    assert 1.00 in subsets

    s1 = set(subsets[0.01].sample_ids)
    s10 = set(subsets[0.10].sample_ids)
    s100 = set(subsets[1.00].sample_ids)

    assert s1.issubset(s10)
    assert s10.issubset(s100)
    assert len(s1) == 450  # 1% of 45,000
    assert len(s10) == 4500  # 10% of 45,000
    assert len(s100) == 45000  # 100% of 45,000


@pytest.mark.unit
def test_cifar_adapter_optional_torchvision_missing() -> None:
    """Verify missing torchvision raises ConfigurationError on raw dataset load."""
    adapter = CIFAR10Adapter()

    # When torchvision is missing, load_raw_dataset should raise ConfigurationError
    with pytest.raises(ConfigurationError, match="torchvision is required"):
        adapter.load_raw_dataset(split="train", download=False)
