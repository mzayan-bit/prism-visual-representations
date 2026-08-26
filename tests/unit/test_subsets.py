"""Unit tests for SubsetManifest and nested data-budget subset generation."""

import pytest

from prism.core.errors import ValidationError
from prism.data.partitions import (
    PartitionManifest,
    generate_partition_manifest,
)
from prism.data.samples import CanonicalSampleManifest, SampleRecord
from prism.data.subsets import (
    SubsetManifest,
    generate_nested_subsets,
)

SyntheticFixture = tuple[CanonicalSampleManifest, PartitionManifest]


@pytest.fixture
def synthetic_data() -> SyntheticFixture:
    # 200 samples across 2 classes (100 train, 100 test)
    samples: list[SampleRecord] = []
    for i in range(100):
        samples.append(
            SampleRecord(
                sample_id=f"ds-synth/train/{i:04d}",
                source_split="train",
                source_index=i,
                target=i % 2,
            )
        )
    for i in range(100):
        samples.append(
            SampleRecord(
                sample_id=f"ds-synth/test/{i:04d}",
                source_split="test",
                source_index=i,
                target=i % 2,
            )
        )
    canonical = CanonicalSampleManifest.create(
        dataset_id="ds-synthetic",
        samples=samples,
    )
    partition = generate_partition_manifest(
        canonical_manifest=canonical,
        split_ratios={"train": 1.0},
        source_split_filter="train",
        isolated_splits={"test": "test"},
        seed=42,
    )
    return canonical, partition


@pytest.mark.unit
def test_nested_subsets_strict_nesting(
    synthetic_data: SyntheticFixture,
) -> None:
    """Verify strictly nested subset property: S_1 ⊆ S_5 ⊆ ... ⊆ S_100."""
    canonical, partition = synthetic_data
    budgets = (0.01, 0.05, 0.10, 0.25, 0.50, 1.00)

    subsets = generate_nested_subsets(
        partition_manifest=partition,
        canonical_manifest=canonical,
        budget_ratios=budgets,
        target_split="train",
        seed=42,
        strategy="nested_stratified",
    )

    s1 = set(subsets[0.01].sample_ids)
    s5 = set(subsets[0.05].sample_ids)
    s10 = set(subsets[0.10].sample_ids)
    s25 = set(subsets[0.25].sample_ids)
    s50 = set(subsets[0.50].sample_ids)
    s100 = set(subsets[1.00].sample_ids)

    # 1. Check subset nesting
    assert s1.issubset(s5)
    assert s5.issubset(s10)
    assert s10.issubset(s25)
    assert s25.issubset(s50)
    assert s50.issubset(s100)

    # 2. Check 100% equals full training split
    train_ids = set(partition.get_split("train").sample_ids)
    assert s100 == train_ids

    # 3. Check exact sample counts
    assert len(s1) == 1
    assert len(s5) == 5
    assert len(s10) == 10
    assert len(s25) == 25
    assert len(s50) == 50
    assert len(s100) == 100


@pytest.mark.unit
def test_nested_subsets_deterministic_fingerprints(
    synthetic_data: SyntheticFixture,
) -> None:
    """Verify repeated subset generation produces identical fingerprints."""
    canonical, partition = synthetic_data

    subsets1 = generate_nested_subsets(
        partition_manifest=partition,
        canonical_manifest=canonical,
        budget_ratios=(0.10, 0.50, 1.00),
        target_split="train",
        seed=42,
    )

    subsets2 = generate_nested_subsets(
        partition_manifest=partition,
        canonical_manifest=canonical,
        budget_ratios=(0.10, 0.50, 1.00),
        target_split="train",
        seed=42,
    )

    for b in (0.10, 0.50, 1.00):
        assert subsets1[b].compute_fingerprint() == subsets2[b].compute_fingerprint()


@pytest.mark.unit
def test_nested_subsets_no_outside_samples(
    synthetic_data: SyntheticFixture,
) -> None:
    """Verify subsets contain zero samples outside parent training partition."""
    canonical, partition = synthetic_data
    subsets = generate_nested_subsets(
        partition_manifest=partition,
        canonical_manifest=canonical,
        budget_ratios=(0.10, 0.25),
        target_split="train",
        seed=42,
    )

    train_ids = set(partition.get_split("train").sample_ids)
    test_ids = set(partition.get_split("test").sample_ids)

    for _b, sub in subsets.items():
        sub_set = set(sub.sample_ids)
        assert sub_set.issubset(train_ids)
        assert len(sub_set.intersection(test_ids)) == 0


@pytest.mark.unit
def test_nested_subsets_rejects_invalid_budgets(
    synthetic_data: SyntheticFixture,
) -> None:
    """Verify generator rejects invalid budget fractions."""
    canonical, partition = synthetic_data

    with pytest.raises(ValidationError, match="Budget ratios must be in"):
        generate_nested_subsets(
            partition_manifest=partition,
            canonical_manifest=canonical,
            budget_ratios=(0.0, 0.5),  # 0.0 is invalid
            target_split="train",
        )

    with pytest.raises(ValidationError, match="Budget ratios must be in"):
        generate_nested_subsets(
            partition_manifest=partition,
            canonical_manifest=canonical,
            budget_ratios=(1.5,),  # > 1.0 is invalid
            target_split="train",
        )


@pytest.mark.unit
def test_subset_manifest_serialization_round_trip(
    synthetic_data: SyntheticFixture,
) -> None:
    """Verify SubsetManifest serialization round trips."""
    canonical, partition = synthetic_data
    subsets = generate_nested_subsets(
        partition_manifest=partition,
        canonical_manifest=canonical,
        budget_ratios=(0.10,),
        target_split="train",
        seed=42,
    )

    sub = subsets[0.10]
    dumped_dict = sub.to_dict()
    restored_dict = SubsetManifest.from_dict(dumped_dict)
    assert sub.compute_fingerprint() == restored_dict.compute_fingerprint()

    json_str = sub.to_json()
    restored_json = SubsetManifest.from_json(json_str)
    assert sub.compute_fingerprint() == restored_json.compute_fingerprint()
