"""Unit tests for PartitionManifest and deterministic partition generation."""

import pytest

from prism.core.errors import ValidationError
from prism.data.partitions import (
    PartitionManifest,
    generate_partition_manifest,
)
from prism.data.samples import CanonicalSampleManifest, SampleRecord


@pytest.fixture
def canonical_universe() -> CanonicalSampleManifest:
    # 100 samples across 2 classes (50 train, 50 test in official split)
    samples: list[SampleRecord] = []
    for i in range(50):
        samples.append(
            SampleRecord(
                sample_id=f"ds-synth/train/{i:04d}",
                source_split="train",
                source_index=i,
                target=i % 2,
            )
        )
    for i in range(50):
        samples.append(
            SampleRecord(
                sample_id=f"ds-synth/test/{i:04d}",
                source_split="test",
                source_index=i,
                target=i % 2,
            )
        )
    return CanonicalSampleManifest.create(
        dataset_id="ds-synthetic",
        samples=samples,
    )


@pytest.mark.unit
def test_partition_manifest_generation_deterministic(
    canonical_universe: CanonicalSampleManifest,
) -> None:
    """Verify same seed produces identical partition fingerprints."""
    part1 = generate_partition_manifest(
        canonical_manifest=canonical_universe,
        split_ratios={"train": 0.8, "val": 0.2},
        seed=42,
        strategy="stratified",
        source_split_filter="train",
        isolated_splits={"test": "test"},
    )

    part2 = generate_partition_manifest(
        canonical_manifest=canonical_universe,
        split_ratios={"train": 0.8, "val": 0.2},
        seed=42,
        strategy="stratified",
        source_split_filter="train",
        isolated_splits={"test": "test"},
    )

    assert part1.total_samples == 100
    assert part1.get_split("train").num_samples == 40
    assert part1.get_split("val").num_samples == 10
    assert part1.get_split("test").num_samples == 50
    assert part1.compute_fingerprint() == part2.compute_fingerprint()


@pytest.mark.unit
def test_partition_manifest_different_seeds_diverge(
    canonical_universe: CanonicalSampleManifest,
) -> None:
    """Verify different seeds produce different valid partition assignments."""
    part1 = generate_partition_manifest(
        canonical_manifest=canonical_universe,
        split_ratios={"train": 0.8, "val": 0.2},
        seed=42,
        strategy="stratified",
        source_split_filter="train",
        isolated_splits={"test": "test"},
    )

    part2 = generate_partition_manifest(
        canonical_manifest=canonical_universe,
        split_ratios={"train": 0.8, "val": 0.2},
        seed=999,
        strategy="stratified",
        source_split_filter="train",
        isolated_splits={"test": "test"},
    )

    assert part1.compute_fingerprint() != part2.compute_fingerprint()


@pytest.mark.unit
def test_partition_mutual_exclusivity(
    canonical_universe: CanonicalSampleManifest,
) -> None:
    """Verify partition splits have zero sample overlap."""
    part = generate_partition_manifest(
        canonical_manifest=canonical_universe,
        split_ratios={"train": 0.8, "val": 0.2},
        seed=42,
        strategy="stratified",
        source_split_filter="train",
        isolated_splits={"test": "test"},
    )

    train_ids = set(part.get_split("train").sample_ids)
    val_ids = set(part.get_split("val").sample_ids)
    test_ids = set(part.get_split("test").sample_ids)

    assert len(train_ids.intersection(val_ids)) == 0
    assert len(train_ids.intersection(test_ids)) == 0
    assert len(val_ids.intersection(test_ids)) == 0


@pytest.mark.unit
def test_partition_validation_against_canonical(
    canonical_universe: CanonicalSampleManifest,
) -> None:
    """Verify validate_against_canonical passes on valid and fails on mismatch."""
    part = generate_partition_manifest(
        canonical_manifest=canonical_universe,
        split_ratios={"train": 0.8, "val": 0.2},
        seed=42,
        strategy="stratified",
        source_split_filter="train",
        isolated_splits={"test": "test"},
    )

    # Valid universe passes cleanly
    part.validate_against_canonical(canonical_universe)

    # Mismatched canonical universe raises ValidationError
    other_samples = [
        SampleRecord(
            sample_id=f"ds-other/train/{i:04d}",
            source_split="train",
            source_index=i,
        )
        for i in range(10)
    ]
    other_canonical = CanonicalSampleManifest.create(
        dataset_id="ds-other",
        samples=other_samples,
    )

    with pytest.raises(ValidationError, match="fingerprint"):
        part.validate_against_canonical(other_canonical)


@pytest.mark.unit
def test_partition_stratified_distribution(
    canonical_universe: CanonicalSampleManifest,
) -> None:
    """Verify stratified partitioning preserves balanced class distributions."""
    part = generate_partition_manifest(
        canonical_manifest=canonical_universe,
        split_ratios={"train": 0.8, "val": 0.2},
        seed=42,
        strategy="stratified",
        source_split_filter="train",
    )

    train_dist = part.get_split("train").class_distribution
    val_dist = part.get_split("val").class_distribution

    # 40 train samples = 20 of class 0, 20 of class 1
    assert train_dist == {"0": 20, "1": 20}
    # 10 val samples = 5 of class 0, 5 of class 1
    assert val_dist == {"0": 5, "1": 5}


@pytest.mark.unit
def test_partition_rejects_invalid_ratios(
    canonical_universe: CanonicalSampleManifest,
) -> None:
    """Verify generator rejects ratios not summing to 1.0."""
    with pytest.raises(ValidationError, match=r"Split ratios must sum to 1\.0"):
        generate_partition_manifest(
            canonical_manifest=canonical_universe,
            split_ratios={"train": 0.5, "val": 0.2},  # sums to 0.7
            seed=42,
        )


@pytest.mark.unit
def test_partition_manifest_serialization_round_trip(
    canonical_universe: CanonicalSampleManifest,
) -> None:
    """Verify PartitionManifest serializes and deserializes cleanly."""
    part = generate_partition_manifest(
        canonical_manifest=canonical_universe,
        split_ratios={"train": 0.8, "val": 0.2},
        seed=42,
        strategy="stratified",
        source_split_filter="train",
        isolated_splits={"test": "test"},
    )

    dumped_dict = part.to_dict()
    restored_dict = PartitionManifest.from_dict(dumped_dict)
    assert part.compute_fingerprint() == restored_dict.compute_fingerprint()

    json_str = part.to_json()
    restored_json = PartitionManifest.from_json(json_str)
    assert part.compute_fingerprint() == restored_json.compute_fingerprint()
