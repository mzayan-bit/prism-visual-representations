"""Unit tests for SampleRecord and CanonicalSampleManifest."""

import pydantic
import pytest

from prism.core.errors import ValidationError
from prism.data.samples import CanonicalSampleManifest, SampleRecord


@pytest.fixture
def sample_records() -> list[SampleRecord]:
    return [
        SampleRecord(
            sample_id=f"ds-synth/train/{i:04d}",
            source_split="train",
            source_index=i,
            target=i % 2,
        )
        for i in range(10)
    ]


@pytest.mark.unit
def test_sample_record_valid(sample_records: list[SampleRecord]) -> None:
    """Verify SampleRecord creation and immutability."""
    rec = sample_records[0]
    assert rec.sample_id == "ds-synth/train/0000"
    assert rec.source_split == "train"
    assert rec.source_index == 0
    assert rec.target == 0

    with pytest.raises(pydantic.ValidationError):
        rec.__setattr__("sample_id", "new_id")


@pytest.mark.unit
def test_canonical_manifest_creation_and_fingerprint(
    sample_records: list[SampleRecord],
) -> None:
    """Verify CanonicalSampleManifest creation and deterministic fingerprinting."""
    manifest1 = CanonicalSampleManifest.create(
        dataset_id="ds-synthetic",
        samples=sample_records,
        dataset_version="1.0.0",
    )

    manifest2 = CanonicalSampleManifest.create(
        dataset_id="ds-synthetic",
        samples=sample_records,
        dataset_version="1.0.0",
    )

    assert manifest1.num_samples == 10
    assert manifest1.class_distribution == {"0": 5, "1": 5}
    assert manifest1.compute_fingerprint() == manifest2.compute_fingerprint()


@pytest.mark.unit
def test_canonical_manifest_rejects_duplicate_sample_ids() -> None:
    """Verify manifest rejects duplicate sample IDs."""
    duplicate_records = [
        SampleRecord(
            sample_id="ds-synth/train/0001",
            source_split="train",
            source_index=0,
        ),
        SampleRecord(
            sample_id="ds-synth/train/0001",  # duplicate ID
            source_split="train",
            source_index=1,
        ),
    ]

    with pytest.raises(ValidationError, match="Duplicate sample_id"):
        CanonicalSampleManifest(
            dataset_id="ds-synth",
            samples=duplicate_records,
            num_samples=2,
        )


@pytest.mark.unit
def test_canonical_manifest_rejects_duplicate_source_coordinates() -> None:
    """Verify manifest rejects duplicate (source_split, source_index) coordinates."""
    duplicate_coords = [
        SampleRecord(
            sample_id="ds-synth/train/0001",
            source_split="train",
            source_index=0,
        ),
        SampleRecord(
            sample_id="ds-synth/train/0002",
            source_split="train",
            source_index=0,  # duplicate coordinate
        ),
    ]

    with pytest.raises(ValidationError, match="Duplicate source coordinate"):
        CanonicalSampleManifest(
            dataset_id="ds-synth",
            samples=duplicate_coords,
            num_samples=2,
        )


@pytest.mark.unit
def test_canonical_manifest_rejects_count_mismatch(
    sample_records: list[SampleRecord],
) -> None:
    """Verify manifest rejects declared num_samples mismatch."""
    with pytest.raises(ValidationError, match="Declared num_samples"):
        CanonicalSampleManifest(
            dataset_id="ds-synth",
            samples=sample_records,
            num_samples=999,
        )


@pytest.mark.unit
def test_canonical_manifest_fingerprint_sensitivity(
    sample_records: list[SampleRecord],
) -> None:
    """Verify fingerprint changes when samples or order change."""
    manifest_base = CanonicalSampleManifest.create(
        dataset_id="ds-synthetic",
        samples=sample_records,
    )

    # Reordered samples
    manifest_reordered = CanonicalSampleManifest.create(
        dataset_id="ds-synthetic",
        samples=list(reversed(sample_records)),
    )
    assert (
        manifest_base.compute_fingerprint() != manifest_reordered.compute_fingerprint()
    )

    # Modified target
    modified_records = list(sample_records)
    modified_records[0] = SampleRecord(
        sample_id=sample_records[0].sample_id,
        source_split=sample_records[0].source_split,
        source_index=sample_records[0].source_index,
        target=99,  # modified target
    )
    manifest_modified = CanonicalSampleManifest.create(
        dataset_id="ds-synthetic",
        samples=modified_records,
    )
    assert (
        manifest_base.compute_fingerprint() != manifest_modified.compute_fingerprint()
    )


@pytest.mark.unit
def test_canonical_manifest_serialization_round_trip(
    sample_records: list[SampleRecord],
) -> None:
    """Verify JSON and dict serialization round trips preserve fingerprint."""
    manifest = CanonicalSampleManifest.create(
        dataset_id="ds-synthetic",
        samples=sample_records,
    )

    # Dict round-trip
    dumped_dict = manifest.to_dict()
    restored_dict = CanonicalSampleManifest.from_dict(dumped_dict)
    assert manifest.compute_fingerprint() == restored_dict.compute_fingerprint()

    # JSON round-trip
    json_str = manifest.to_json()
    restored_json = CanonicalSampleManifest.from_json(json_str)
    assert manifest.compute_fingerprint() == restored_json.compute_fingerprint()


@pytest.mark.unit
def test_canonical_manifest_lookups(
    sample_records: list[SampleRecord],
) -> None:
    """Verify get_sample and filter_by_source_split utilities."""
    manifest = CanonicalSampleManifest.create(
        dataset_id="ds-synthetic",
        samples=sample_records,
    )

    s = manifest.get_sample("ds-synth/train/0000")
    assert s.source_index == 0
    assert s.target == 0

    with pytest.raises(KeyError):
        manifest.get_sample("nonexistent_id")

    train_samples = manifest.filter_by_source_split("train")
    assert len(train_samples) == 10

    test_samples = manifest.filter_by_source_split("test")
    assert len(test_samples) == 0
