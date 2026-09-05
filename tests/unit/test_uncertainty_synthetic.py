"""Unit tests for synthetic in-distribution and OOD dataset generation."""

from __future__ import annotations

from prism.uncertainty.contracts import OODSample
from prism.uncertainty.enums import OODCategory
from prism.uncertainty.synthetic import (
    SyntheticOODSpec,
    generate_synthetic_ood_dataset,
)


def test_synthetic_ood_dataset_generation() -> None:
    """Verify deterministic generation of ID, near-OOD, and far-OOD image samples."""
    spec = SyntheticOODSpec(
        dataset_name="test-ood-v1",
        num_samples=30,
        image_shape=(3, 16, 16),
        seed=42,
    )

    samples, meta = generate_synthetic_ood_dataset(spec)

    assert isinstance(samples, list)
    assert len(samples) == 30
    assert isinstance(meta, dict)
    assert "ood_dataset_fingerprint" in meta
    assert meta["num_samples"] == 30

    for s in samples:
        assert isinstance(s, OODSample)
        assert s.category in [
            OODCategory.OUT_OF_DISTRIBUTION,
            OODCategory.NEAR_OOD,
            OODCategory.CORRUPTED_IN_DISTRIBUTION,
        ]
        assert len(s.image) == 3
        assert len(s.image[0]) == 16
        assert len(s.image[0][0]) == 16

    # Determinism with same seed
    samples2, meta2 = generate_synthetic_ood_dataset(spec)
    assert meta["ood_dataset_fingerprint"] == meta2["ood_dataset_fingerprint"]
    assert samples[0].sample_id == samples2[0].sample_id
