"""Unit tests for image corruption operators, specifications, and dataset views."""

import math

import pytest

from prism.data.materialized import MaterializedDataset, MaterializedSample
from prism.robustness.corruptions import (
    CorruptedDatasetView,
    CorruptionSpecification,
    CorruptionType,
    apply_brightness_shift,
    apply_contrast_shift,
    apply_corruption,
    apply_gaussian_noise,
    apply_rectangular_occlusion,
    apply_resolution_degradation,
    apply_spatial_blur,
)


def _make_dummy_image(c: int = 3, h: int = 8, w: int = 8) -> list[list[list[float]]]:
    """Create a deterministic synthetic image in [0, 1]."""
    img: list[list[list[float]]] = []
    for ch in range(c):
        plane: list[list[float]] = []
        for r in range(h):
            row: list[float] = []
            for col in range(w):
                val = 0.5 + 0.3 * math.sin(float(ch * 10 + r * 2 + col))
                row.append(max(0.0, min(1.0, val)))
            plane.append(row)
        img.append(plane)
    return img


def test_gaussian_noise_operator() -> None:
    img = _make_dummy_image()
    corrupted_1 = apply_gaussian_noise(img, sigma=0.1, seed=42)
    corrupted_2 = apply_gaussian_noise(img, sigma=0.1, seed=42)

    # Shape preservation
    assert len(corrupted_1) == 3
    assert len(corrupted_1[0]) == 8
    assert len(corrupted_1[0][0]) == 8

    # Determinism
    assert corrupted_1 == corrupted_2

    # Clamping range [0, 1]
    for ch in corrupted_1:
        for row in ch:
            for val in row:
                assert 0.0 <= val <= 1.0


def test_spatial_blur_operator() -> None:
    img = _make_dummy_image()
    corrupted = apply_spatial_blur(img, kernel_size=3, sigma=1.0)
    assert len(corrupted) == 3
    assert len(corrupted[0]) == 8
    assert len(corrupted[0][0]) == 8
    for ch in corrupted:
        for row in ch:
            for val in row:
                assert 0.0 <= val <= 1.0


def test_brightness_shift_operator() -> None:
    img = _make_dummy_image()
    corrupted = apply_brightness_shift(img, delta=0.2)
    assert len(corrupted) == 3
    assert corrupted[0][0][0] >= img[0][0][0]
    for ch in corrupted:
        for row in ch:
            for val in row:
                assert 0.0 <= val <= 1.0


def test_contrast_shift_operator() -> None:
    img = _make_dummy_image()
    corrupted = apply_contrast_shift(img, factor=0.5)
    assert len(corrupted) == 3
    for ch in corrupted:
        for row in ch:
            for val in row:
                assert 0.0 <= val <= 1.0


def test_rectangular_occlusion_operator() -> None:
    img = _make_dummy_image()
    corrupted = apply_rectangular_occlusion(
        img, area_ratio=0.25, fill_value=0.0, location="center"
    )
    assert len(corrupted) == 3


def test_resolution_degradation_operator() -> None:
    img = _make_dummy_image()
    corrupted = apply_resolution_degradation(img, downsample_factor=2)
    assert len(corrupted) == 3
    assert len(corrupted[0]) == 8
    assert len(corrupted[0][0]) == 8


def test_apply_corruption_generic() -> None:
    img = _make_dummy_image()
    for c_type in CorruptionType:
        for sev in [1, 3, 5]:
            spec = CorruptionSpecification(
                corruption_type=c_type, severity=sev, seed=123
            )
            corrupted = apply_corruption(img, spec)
            assert len(corrupted) == 3
            assert len(corrupted[0]) == 8
            assert len(corrupted[0][0]) == 8


def test_corruption_spec_validation() -> None:
    from pydantic import ValidationError as PydanticValidationError

    with pytest.raises(PydanticValidationError):
        CorruptionSpecification(
            corruption_type=CorruptionType.GAUSSIAN_NOISE, severity=0
        )

    with pytest.raises(PydanticValidationError):
        CorruptionSpecification(
            corruption_type=CorruptionType.GAUSSIAN_NOISE, severity=6
        )

    spec = CorruptionSpecification(corruption_type=CorruptionType.BLUR, severity=3)
    fp1 = spec.fingerprint()
    fp2 = spec.fingerprint()
    assert fp1 == fp2
    assert len(fp1) == 64  # SHA-256 hex digest


def test_corrupted_dataset_view() -> None:
    samples = [
        MaterializedSample(
            sample_id=f"sample_{i}",
            source_split="test",
            source_index=i,
            data=_make_dummy_image(),
            target=i % 2,
        )
        for i in range(4)
    ]
    base_ds = MaterializedDataset(
        dataset_id="ds_base", samples=samples, split_name="test"
    )

    spec = CorruptionSpecification(
        corruption_type=CorruptionType.GAUSSIAN_NOISE, severity=3, seed=42
    )
    view = CorruptedDatasetView(base_dataset=base_ds, corruption_spec=spec)

    assert len(view) == 4
    assert view.sample_ids == base_ds.sample_ids
    assert view.targets == base_ds.targets

    # Check non-destructive behavior on base_dataset
    clean_sample = base_ds[0]
    corrupted_sample = view[0]
    assert clean_sample.sample_id == corrupted_sample.sample_id
    assert clean_sample.target == corrupted_sample.target
    assert clean_sample.data != corrupted_sample.data
