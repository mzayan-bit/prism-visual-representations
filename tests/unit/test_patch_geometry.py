"""Unit tests for PatchGeometry abstraction, divisibility checks, and serialization."""

import pytest

from prism.core.errors import SerializationError, ValidationError
from prism.models.patches import PatchGeometry


@pytest.mark.unit
def test_patch_geometry_creation_and_derived_dimensions() -> None:
    """Verify PatchGeometry calculates correct grid counts and flattened dimension."""
    # 32x32 image with 4x4 patches, 3 channels
    geom = PatchGeometry.create(image_size=(32, 32), patch_size=(4, 4), channels=3)

    assert geom.image_height == 32
    assert geom.image_width == 32
    assert geom.channels == 3
    assert geom.patch_height == 4
    assert geom.patch_width == 4
    assert geom.patches_per_row == 8
    assert geom.patches_per_column == 8
    assert geom.total_patches == 64
    assert geom.flattened_patch_dimension == 48  # 3 * 4 * 4


@pytest.mark.unit
def test_patch_geometry_rectangular_support() -> None:
    """Verify PatchGeometry handles rectangular images and non-square patches."""
    # 24x32 image with 6x8 patches, 1 channel
    geom = PatchGeometry.create(image_size=(24, 32), patch_size=(6, 8), channels=1)

    assert geom.patches_per_column == 4  # 24 / 6
    assert geom.patches_per_row == 4  # 32 / 8
    assert geom.total_patches == 16
    assert geom.flattened_patch_dimension == 48  # 1 * 6 * 8


@pytest.mark.unit
def test_patch_geometry_validation_rejections() -> None:
    """Verify PatchGeometry rejects non-divisible dimensions and non-positive values."""
    # Image height 30 not divisible by patch height 4
    with pytest.raises(ValidationError, match="not divisible by patch height"):
        PatchGeometry.create(image_size=(30, 32), patch_size=(4, 4))

    # Image width 30 not divisible by patch width 4
    with pytest.raises(ValidationError, match="not divisible by patch width"):
        PatchGeometry.create(image_size=(32, 30), patch_size=(4, 4))

    # Non-positive channel count
    with pytest.raises(ValidationError, match="channels must be positive"):
        PatchGeometry.create(image_size=32, patch_size=4, channels=0)

    # Inconsistent derived fields passed directly
    with pytest.raises(ValidationError, match="total_patches mismatch"):
        PatchGeometry(
            image_height=32,
            image_width=32,
            channels=3,
            patch_height=4,
            patch_width=4,
            patches_per_row=8,
            patches_per_column=8,
            total_patches=99,  # Incorrect
            flattened_patch_dimension=48,
        )


@pytest.mark.unit
def test_patch_geometry_serialization_roundtrip() -> None:
    """Verify PatchGeometry dictionary and JSON serialization roundtrips."""
    geom = PatchGeometry.create(image_size=64, patch_size=8, channels=3)

    # Dictionary roundtrip
    geom_dict = geom.to_dict()
    restored_dict = PatchGeometry.from_dict(geom_dict)
    assert restored_dict == geom

    # JSON roundtrip
    json_str = geom.to_json(indent=2)
    restored_json = PatchGeometry.from_json(json_str)
    assert restored_json == geom

    # Rejection of invalid JSON
    with pytest.raises(SerializationError, match="Invalid JSON string"):
        PatchGeometry.from_json("invalid json {")
