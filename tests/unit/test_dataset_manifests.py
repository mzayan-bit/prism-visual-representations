"""Unit tests for DatasetManifest and data policy specifications."""

import pytest
from pydantic import ValidationError as PydanticValidationError

from prism.core.enums import TaskType
from prism.core.errors import ValidationError
from prism.data.manifests import (
    AugmentationPolicy,
    DatasetManifest,
    PreprocessingPolicy,
    SplitSpecification,
)


@pytest.fixture
def valid_splits() -> list[SplitSpecification]:
    return [
        SplitSpecification(split_name="train", num_samples=50000),
        SplitSpecification(split_name="test", num_samples=10000),
    ]


@pytest.mark.unit
def test_valid_dataset_manifest(valid_splits: list[SplitSpecification]) -> None:
    """Verify construction of a valid dataset manifest."""
    manifest = DatasetManifest(
        dataset_id="ds-cifar10",
        name="CIFAR-10",
        description="Standard 10-class vision dataset",
        version="1.0.0",
        compatible_tasks=[TaskType.CLASSIFICATION],
        splits=valid_splits,
        classes=["airplane", "automobile", "bird", "cat", "deer"],
        num_classes=5,
        preprocessing=PreprocessingPolicy(
            resize=(32, 32),
            normalization_mean=(0.4914, 0.4822, 0.4465),
            normalization_std=(0.2023, 0.1994, 0.2010),
        ),
        augmentation=AugmentationPolicy(
            name="random_crop_flip",
            enabled=True,
            params={"crop_padding": 4, "horizontal_flip": True},
        ),
    )

    assert manifest.dataset_id == "ds-cifar10"
    assert manifest.num_classes == 5
    assert len(manifest.splits) == 2
    assert manifest.preprocessing.resize == (32, 32)
    assert manifest.augmentation.enabled is True


@pytest.mark.unit
def test_empty_splits_rejected() -> None:
    """Verify that a manifest without splits raises a validation error."""
    with pytest.raises((PydanticValidationError, ValueError)):
        DatasetManifest(
            dataset_id="ds-empty",
            name="Empty Dataset",
            splits=[],
        )


@pytest.mark.unit
def test_duplicate_split_names_rejected() -> None:
    """Verify that duplicate split partition names are rejected."""
    splits = [
        SplitSpecification(split_name="train", num_samples=100),
        SplitSpecification(split_name="train", num_samples=200),
    ]
    with pytest.raises((PydanticValidationError, ValueError)):
        DatasetManifest(
            dataset_id="ds-dup",
            name="Duplicate Split Dataset",
            splits=splits,
        )


@pytest.mark.unit
def test_mismatched_class_count_rejected(
    valid_splits: list[SplitSpecification],
) -> None:
    """Verify that mismatched class list length and num_classes is rejected."""
    with pytest.raises(ValidationError, match="Length of classes"):
        DatasetManifest(
            dataset_id="ds-class-mismatch",
            name="Class Mismatch Dataset",
            splits=valid_splits,
            classes=["cat", "dog"],
            num_classes=10,
        )


@pytest.mark.unit
def test_invalid_normalization_std() -> None:
    """Verify that zero or negative normalization std raises an error."""
    with pytest.raises(ValueError, match="must be positive"):
        PreprocessingPolicy(
            normalization_mean=(0.5, 0.5, 0.5),
            normalization_std=(0.5, 0.0, 0.5),  # zero std is invalid
        )


@pytest.mark.unit
def test_invalid_subset_fraction(
    valid_splits: list[SplitSpecification],
) -> None:
    """Verify that subset_fraction outside (0, 1] is rejected."""
    with pytest.raises((PydanticValidationError, ValueError)):
        DatasetManifest(
            dataset_id="ds-invalid-subset",
            name="Invalid Subset Dataset",
            splits=valid_splits,
            subset_fraction=1.5,  # > 1.0 is invalid
        )
