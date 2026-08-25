"""Dataset manifest and preprocessing/augmentation specifications."""

from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from prism.core.enums import TaskType
from prism.core.errors import ValidationError
from prism.core.identifiers import ensure_valid_identifier


class PreprocessingPolicy(BaseModel):
    """Declarative specification for deterministic input preprocessing."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    resize: tuple[int, int] | None = Field(
        default=None,
        description="Target (height, width) for deterministic image resize",
    )
    crop_size: tuple[int, int] | None = Field(
        default=None,
        description="Target (height, width) for center crop",
    )
    normalization_mean: tuple[float, ...] | None = Field(
        default=None,
        description="Per-channel mean values (e.g. (0.485, 0.456, 0.406))",
    )
    normalization_std: tuple[float, ...] | None = Field(
        default=None,
        description="Per-channel std values (e.g. (0.229, 0.224, 0.225))",
    )
    color_space: str = Field(
        default="RGB",
        description="Expected input color space (e.g. 'RGB', 'GRAY', 'LAB')",
    )
    extra_params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("normalization_std")
    @classmethod
    def validate_std(cls, v: tuple[float, ...] | None) -> tuple[float, ...] | None:
        if v is not None:
            for val in v:
                if val <= 0:
                    raise ValueError(
                        f"Standard deviation values must be positive, got {val}"
                    )
        return v


class AugmentationPolicy(BaseModel):
    """Declarative specification for data augmentation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(default="none", description="Augmentation scheme name")
    enabled: bool = Field(default=False, description="Whether augmentation is applied")
    params: dict[str, Any] = Field(default_factory=dict)


class SplitSpecification(BaseModel):
    """Declarative partition description for a dataset split."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    split_name: str = Field(
        description="Name of partition (e.g. 'train', 'val', 'test')"
    )
    num_samples: int | None = Field(
        default=None,
        ge=0,
        description="Declared number of samples in the split",
    )
    checksum: str | None = Field(
        default=None,
        description="SHA-256 digest of sample index file or manifest",
    )
    file_pattern: str | None = Field(
        default=None,
        description="Glob or path pattern for samples within this split",
    )

    @field_validator("split_name")
    @classmethod
    def validate_split_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Split name must not be empty.")
        return v.strip().lower()


class DatasetManifest(BaseModel):
    """Declarative dataset contract without loading tensors into memory."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_id: str = Field(description="Unique dataset identifier (e.g. 'ds-cifar10')")
    name: str = Field(description="Human-readable dataset display name")
    description: str = Field(default="", description="Detailed dataset description")
    source_uri: str | None = Field(
        default=None,
        description="Canonical source URL, HF dataset ID, or local root path",
    )
    version: str = Field(
        default="1.0.0",
        description="Declared semantic version of the dataset",
    )
    compatible_tasks: list[TaskType] = Field(
        default_factory=lambda: [TaskType.CLASSIFICATION],
        description="Research task paradigms supported by this dataset",
    )
    splits: list[SplitSpecification] = Field(
        description="Declared splits making up this dataset",
    )
    classes: list[str] | None = Field(
        default=None,
        description="Optional ordered list of category class names",
    )
    num_classes: int | None = Field(
        default=None,
        ge=1,
        description="Total number of discrete classes if applicable",
    )
    preprocessing: PreprocessingPolicy = Field(
        default_factory=PreprocessingPolicy,
        description="Deterministic preprocessing policy",
    )
    augmentation: AugmentationPolicy = Field(
        default_factory=AugmentationPolicy,
        description="Augmentation policy applied during training",
    )
    subset_fraction: float | None = Field(
        default=None,
        gt=0.0,
        le=1.0,
        description="Fraction of dataset used for low-data experiments",
    )
    fingerprint: str | None = Field(
        default=None,
        description="Cryptographic fingerprint of the dataset manifest",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary domain-specific metadata",
    )

    @field_validator("dataset_id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        return ensure_valid_identifier(v, field_name="dataset_id")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Dataset name cannot be empty.")
        return v.strip()

    @field_validator("splits")
    @classmethod
    def validate_splits(cls, v: list[SplitSpecification]) -> list[SplitSpecification]:
        if not v:
            raise ValueError("Dataset manifest must define at least one split.")
        split_names = [s.split_name for s in v]
        if len(split_names) != len(set(split_names)):
            raise ValueError(
                f"Duplicate split names found in dataset manifest: {split_names}"
            )
        return v

    @field_validator("compatible_tasks")
    @classmethod
    def validate_tasks(cls, v: list[TaskType]) -> list[TaskType]:
        if not v:
            raise ValueError(
                "Dataset must declare compatibility with at least one TaskType."
            )
        return v

    @model_validator(mode="after")
    def validate_class_consistency(self) -> "DatasetManifest":
        if (
            self.classes is not None
            and self.num_classes is not None
            and len(self.classes) != self.num_classes
        ):
            raise ValidationError(
                f"Length of classes ({len(self.classes)}) does not match "
                f"num_classes ({self.num_classes})."
            )
        return self
