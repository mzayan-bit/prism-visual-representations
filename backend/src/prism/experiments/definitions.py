"""Declarative experiment definitions and scientific specifications."""

from datetime import datetime, timezone
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from prism.core.enums import TaskType
from prism.core.errors import SerializationError, ValidationError
from prism.core.identifiers import ensure_valid_identifier
from prism.data.manifests import DatasetManifest
from prism.evaluation.configuration import EvaluationConfiguration
from prism.experiments.hashing import compute_configuration_fingerprint
from prism.experiments.reproducibility import ReproducibilityConfiguration
from prism.models.specifications import ModelSpecification
from prism.training.configuration import TrainingConfiguration


def _utc_now() -> datetime:
    """Return timezone-aware current UTC datetime."""
    return datetime.now(timezone.utc)


class ExperimentDefinition(BaseModel):
    """Immutable, declarative specification of an experiment in PRISM."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_id: str = Field(
        description="Unique experiment identifier (e.g. 'exp-cifar10-resnet18')"
    )
    name: str = Field(description="Human-readable title of the experiment")
    description: str = Field(
        default="",
        description="Detailed narrative description of the experiment",
    )
    task_type: TaskType = Field(
        description="Primary machine learning task paradigm being evaluated"
    )
    hypothesis: str = Field(
        default="",
        description="Scientific question or hypothesis this experiment tests",
    )
    dataset: DatasetManifest = Field(
        description="Dataset manifest and partitioning specification"
    )
    model: ModelSpecification = Field(description="Model architecture specification")
    training: TrainingConfiguration = Field(
        description="Optimization and training configuration"
    )
    evaluation: EvaluationConfiguration = Field(
        description="Evaluation protocols and target metrics"
    )
    reproducibility: ReproducibilityConfiguration = Field(
        default_factory=ReproducibilityConfiguration,
        description="Deterministic seeding and provenance audit settings",
    )
    tags: list[str] = Field(default_factory=list, description="Categorization tags")
    created_at: datetime = Field(
        default_factory=_utc_now, description="UTC creation timestamp"
    )
    created_by: str | None = Field(
        default=None,
        description="Author or system responsible for definition",
    )
    schema_version: str = Field(default="1.0.0", description="Contract schema version")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary domain metadata"
    )

    @field_validator("experiment_id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        return ensure_valid_identifier(v, field_name="experiment_id")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Experiment name cannot be empty.")
        return v.strip()

    @model_validator(mode="after")
    def validate_task_compatibility(self) -> "ExperimentDefinition":
        """Validate that dataset, model, and experiment agree on task paradigm."""
        if self.task_type not in self.dataset.compatible_tasks:
            raise ValidationError(
                f"Dataset '{self.dataset.dataset_id}' does not support task "
                f"'{self.task_type}'. Compatible: {self.dataset.compatible_tasks}"
            )

        if self.task_type not in self.model.compatible_tasks:
            raise ValidationError(
                f"Model '{self.model.model_id}' does not support task "
                f"'{self.task_type}'. Compatible: {self.model.compatible_tasks}"
            )

        # In classification experiments, ensure class counts match if both specified
        if (
            self.task_type == TaskType.CLASSIFICATION
            and self.dataset.num_classes is not None
            and self.model.num_classes is not None
            and self.dataset.num_classes != self.model.num_classes
        ):
            raise ValidationError(
                f"Class count mismatch: Dataset has {self.dataset.num_classes}, "
                f"Model has {self.model.num_classes} output logits."
            )

        return self

    def compute_fingerprint(self) -> str:
        """Compute the deterministic SHA-256 semantic configuration fingerprint."""
        return compute_configuration_fingerprint(self)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the experiment definition to a JSON-compatible dictionary."""
        return self.model_dump(mode="json")

    def to_json(self, indent: int | None = None) -> str:
        """Serialize the experiment definition to a JSON string."""
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentDefinition":
        """Deserialize an experiment definition from a dictionary."""
        try:
            return cls.model_validate(data)
        except Exception as exc:
            raise SerializationError(
                f"Failed to deserialize ExperimentDefinition from dict: {exc}"
            ) from exc

    @classmethod
    def from_json(cls, json_str: str) -> "ExperimentDefinition":
        """Deserialize an experiment definition from a JSON string."""
        try:
            return cls.model_validate_json(json_str)
        except Exception as exc:
            raise SerializationError(
                f"Failed to deserialize ExperimentDefinition from JSON: {exc}"
            ) from exc
