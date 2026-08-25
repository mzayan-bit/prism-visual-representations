"""Framework-neutral model specifications and architectural descriptors."""

from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from prism.core.enums import InitializationStrategy, ModelFamily, TaskType
from prism.core.errors import ValidationError
from prism.core.identifiers import ensure_valid_identifier


class ModelSpecification(BaseModel):
    """Declarative specification describing a vision model architecture."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str = Field(
        description="Unique model identifier (e.g. 'model-resnet18-scratch')"
    )
    name: str = Field(description="Human-readable model name")
    family: ModelFamily = Field(description="Architectural family classification")
    architecture: str = Field(
        description="Architecture key (e.g. 'resnet18', 'vit_tiny_patch16_224')"
    )
    compatible_tasks: list[TaskType] = Field(
        default_factory=lambda: [TaskType.CLASSIFICATION],
        description="Task types supported by this model definition",
    )
    initialization: InitializationStrategy = Field(
        default=InitializationStrategy.RANDOM,
        description="Weight initialization or pretraining source",
    )
    pretrained_source: str | None = Field(
        default=None,
        description="URI or checkpoint path if using pretrained weights",
    )
    input_shape: tuple[int, ...] = Field(
        default=(3, 224, 224),
        description="Expected tensor input shape (e.g. (C, H, W))",
    )
    num_classes: int | None = Field(
        default=None,
        ge=1,
        description="Number of output logits for classification heads",
    )
    backbone_freeze: bool = Field(
        default=False,
        description="True if backbone feature extractor remains frozen",
    )
    probe_head: str | None = Field(
        default=None,
        description="Probe head attached to frozen representations",
    )
    hyperparameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Model hyperparameters (e.g. embed_dim, depth, dropout)",
    )
    framework: str = Field(
        default="pytorch",
        description="Intended future execution framework backend",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (e.g. parameter count, FLOPs)",
    )

    @field_validator("model_id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        return ensure_valid_identifier(v, field_name="model_id")

    @field_validator("name", "architecture")
    @classmethod
    def validate_non_empty_str(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty.")
        return v.strip()

    @field_validator("compatible_tasks")
    @classmethod
    def validate_tasks(cls, v: list[TaskType]) -> list[TaskType]:
        if not v:
            raise ValueError(
                "Model must declare compatibility with at least one TaskType."
            )
        return v

    @field_validator("input_shape")
    @classmethod
    def validate_input_shape(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        if not v or any(dim <= 0 for dim in v):
            raise ValueError(f"Input shape dimensions must all be positive, got {v}")
        return v

    @model_validator(mode="after")
    def validate_initialization_consistency(self) -> "ModelSpecification":
        pretrained_strategies = (
            InitializationStrategy.PRETRAINED,
            InitializationStrategy.TRANSFER_FROZEN,
            InitializationStrategy.TRANSFER_FINETUNE,
        )
        if self.initialization in pretrained_strategies and not self.pretrained_source:
            raise ValidationError(
                f"Model initialization strategy '{self.initialization}' requires a "
                "'pretrained_source' (e.g. checkpoint path or registry identifier)."
            )
        return self
