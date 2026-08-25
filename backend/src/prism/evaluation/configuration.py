"""Evaluation configurations and metric evaluation specifications."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from prism.core.enums import MetricDirection


class MetricSpecification(BaseModel):
    """Specification of an evaluation metric to collect during evaluation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(
        description="Metric name (e.g. 'top1_accuracy', 'f1_macro', 'loss')"
    )
    direction: MetricDirection = Field(
        default=MetricDirection.MAXIMIZE,
        description="Optimization or evaluation direction",
    )
    target_split: str = Field(
        default="test",
        description="Target dataset split on which to evaluate this metric",
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metric parameters (e.g. top_k, average, threshold)",
    )

    @field_validator("name", "target_split")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty.")
        return v.strip().lower()


class EvaluationConfiguration(BaseModel):
    """Configuration governing evaluation protocols, target splits, and metrics."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    target_splits: list[str] = Field(
        default_factory=lambda: ["test"],
        description="Dataset splits to evaluate (e.g. ['val', 'test', 'ood'])",
    )
    metrics: list[MetricSpecification] = Field(
        description="Evaluation metrics to calculate across target splits",
    )
    batch_size: int = Field(
        default=64,
        gt=0,
        description="Batch size for evaluation inference",
    )
    save_predictions: bool = Field(
        default=False,
        description="Whether to persist raw prediction tensors as artifacts",
    )
    compute_per_class: bool = Field(
        default=False,
        description="Whether to compute and report per-class breakdown statistics",
    )
    confidence_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for selective prediction or calibration",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional evaluation options",
    )

    @field_validator("target_splits")
    @classmethod
    def validate_splits(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError(
                "Evaluation configuration must specify at least one target split."
            )
        cleaned = [s.strip().lower() for s in v if s and s.strip()]
        if not cleaned:
            raise ValueError("Target splits cannot be empty.")
        return cleaned

    @field_validator("metrics")
    @classmethod
    def validate_metrics(
        cls, v: list[MetricSpecification]
    ) -> list[MetricSpecification]:
        if not v:
            raise ValueError(
                "Evaluation configuration must specify at least one metric."
            )
        return v
