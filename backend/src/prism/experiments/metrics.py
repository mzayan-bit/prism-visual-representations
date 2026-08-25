"""Metric record schemas for logging quantitative evaluation telemetry."""

import math
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from prism.core.enums import MetricDirection


def _utc_now() -> datetime:
    """Return timezone-aware current UTC datetime."""
    return datetime.now(timezone.utc)


class MetricRecord(BaseModel):
    """Immutable record of an individual quantitative evaluation or training metric."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    metric_name: str = Field(
        description="Canonical metric key (e.g. 'val_top1_accuracy', 'train_loss')"
    )
    value: float = Field(description="Scalar numerical metric value")
    split: str = Field(
        default="test",
        description="Dataset split measured (e.g. 'train', 'val', 'test')",
    )
    step: int | None = Field(
        default=None,
        ge=0,
        description="Global training or evaluation step counter",
    )
    epoch: int | None = Field(
        default=None,
        ge=0,
        description="Training epoch index at metric measurement",
    )
    direction: MetricDirection = Field(
        default=MetricDirection.MAXIMIZE,
        description="Whether higher or lower is considered superior",
    )
    timestamp: datetime = Field(
        default_factory=_utc_now,
        description="UTC timestamp when the measurement was taken",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional contextual metadata (e.g. class index)",
    )

    @field_validator("metric_name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Metric name cannot be empty.")
        return v.strip().lower()

    @field_validator("value")
    @classmethod
    def validate_finite_value(cls, v: float) -> float:
        if math.isnan(v) or math.isinf(v):
            raise ValueError(f"Metric value must be finite, got {v}")
        return float(v)
