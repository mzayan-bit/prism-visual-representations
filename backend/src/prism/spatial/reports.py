"""Structured reports and summary contracts for spatial representation transfer."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from prism.spatial.enums import (
    PretrainingObjective,
    SpatialTaskType,
)
from prism.spatial.metrics import (
    DetectionEvaluationResult,
    SegmentationMetricsResult,
)
from prism.spatial.specification import SpatialTransferSpecification


class SpatialTransferReport(BaseModel):
    """Immutable report capturing downstream spatial performance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    report_id: str = Field(..., description="Unique report identifier")
    specification: SpatialTransferSpecification = Field(
        ..., description="Complete run specification"
    )
    total_parameters: int = Field(
        ..., description="Total parameters in composite model"
    )
    frozen_parameters: int = Field(
        ..., description="Frozen parameters in composite model"
    )
    trainable_parameters: int = Field(
        ..., description="Trainable parameters in composite model"
    )
    head_parameters: int = Field(..., description="Parameters inside spatial task head")
    trainable_fraction: float = Field(
        ..., ge=0.0, le=1.0, description="Fraction of trainable parameters"
    )
    feature_shape: tuple[int, int, int] = Field(
        ..., description="Extracted feature shape (C_f, H_f, W_f)"
    )
    feature_resolution: str = Field(
        ..., description="Human-readable spatial resolution, e.g. '16x16'"
    )
    training_loss_trajectory: list[float] = Field(
        default_factory=list, description="Epoch-by-epoch loss trajectory"
    )
    epochs_completed: int = Field(..., description="Total completed epochs")
    detection_metrics: DetectionEvaluationResult | None = Field(
        default=None, description="Evaluation metrics for object detection"
    )
    segmentation_metrics: SegmentationMetricsResult | None = Field(
        default=None, description="Evaluation metrics for semantic segmentation"
    )
    spatial_representation_drift_cosine: float = Field(
        default=0.0,
        description="Cosine distance drift of representations before vs after transfer",
    )
    spatial_representation_drift_rmse: float = Field(
        default=0.0,
        description="RMSE drift of representations before vs after transfer",
    )
    warnings: list[str] = Field(
        default_factory=list, description="Diagnostic alerts or limitations"
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        return self.model_dump(mode="json")

    def to_json(self, indent: int | None = None) -> str:
        """Serialize report to JSON."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


class SpatialObjectiveComparisonSummary(BaseModel):
    """Comparison across Supervised, SimCLR, Reconstruction, and Scratch."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    comparison_id: str = Field(..., description="Identifier for comparison")
    architecture: str = Field(..., description="Model architecture under evaluation")
    task_type: SpatialTaskType = Field(..., description="Downstream spatial task")
    reports_by_objective: dict[str, SpatialTransferReport] = Field(
        ..., description="Mapping from pretraining objective to report"
    )
    metric_name: str = Field(
        default="mean_iou", description="Primary comparison metric"
    )


class SpatialLayerTransferabilitySummary(BaseModel):
    """Study of downstream spatial performance across encoder depth."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    architecture: str = Field(..., description="Encoder architecture")
    task_type: SpatialTaskType = Field(..., description="Target spatial task")
    source_objective: PretrainingObjective = Field(
        ..., description="Pretraining objective"
    )
    layer_evaluations: list[dict[str, Any]] = Field(
        ..., description="List of per-layer evaluation records"
    )


class SpatialDataEfficiencySummary(BaseModel):
    """Downstream spatial transfer performance scaling with annotation budget."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    architecture: str = Field(..., description="Encoder architecture")
    task_type: SpatialTaskType = Field(..., description="Target spatial task")
    curves_by_objective: dict[str, list[dict[str, float]]] = Field(
        ..., description="Mapping from objective to budget performance points"
    )
