"""Spatial transfer configuration specifications and parameter contracts."""

from __future__ import annotations

import hashlib

from pydantic import BaseModel, ConfigDict, Field

from prism.models.specifications import ModelSpecification
from prism.spatial.enums import (
    PretrainingObjective,
    SegmentationResizePolicy,
    SpatialTaskType,
    SpatialTransferStrategy,
)


class SpatialTransferSpecification(BaseModel):
    """Immutable specification declaring a spatial transfer study."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    specification_id: str = Field(
        ..., description="Unique deterministic identifier for the transfer run"
    )
    source_objective: PretrainingObjective = Field(
        ..., description="Source pretraining objective defining representation origins"
    )
    source_experiment_id: str = Field(
        ..., description="Identifier of the source pretrained model/experiment"
    )
    model_spec: ModelSpecification = Field(
        ..., description="Encoder architecture specification"
    )
    task_type: SpatialTaskType = Field(
        ..., description="Target spatial downstream task (detection or segmentation)"
    )
    spatial_layer: str = Field(
        default="final_spatial",
        description="Encoder layer name where spatial features are extracted",
    )
    transfer_strategy: SpatialTransferStrategy = Field(
        default=SpatialTransferStrategy.FROZEN_SPATIAL_PROBE,
        description="Parameter update strategy",
    )
    num_classes: int = Field(
        default=3, gt=0, description="Number of target semantic categories"
    )
    learning_rate: float = Field(
        default=0.01, gt=0.0, description="Optimizer step learning rate"
    )
    epochs: int = Field(default=5, gt=0, description="Number of training epochs")
    batch_size: int = Field(default=4, gt=0, description="Mini-batch size")
    data_budget_fraction: float = Field(
        default=1.0,
        gt=0.0,
        le=1.0,
        description="Fraction of training samples utilized (0.1 to 1.0)",
    )
    lambda_obj: float = Field(
        default=1.0, ge=0.0, description="Detection objectness loss weight"
    )
    lambda_cls: float = Field(
        default=1.0, ge=0.0, description="Classification loss weight"
    )
    lambda_box: float = Field(
        default=2.0, ge=0.0, description="Detection box regression loss weight"
    )
    resize_policy: SegmentationResizePolicy = Field(
        default=SegmentationResizePolicy.NEAREST,
        description="Spatial interpolation policy for segmentation upsampling",
    )
    seed: int = Field(
        default=42, description="Deterministic pseudo-random generator seed"
    )

    @classmethod
    def create(
        cls,
        source_objective: PretrainingObjective,
        source_experiment_id: str,
        model_spec: ModelSpecification,
        task_type: SpatialTaskType,
        spatial_layer: str = "final_spatial",
        transfer_strategy: SpatialTransferStrategy = (
            SpatialTransferStrategy.FROZEN_SPATIAL_PROBE
        ),
        num_classes: int = 3,
        learning_rate: float = 0.01,
        epochs: int = 5,
        batch_size: int = 4,
        data_budget_fraction: float = 1.0,
        lambda_obj: float = 1.0,
        lambda_cls: float = 1.0,
        lambda_box: float = 2.0,
        resize_policy: SegmentationResizePolicy = SegmentationResizePolicy.NEAREST,
        seed: int = 42,
    ) -> SpatialTransferSpecification:
        """Create a deterministic SpatialTransferSpecification."""
        hasher = hashlib.sha256()
        payload = (
            f"{source_objective}_{source_experiment_id}_{model_spec.model_id}_"
            f"{task_type}_{spatial_layer}_{transfer_strategy}_{num_classes}_"
            f"{data_budget_fraction}_{seed}"
        )
        hasher.update(payload.encode())
        spec_id = f"spatial_spec_{hasher.hexdigest()[:12]}"

        return cls(
            specification_id=spec_id,
            source_objective=source_objective,
            source_experiment_id=source_experiment_id,
            model_spec=model_spec,
            task_type=task_type,
            spatial_layer=spatial_layer,
            transfer_strategy=transfer_strategy,
            num_classes=num_classes,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            data_budget_fraction=data_budget_fraction,
            lambda_obj=lambda_obj,
            lambda_cls=lambda_cls,
            lambda_box=lambda_box,
            resize_policy=resize_policy,
            seed=seed,
        )
