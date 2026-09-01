"""Transfer learning specifications, strategy enums, and configuration contracts."""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from prism.training.configuration import OptimizerSpecification, SchedulerSpecification


class TransferStrategy(str, Enum):
    """Core strategies for representation reuse and transfer learning."""

    LINEAR_PROBE = "linear_probe"
    PARTIAL_FINE_TUNE = "partial_fine_tune"
    FULL_FINE_TUNE = "full_fine_tune"
    SCRATCH_BASELINE = "scratch_baseline"


class NormalizationTransferPolicy(str, Enum):
    """Policy for handling normalization running statistics during transfer."""

    FREEZE_SOURCE_STATS = "freeze_source_stats"
    ADAPT_RUNNING_STATS = "adapt_running_stats"


class TransferLearningSpecification(BaseModel):
    """Declarative specification for a transfer learning experiment.

    Defines the exact source model state, target dataset, transfer strategy,
    parameter freezing policies, and training configuration.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    transfer_id: str = Field(
        ..., description="Unique semantic identifier for this transfer configuration"
    )
    source_experiment_id: str = Field(
        ..., description="Experiment ID that produced the source model state"
    )
    source_model_id: str = Field(
        ..., description="Model ID of the source trained architecture"
    )
    source_dataset_id: str = Field(
        ..., description="Source dataset universe/partition identifier"
    )
    source_task: str = Field(
        default="classification", description="Source training objective/task"
    )
    target_dataset_id: str = Field(
        ..., description="Target dataset universe/partition identifier"
    )
    target_task: str = Field(
        default="classification", description="Target training objective/task"
    )
    target_num_classes: int = Field(
        ...,
        ge=2,
        description="Number of target classes for the new classification head",
    )
    strategy: TransferStrategy = Field(
        default=TransferStrategy.LINEAR_PROBE,
        description="Transfer strategy (linear, partial, full, or scratch)",
    )
    normalization_policy: NormalizationTransferPolicy = Field(
        default=NormalizationTransferPolicy.FREEZE_SOURCE_STATS,
        description="Normalization running statistics policy during transfer",
    )
    representation_layer: str = Field(
        default="final_hidden",
        description="Logical representation layer extracted for transfer analysis",
    )
    frozen_prefixes: list[str] = Field(
        default_factory=list,
        description="Parameter name prefixes explicitly frozen during training",
    )
    trainable_prefixes: list[str] = Field(
        default_factory=list,
        description="Parameter name prefixes explicitly trainable during training",
    )
    target_data_budget: float = Field(
        default=1.0,
        ge=0.01,
        le=1.0,
        description="Fraction of target dataset partition used (0.01 to 1.0)",
    )
    target_optimizer: OptimizerSpecification = Field(
        default_factory=lambda: OptimizerSpecification(type="sgd", lr=0.01),
        description="Optimizer specification for target training",
    )
    target_scheduler: SchedulerSpecification = Field(
        default_factory=lambda: SchedulerSpecification(type="none"),
        description="Learning rate scheduler specification for target training",
    )
    target_epochs: int = Field(
        default=10, ge=1, description="Number of target training epochs"
    )
    seed: int = Field(
        default=42, description="RNG seed for target initialization and training"
    )

    def fingerprint(self) -> str:
        """Compute deterministic SHA-256 digest of transfer specification."""
        payload: dict[str, Any] = {
            "transfer_id": self.transfer_id,
            "source_experiment_id": self.source_experiment_id,
            "source_model_id": self.source_model_id,
            "source_dataset_id": self.source_dataset_id,
            "source_task": self.source_task,
            "target_dataset_id": self.target_dataset_id,
            "target_task": self.target_task,
            "target_num_classes": self.target_num_classes,
            "strategy": self.strategy.value,
            "normalization_policy": self.normalization_policy.value,
            "representation_layer": self.representation_layer,
            "frozen_prefixes": sorted(self.frozen_prefixes),
            "trainable_prefixes": sorted(self.trainable_prefixes),
            "target_data_budget": float(self.target_data_budget),
            "target_optimizer": self.target_optimizer.model_dump(mode="json"),
            "target_scheduler": self.target_scheduler.model_dump(mode="json"),
            "target_epochs": self.target_epochs,
            "seed": self.seed,
        }
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
