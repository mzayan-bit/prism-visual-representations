"""Transfer learning structured experiment reports, comparisons, and suites."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from prism.core.errors import SerializationError
from prism.transfer.freezing import ParameterFreezePlan
from prism.transfer.probes import LayerTransferProbeResult
from prism.transfer.retention import TransferRepresentationDriftSummary
from prism.transfer.specification import TransferLearningSpecification, TransferStrategy


class DataEfficiencyTransferPoint(BaseModel):
    """Single empirical point in a target-data efficiency curve."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    data_budget: float = Field(
        ..., ge=0.01, le=1.0, description="Target data fraction (e.g. 0.01, 0.25)"
    )
    sample_count: int = Field(
        ..., ge=1, description="Number of training samples in this budget"
    )
    strategy: TransferStrategy = Field(..., description="Transfer strategy evaluated")
    val_accuracy: float = Field(
        ..., ge=0.0, le=1.0, description="Achieved validation accuracy"
    )
    test_accuracy: float | None = Field(
        default=None, description="Achieved test accuracy"
    )
    train_loss: float = Field(..., ge=0.0, description="Final training loss")
    val_loss: float = Field(..., ge=0.0, description="Final validation loss")
    epochs_trained: int = Field(..., ge=1, description="Number of epochs trained")
    best_epoch: int = Field(..., ge=0, description="Best validation epoch")


class SampleEfficiencyTransferSummary(BaseModel):
    """Summarizes target accuracy scaling trajectories across data budgets."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    architecture: str = Field(..., description="Model architecture")
    target_dataset_id: str = Field(..., description="Target dataset identifier")
    points: list[DataEfficiencyTransferPoint] = Field(
        default_factory=list, description="List of budget evaluation points"
    )
    normalized_auc: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalized area under the budget-accuracy curve",
    )


class TransferStrategyComparisonSummary(BaseModel):
    """Side-by-side comparison across all four transfer strategies."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scratch_accuracy: float = Field(
        ..., ge=0.0, le=1.0, description="Scratch baseline accuracy"
    )
    linear_probe_accuracy: float = Field(
        ..., ge=0.0, le=1.0, description="Linear probe accuracy"
    )
    partial_fine_tune_accuracy: float = Field(
        ..., ge=0.0, le=1.0, description="Partial fine-tuning accuracy"
    )
    full_fine_tune_accuracy: float = Field(
        ..., ge=0.0, le=1.0, description="Full fine-tuning accuracy"
    )
    linear_probe_gain: float = Field(
        ..., description="Linear probe accuracy delta vs scratch"
    )
    partial_fine_tune_gain: float = Field(
        ..., description="Partial fine-tune accuracy delta vs scratch"
    )
    full_fine_tune_gain: float = Field(
        ..., description="Full fine-tune accuracy delta vs scratch"
    )


class TransferLearningReport(BaseModel):
    """Comprehensive report capturing transfer learning results."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    transfer_id: str = Field(..., description="Unique transfer run identifier")
    specification: TransferLearningSpecification = Field(
        ..., description="Transfer specification and configuration"
    )
    freeze_plan: ParameterFreezePlan = Field(
        ..., description="Parameter freeze plan indicating trainable vs frozen elements"
    )
    source_model_id: str = Field(..., description="Source model identifier")
    target_model_id: str = Field(..., description="Target model identifier")
    architecture: str = Field(..., description="Model architecture family")
    strategy: TransferStrategy = Field(..., description="Applied transfer strategy")
    train_loss: float = Field(
        ..., ge=0.0, description="Final training loss on target data"
    )
    val_loss: float = Field(
        ..., ge=0.0, description="Final validation loss on target data"
    )
    train_accuracy: float = Field(
        ..., ge=0.0, le=1.0, description="Final target training accuracy"
    )
    val_accuracy: float = Field(
        ..., ge=0.0, le=1.0, description="Final target validation accuracy"
    )
    test_accuracy: float | None = Field(
        default=None, description="Final test accuracy if test split was evaluated"
    )
    epochs_trained: int = Field(
        ..., ge=1, description="Number of target epochs trained"
    )
    best_epoch: int = Field(
        ..., ge=0, description="Epoch with highest target validation accuracy"
    )
    scratch_comparison: TransferStrategyComparisonSummary | None = Field(
        default=None, description="Comparative summary against scratch baseline"
    )
    layer_probes: list[LayerTransferProbeResult] = Field(
        default_factory=list,
        description="Layer transferability probe results across model depth",
    )
    representation_drift: TransferRepresentationDriftSummary | None = Field(
        default=None, description="Pre/post transfer representation retention metrics"
    )
    sample_efficiency: SampleEfficiencyTransferSummary | None = Field(
        default=None, description="Data-efficiency trajectory over target data budgets"
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Scientific disclaimers or experimental warnings",
    )
    duration_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Wall-clock duration of transfer experiment in seconds",
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        return self.model_dump(mode="json")

    def to_json(self, indent: int | None = None) -> str:
        """Serialize report to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TransferLearningReport:
        """Deserialize report from dictionary."""
        try:
            return cls.model_validate(data)
        except Exception as exc:
            raise SerializationError(
                f"Failed to deserialize TransferLearningReport: {exc}"
            ) from exc


class TransferExperimentSuite(BaseModel):
    """Declarative suite planning combinatorial transfer studies."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    suite_id: str = Field(..., description="Unique suite identifier")
    name: str = Field(..., description="Human-readable suite title")
    description: str = Field(..., description="Research motivation and hypotheses")
    architectures: list[str] = Field(
        default_factory=lambda: ["cnn", "resnet", "vit"],
        description="Target architectures evaluated in the suite",
    )
    strategies: list[TransferStrategy] = Field(
        default_factory=lambda: [
            TransferStrategy.SCRATCH_BASELINE,
            TransferStrategy.LINEAR_PROBE,
            TransferStrategy.PARTIAL_FINE_TUNE,
            TransferStrategy.FULL_FINE_TUNE,
        ],
        description="Transfer strategies evaluated",
    )
    target_budgets: list[float] = Field(
        default_factory=lambda: [0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
        description="Target data budgets evaluated",
    )
    seeds: list[int] = Field(
        default_factory=lambda: [42],
        description="RNG seeds for repeated-trial analysis",
    )
