"""Structured evaluation reports and benchmark summaries for temporal experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from prism.temporal.contracts import (
    RNNDynamicsSummary,
    TemporalConsistencySummary,
    TemporalWeightSummary,
)
from prism.temporal.specification import TemporalTransferSpecification


@dataclass(frozen=True)
class TemporalRobustnessSummary:
    """Robustness evaluation metrics under specific temporal perturbations."""

    corruption_type: str
    clean_accuracy: float
    perturbed_accuracy: float
    accuracy_delta: float
    sequence_representation_drift: float
    temporal_consistency_change: float
    lineage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize robustness summary."""
        return {
            "corruption_type": self.corruption_type,
            "clean_accuracy": self.clean_accuracy,
            "perturbed_accuracy": self.perturbed_accuracy,
            "accuracy_delta": self.accuracy_delta,
            "sequence_representation_drift": self.sequence_representation_drift,
            "temporal_consistency_change": self.temporal_consistency_change,
            "lineage": self.lineage,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TemporalRobustnessSummary:
        """Deserialize robustness summary."""
        return cls(
            corruption_type=str(data["corruption_type"]),
            clean_accuracy=float(data["clean_accuracy"]),
            perturbed_accuracy=float(data["perturbed_accuracy"]),
            accuracy_delta=float(data["accuracy_delta"]),
            sequence_representation_drift=float(data["sequence_representation_drift"]),
            temporal_consistency_change=float(data["temporal_consistency_change"]),
            lineage=dict(data.get("lineage", {})),
        )


@dataclass(frozen=True)
class TemporalRepresentationRetentionRecord:
    """Tracks representation stability of encoder features before vs after transfer."""

    mean_frame_drift: float
    sequence_drift: float
    per_timestep_drift: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize retention record."""
        return {
            "mean_frame_drift": self.mean_frame_drift,
            "sequence_drift": self.sequence_drift,
            "per_timestep_drift": self.per_timestep_drift,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TemporalRepresentationRetentionRecord:
        """Deserialize retention record."""
        return cls(
            mean_frame_drift=float(data["mean_frame_drift"]),
            sequence_drift=float(data["sequence_drift"]),
            per_timestep_drift=[float(x) for x in data.get("per_timestep_drift", [])],
        )


@dataclass(frozen=True)
class TemporalLayerTransferabilityRecord:
    """Benchmark metrics for an encoder layer depth adapted for temporal tasks."""

    layer_name: str
    depth_fraction: float
    feature_dimension: int
    video_accuracy: float
    temporal_consistency: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize layer record."""
        return {
            "layer_name": self.layer_name,
            "depth_fraction": self.depth_fraction,
            "feature_dimension": self.feature_dimension,
            "video_accuracy": self.video_accuracy,
            "temporal_consistency": self.temporal_consistency,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TemporalLayerTransferabilityRecord:
        """Deserialize layer record."""
        return cls(
            layer_name=str(data["layer_name"]),
            depth_fraction=float(data["depth_fraction"]),
            feature_dimension=int(data["feature_dimension"]),
            video_accuracy=float(data["video_accuracy"]),
            temporal_consistency=float(data["temporal_consistency"]),
        )


@dataclass(frozen=True)
class TemporalObjectiveComparisonSummary:
    """Comparative summary across Supervised, SimCLR, Reconstruction, and Scratch."""

    objective: str
    frozen_accuracy: float
    finetune_accuracy: float
    temporal_consistency: float
    sequence_drift: float
    trainable_fraction: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize objective comparison."""
        return {
            "objective": self.objective,
            "frozen_accuracy": self.frozen_accuracy,
            "finetune_accuracy": self.finetune_accuracy,
            "temporal_consistency": self.temporal_consistency,
            "sequence_drift": self.sequence_drift,
            "trainable_fraction": self.trainable_fraction,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TemporalObjectiveComparisonSummary:
        """Deserialize objective comparison."""
        return cls(
            objective=str(data["objective"]),
            frozen_accuracy=float(data["frozen_accuracy"]),
            finetune_accuracy=float(data["finetune_accuracy"]),
            temporal_consistency=float(data["temporal_consistency"]),
            sequence_drift=float(data["sequence_drift"]),
            trainable_fraction=float(data["trainable_fraction"]),
        )


@dataclass
class TemporalRepresentationReport:
    """Comprehensive experimental report for PRISM temporal representation studies."""

    spec: TemporalTransferSpecification
    video_accuracy: float
    frame_baseline_accuracy: float
    temporal_consistency: TemporalConsistencySummary
    mean_sequence_drift: float
    trainable_fraction: float
    drift_curve: list[dict[str, float]] = field(default_factory=list)
    motion_sensitivity: dict[str, Any] = field(default_factory=dict)
    weight_summary: TemporalWeightSummary | None = None
    rnn_dynamics: RNNDynamicsSummary | None = None
    robustness_summaries: dict[str, TemporalRobustnessSummary] = field(
        default_factory=dict
    )
    retention_record: TemporalRepresentationRetentionRecord | None = None
    candidate_failures: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize full experiment report."""
        return {
            "spec": self.spec.to_dict(),
            "video_accuracy": self.video_accuracy,
            "frame_baseline_accuracy": self.frame_baseline_accuracy,
            "temporal_consistency": self.temporal_consistency.to_dict(),
            "mean_sequence_drift": self.mean_sequence_drift,
            "trainable_fraction": self.trainable_fraction,
            "drift_curve": self.drift_curve,
            "motion_sensitivity": self.motion_sensitivity,
            "weight_summary": (
                self.weight_summary.to_dict() if self.weight_summary else None
            ),
            "rnn_dynamics": (
                self.rnn_dynamics.to_dict() if self.rnn_dynamics else None
            ),
            "robustness_summaries": {
                k: v.to_dict() for k, v in self.robustness_summaries.items()
            },
            "retention_record": (
                self.retention_record.to_dict() if self.retention_record else None
            ),
            "candidate_failures": self.candidate_failures,
            "warnings": self.warnings,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TemporalRepresentationReport:
        """Deserialize full experiment report."""
        spec = TemporalTransferSpecification.from_dict(data["spec"])
        consistency = TemporalConsistencySummary.from_dict(data["temporal_consistency"])
        weight_sum = (
            TemporalWeightSummary.from_dict(data["weight_summary"])
            if data.get("weight_summary")
            else None
        )
        rnn_dyn = (
            RNNDynamicsSummary.from_dict(data["rnn_dynamics"])
            if data.get("rnn_dynamics")
            else None
        )
        retention = (
            TemporalRepresentationRetentionRecord.from_dict(data["retention_record"])
            if data.get("retention_record")
            else None
        )
        robustness = {
            k: TemporalRobustnessSummary.from_dict(v)
            for k, v in data.get("robustness_summaries", {}).items()
        }

        return cls(
            spec=spec,
            video_accuracy=float(data["video_accuracy"]),
            frame_baseline_accuracy=float(data["frame_baseline_accuracy"]),
            temporal_consistency=consistency,
            mean_sequence_drift=float(data["mean_sequence_drift"]),
            trainable_fraction=float(data["trainable_fraction"]),
            drift_curve=list(data.get("drift_curve", [])),
            motion_sensitivity=dict(data.get("motion_sensitivity", {})),
            weight_summary=weight_sum,
            rnn_dynamics=rnn_dyn,
            robustness_summaries=robustness,
            retention_record=retention,
            candidate_failures=list(data.get("candidate_failures", [])),
            warnings=[str(w) for w in data.get("warnings", [])],
        )
