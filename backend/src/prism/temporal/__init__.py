"""PRISM Video & Temporal Representation Learning Subsystem."""

from prism.temporal.adapter import TemporalFrameEncoder, get_available_temporal_layers
from prism.temporal.aggregators import (
    BaseTemporalAggregator,
    LastFramePooling,
    LearnedTemporalPooling,
    MaxTemporalPooling,
    MeanTemporalPooling,
    SimpleRNN,
)
from prism.temporal.contracts import (
    FrameMetadata,
    MotionTrajectory,
    RNNDynamicsSummary,
    TemporalConsistencySummary,
    TemporalWeightSummary,
    VideoBatch,
    VideoSample,
)
from prism.temporal.enums import (
    PretrainingObjective,
    RNNAggregationMode,
    TemporalAggregationType,
    TemporalCorruptionType,
    TemporalFailureType,
    TemporalTaskType,
    TemporalTransferStrategy,
)
from prism.temporal.heads import TemporalClassificationHead, TemporalRepresentationModel
from prism.temporal.metrics import (
    compute_motion_sensitivity,
    compute_temporal_consistency,
    compute_temporal_drift_curve,
    compute_video_classification_metrics,
)
from prism.temporal.synthetic import SyntheticVideoGenerator

__all__ = [
    "BaseTemporalAggregator",
    "FrameMetadata",
    "LastFramePooling",
    "LearnedTemporalPooling",
    "MaxTemporalPooling",
    "MeanTemporalPooling",
    "MotionTrajectory",
    "PretrainingObjective",
    "RNNAggregationMode",
    "RNNDynamicsSummary",
    "SimpleRNN",
    "SyntheticVideoGenerator",
    "TemporalAggregationType",
    "TemporalClassificationHead",
    "TemporalConsistencySummary",
    "TemporalCorruptionType",
    "TemporalFailureType",
    "TemporalFrameEncoder",
    "TemporalRepresentationModel",
    "TemporalTaskType",
    "TemporalTransferStrategy",
    "TemporalWeightSummary",
    "VideoBatch",
    "VideoSample",
    "compute_motion_sensitivity",
    "compute_temporal_consistency",
    "compute_temporal_drift_curve",
    "compute_video_classification_metrics",
    "get_available_temporal_layers",
]
