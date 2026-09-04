"""Enumerations and domain types for PRISM Video & Temporal Representation Learning."""

from enum import Enum


class TemporalTaskType(str, Enum):
    """Supported temporal downstream tasks."""

    VIDEO_CLASSIFICATION = "video_classification"
    TEMPORAL_REPRESENTATION_ANALYSIS = "temporal_representation_analysis"
    FRAME_CLASSIFICATION = "frame_classification"


class TemporalAggregationType(str, Enum):
    """Mechanisms for aggregating frame representations over time."""

    MEAN_POOL = "mean_pool"
    MAX_POOL = "max_pool"
    LAST_FRAME = "last_frame"
    LEARNED_TEMPORAL_POOLING = "learned_temporal_pooling"
    SIMPLE_RNN = "simple_rnn"


class TemporalTransferStrategy(str, Enum):
    """Parameter update and freeze strategies for temporal adaptation."""

    FROZEN_FRAME_ENCODER = "frozen_frame_encoder"
    PARTIAL_FINE_TUNE = "partial_fine_tune"
    FULL_FINE_TUNE = "full_fine_tune"
    FRAME_INDEPENDENT = "frame_independent"


class TemporalCorruptionType(str, Enum):
    """Deterministic temporal perturbations for video robustness testing."""

    FRAME_DROP = "frame_drop"
    FRAME_DUPLICATION = "frame_duplication"
    FRAME_SHUFFLE = "frame_shuffle"
    TEMPORAL_SUBSAMPLING = "temporal_subsampling"
    SPATIAL_COMPOSITE = "spatial_composite"


class RNNAggregationMode(str, Enum):
    """Sequence pooling mode for SimpleRNN outputs."""

    LAST_HIDDEN = "last_hidden"
    MEAN_HIDDEN = "mean_hidden"


class PretrainingObjective(str, Enum):
    """Pretraining objectives evaluated for temporal transfer."""

    SUPERVISED = "supervised"
    SIMCLR = "simclr"
    RECONSTRUCTION = "reconstruction"
    SCRATCH = "scratch"


class TemporalFailureType(str, Enum):
    """Descriptive taxonomy of temporal representation and prediction failures."""

    TEMPORAL_INCONSISTENCY = "temporal_inconsistency"
    ORDER_SENSITIVITY_FAILURE = "order_sensitivity_failure"
    FRAME_DROP_FAILURE = "frame_drop_failure"
    HIGH_SEQUENCE_DRIFT = "high_sequence_drift"
    STATIC_SEQUENCE_DRIFT = "static_sequence_drift"
    MOTION_INSENSITIVITY = "motion_insensitivity"
    OVER_CONCENTRATED_TEMPORAL_WEIGHTING = "over_concentrated_temporal_weighting"
