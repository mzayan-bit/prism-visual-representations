"""Centralized enums and domain literals for PRISM."""

from enum import Enum


class TaskType(str, Enum):
    """Supported machine learning and research task types."""

    CLASSIFICATION = "classification"
    REPRESENTATION_LEARNING = "representation_learning"
    ROBUSTNESS = "robustness"
    EXPLAINABILITY = "explainability"
    VIDEO_UNDERSTANDING = "video_understanding"
    OBJECT_DETECTION = "object_detection"
    SEGMENTATION = "segmentation"
    TRANSFER_LEARNING = "transfer_learning"


class RunStatus(str, Enum):
    """Lifecycle status states for experiment runs."""

    PLANNED = "planned"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        """Return True if this status represents a finished terminal state."""
        return self in (
            RunStatus.COMPLETED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        )

    @property
    def is_active(self) -> bool:
        """Return True if the run is currently queued or in-flight."""
        return self in (RunStatus.QUEUED, RunStatus.RUNNING)


class ModelFamily(str, Enum):
    """Architectural families for vision representation models."""

    LINEAR = "linear"
    MLP = "mlp"
    CNN = "cnn"
    RESNET = "resnet"
    VISION_TRANSFORMER = "vision_transformer"
    SELF_SUPERVISED_ENCODER = "self_supervised_encoder"
    RECURRENT_TEMPORAL = "recurrent_temporal"
    HYBRID = "hybrid"


class InitializationStrategy(str, Enum):
    """Parameter initialization or pretraining strategies."""

    RANDOM = "random"
    SCRATCH = "scratch"
    PRETRAINED = "pretrained"
    TRANSFER_FROZEN = "transfer_frozen"
    TRANSFER_FINETUNE = "transfer_finetune"


class ArtifactType(str, Enum):
    """Classifications of generated experiment artifacts."""

    CHECKPOINT = "checkpoint"
    METRICS_JSON = "metrics_json"
    TRAINING_CURVE = "training_curve"
    CONFUSION_MATRIX = "confusion_matrix"
    EMBEDDING_PROJECTION = "embedding_projection"
    ATTENTION_MAP = "attention_map"
    ROBUSTNESS_REPORT = "robustness_report"
    EVALUATION_REPORT = "evaluation_report"
    FIGURE = "figure"
    DATASET_MANIFEST = "dataset_manifest"
    OTHER = "other"


class MetricDirection(str, Enum):
    """Optimization or evaluation direction for a metric."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"
    NEUTRAL = "neutral"


class PrecisionMode(str, Enum):
    """Numerical precision mode for training and inference."""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    AMP = "amp"


class DevicePreference(str, Enum):
    """Hardware device preference for execution."""

    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class SplitName(str, Enum):
    """Canonical dataset split names."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    OOD = "ood"
    CUSTOM = "custom"


class OrderingStrategy(str, Enum):
    """Deterministic sampling and batch ordering strategies."""

    SEQUENTIAL = "sequential"
    FIXED_SHUFFLE = "fixed_shuffle"
    EPOCH_AWARE_SHUFFLE = "epoch_aware_shuffle"
