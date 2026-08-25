"""Core abstractions, base contracts, and domain primitives for PRISM."""

from prism.core.enums import (
    ArtifactType,
    DevicePreference,
    InitializationStrategy,
    MetricDirection,
    ModelFamily,
    PrecisionMode,
    RunStatus,
    SplitName,
    TaskType,
)
from prism.core.errors import (
    ConfigurationError,
    FingerprintError,
    InvalidTransitionError,
    LifecycleError,
    PrismError,
    SerializationError,
    ValidationError,
)
from prism.core.identifiers import (
    ensure_valid_identifier,
    generate_artifact_id,
    generate_dataset_id,
    generate_experiment_id,
    generate_model_id,
    generate_report_id,
    generate_run_id,
    validate_identifier,
)
from prism.core.metadata import (
    CodeRevisionMetadata,
    CreationMetadata,
    EnvironmentMetadata,
)

__all__ = [
    "ArtifactType",
    "CodeRevisionMetadata",
    "ConfigurationError",
    "CreationMetadata",
    "DevicePreference",
    "EnvironmentMetadata",
    "FingerprintError",
    "InitializationStrategy",
    "InvalidTransitionError",
    "LifecycleError",
    "MetricDirection",
    "ModelFamily",
    "PrecisionMode",
    "PrismError",
    "RunStatus",
    "SerializationError",
    "SplitName",
    "TaskType",
    "ValidationError",
    "ensure_valid_identifier",
    "generate_artifact_id",
    "generate_dataset_id",
    "generate_experiment_id",
    "generate_model_id",
    "generate_report_id",
    "generate_run_id",
    "validate_identifier",
]
