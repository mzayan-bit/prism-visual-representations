"""Experiment specification, deterministic execution, and run tracking."""

from prism.experiments.definitions import ExperimentDefinition
from prism.experiments.hashing import (
    DEFAULT_EXCLUDED_KEYS,
    compute_configuration_fingerprint,
)
from prism.experiments.lifecycle import (
    ALLOWED_TRANSITIONS,
    is_valid_transition,
    validate_transition,
)
from prism.experiments.metrics import MetricRecord
from prism.experiments.reproducibility import ReproducibilityConfiguration
from prism.experiments.runs import ExperimentRun, FailureInfo

__all__ = [
    "ALLOWED_TRANSITIONS",
    "DEFAULT_EXCLUDED_KEYS",
    "ExperimentDefinition",
    "ExperimentRun",
    "FailureInfo",
    "MetricRecord",
    "ReproducibilityConfiguration",
    "compute_configuration_fingerprint",
    "is_valid_transition",
    "validate_transition",
]
