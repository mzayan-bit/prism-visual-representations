"""Experiment specification, deterministic execution, and run tracking."""

from prism.experiments.architecture import (
    ArchitectureComparisonSuite,
    ComparisonMode,
    ExperimentFactorAudit,
    ParameterCountAudit,
    SuiteStatus,
    audit_experiment_factors,
    count_model_parameters,
    create_architecture_comparison_suite,
)
from prism.experiments.comparisons import (
    ControlledComparison,
    create_normalization_comparison,
    create_residual_comparison,
    create_scheduler_comparison,
    create_vit_architecture_comparison,
)
from prism.experiments.context import PreparedExecution, RuntimeContext
from prism.experiments.definitions import ExperimentDefinition
from prism.experiments.environment import (
    DEFAULT_ALLOWLIST_PACKAGES,
    capture_environment,
)
from prism.experiments.hardware import probe_hardware
from prism.experiments.harness import ExperimentExecutionHarness
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
from prism.experiments.provenance import (
    GitProvenance,
    inspect_git_provenance,
)
from prism.experiments.reproducibility import ReproducibilityConfiguration
from prism.experiments.runs import ExperimentRun, FailureInfo
from prism.experiments.seeding import (
    SeedInitializationResult,
    initialize_seeds,
)

__all__ = [
    "ALLOWED_TRANSITIONS",
    "DEFAULT_ALLOWLIST_PACKAGES",
    "DEFAULT_EXCLUDED_KEYS",
    "ArchitectureComparisonSuite",
    "ComparisonMode",
    "ControlledComparison",
    "ExperimentDefinition",
    "ExperimentExecutionHarness",
    "ExperimentFactorAudit",
    "ExperimentRun",
    "FailureInfo",
    "GitProvenance",
    "MetricRecord",
    "ParameterCountAudit",
    "PreparedExecution",
    "ReproducibilityConfiguration",
    "RuntimeContext",
    "SeedInitializationResult",
    "SuiteStatus",
    "audit_experiment_factors",
    "capture_environment",
    "compute_configuration_fingerprint",
    "count_model_parameters",
    "create_architecture_comparison_suite",
    "create_normalization_comparison",
    "create_residual_comparison",
    "create_scheduler_comparison",
    "create_vit_architecture_comparison",
    "initialize_seeds",
    "inspect_git_provenance",
    "is_valid_transition",
    "probe_hardware",
    "validate_transition",
]
