"""Compatibility exports for experiment-suite contracts."""

from prism.experiments.architecture import (
    ArchitectureComparisonSuite,
    ComparisonMode,
    ControlledExperimentSuite,
    ExperimentFactorAudit,
    ExperimentSuite,
    ParameterCountAudit,
    SuiteStatus,
    audit_experiment_factors,
    count_model_parameters,
    create_architecture_comparison_suite,
)

__all__ = [
    "ArchitectureComparisonSuite",
    "ComparisonMode",
    "ControlledExperimentSuite",
    "ExperimentFactorAudit",
    "ExperimentSuite",
    "ParameterCountAudit",
    "SuiteStatus",
    "audit_experiment_factors",
    "count_model_parameters",
    "create_architecture_comparison_suite",
]
