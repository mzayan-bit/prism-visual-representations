"""Enumerations for cross-paradigm benchmark orchestration and synthesis."""

from __future__ import annotations

from enum import Enum


class ResultStatus(str, Enum):
    """Explicit status of a benchmark metric measurement."""

    OBSERVED = "observed"
    AGGREGATED = "aggregated"
    MISSING = "missing"
    FAILED = "failed"
    NOT_APPLICABLE = "not_applicable"


class MetricDirection(str, Enum):
    """Optimization direction for a benchmark metric."""

    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"
    NEUTRAL = "neutral"


class MetricCategory(str, Enum):
    """Categorical domain of a benchmark metric."""

    PERFORMANCE = "performance"
    GEOMETRY = "geometry"
    ROBUSTNESS = "robustness"
    EXPLAINABILITY = "explainability"
    TRANSFER = "transfer"
    EFFICIENCY = "efficiency"
    CALIBRATION = "calibration"
    OOD = "ood"
    MULTIMODAL = "multimodal"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"


class FactorID(str, Enum):
    """Canonical experimental factor identifiers."""

    ARCHITECTURE = "architecture"
    PRETRAINING_OBJECTIVE = "pretraining_objective"
    DATASET = "dataset"
    TASK = "task"
    DATA_BUDGET = "data_budget"
    SEED = "seed"
    TRANSFER_STRATEGY = "transfer_strategy"
    REPRESENTATION_LAYER = "representation_layer"
    CORRUPTION = "corruption"
    CORRUPTION_SEVERITY = "corruption_severity"
    SSL_TEMPERATURE = "ssl_temperature"
    MASK_RATIO = "mask_ratio"
    TEMPORAL_AGGREGATOR = "temporal_aggregator"
    MULTIMODAL_TEMPERATURE = "multimodal_temperature"
    CALIBRATION_MODE = "calibration_mode"
    OOD_SCORE = "ood_score"


class ComparisonControlStatus(str, Enum):
    """Scientific control rigor status of an experimental comparison."""

    STRICTLY_CONTROLLED = "strictly_controlled"
    PARTIALLY_CONTROLLED = "partially_controlled"
    DESCRIPTIVE_ONLY = "descriptive_only"
    INVALID_COMPARISON = "invalid_comparison"


class EvidenceStrength(str, Enum):
    """Evidence strength categorization for scientific findings."""

    SUPPORTED_BY_SINGLE_RUN = "supported_by_single_run"
    SUPPORTED_BY_REPEATED_RUNS = "supported_by_repeated_runs"
    DESCRIPTIVE_ONLY = "descriptive_only"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class CampaignStatus(str, Enum):
    """Lifecycle status of a benchmark campaign."""

    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


class ChartType(str, Enum):
    """Supported chart visualization formats for research reports."""

    LINE = "line"
    BAR = "bar"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    PARETO_SCATTER = "pareto_scatter"
    COVERAGE_MATRIX = "coverage_matrix"
    RELIABILITY = "reliability"
