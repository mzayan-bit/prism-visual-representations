"""PRISM Research Benchmarking, Evidence Synthesis & Reporting Subsystem."""

from __future__ import annotations

from prism.benchmarking.adapters import (
    adapt_any_report,
    adapt_architecture_comparison_report,
    adapt_explainability_report,
    adapt_multimodal_report,
    adapt_reconstruction_report,
    adapt_representation_geometry_report,
    adapt_robustness_report,
    adapt_spatial_transfer_report,
    adapt_ssl_report,
    adapt_temporal_report,
    adapt_transfer_report,
    adapt_uncertainty_report,
)
from prism.benchmarking.aggregation import (
    aggregate_repeated_seeds,
    group_and_aggregate,
)
from prism.benchmarking.comparisons import (
    audit_comparison_control,
    compute_pairwise_comparison,
)
from prism.benchmarking.contracts import (
    AggregatedBenchmarkResult,
    BenchmarkCampaign,
    BenchmarkMatrix,
    BenchmarkResultCell,
    BenchmarkTable,
    ComparisonControlAudit,
    EvidenceGap,
    FigureSpecification,
    MetricDefinition,
    MissingExperimentPlan,
    PairwiseComparisonResult,
    ParetoAnalysisResult,
    PRISMResearchReport,
    RepresentationProfile,
    ReproducibilityManifest,
    ResearchFinding,
    ResearchQuestion,
    ResearchReportSpecification,
    TradeoffPoint,
    compute_campaign_fingerprint,
)
from prism.benchmarking.coverage import (
    CampaignCoverageSummary,
    ExperimentCoverageMatrix,
    build_missing_experiment_plan,
    compute_campaign_coverage_summary,
    compute_coverage_matrix,
    detect_evidence_gaps,
)
from prism.benchmarking.enums import (
    CampaignStatus,
    ChartType,
    ComparisonControlStatus,
    EvidenceStrength,
    FactorID,
    MetricCategory,
    MetricDirection,
    ResultStatus,
)
from prism.benchmarking.export import (
    export_matrix_to_csv,
    export_report_to_json,
    export_report_to_markdown,
    export_table_to_csv,
)
from prism.benchmarking.findings import generate_research_findings
from prism.benchmarking.matrices import (
    build_benchmark_matrix,
    build_benchmark_table,
)
from prism.benchmarking.registry import (
    FactorRegistry,
    MetricRegistry,
    canonical_metric_registry,
)
from prism.benchmarking.reporting import (
    build_reproducibility_manifest,
    compile_research_report,
)
from prism.benchmarking.runner import (
    BenchmarkCampaignRunner,
    BenchmarkExecutionFailure,
    BenchmarkExecutionSummary,
)
from prism.benchmarking.service import (
    BenchmarkService,
    create_default_prism_campaign,
)
from prism.benchmarking.store import BenchmarkResultStore
from prism.benchmarking.synthesis import (
    compute_pareto_front,
    extract_representation_profile,
    extract_tradeoff_pairs,
    synthesize_cross_architecture,
    synthesize_cross_objective,
)

__all__ = [
    "AggregatedBenchmarkResult",
    "BenchmarkCampaign",
    "BenchmarkCampaignRunner",
    "BenchmarkExecutionFailure",
    "BenchmarkExecutionSummary",
    "BenchmarkMatrix",
    "BenchmarkResultCell",
    "BenchmarkResultStore",
    "BenchmarkService",
    "BenchmarkTable",
    "CampaignCoverageSummary",
    "CampaignStatus",
    "ChartType",
    "ComparisonControlAudit",
    "ComparisonControlStatus",
    "EvidenceGap",
    "EvidenceStrength",
    "ExperimentCoverageMatrix",
    "FactorID",
    "FactorRegistry",
    "FigureSpecification",
    "MetricCategory",
    "MetricDefinition",
    "MetricDirection",
    "MetricRegistry",
    "MissingExperimentPlan",
    "PRISMResearchReport",
    "PairwiseComparisonResult",
    "ParetoAnalysisResult",
    "RepresentationProfile",
    "ReproducibilityManifest",
    "ResearchFinding",
    "ResearchQuestion",
    "ResearchReportSpecification",
    "ResultStatus",
    "TradeoffPoint",
    "adapt_any_report",
    "adapt_architecture_comparison_report",
    "adapt_explainability_report",
    "adapt_multimodal_report",
    "adapt_reconstruction_report",
    "adapt_representation_geometry_report",
    "adapt_robustness_report",
    "adapt_spatial_transfer_report",
    "adapt_ssl_report",
    "adapt_temporal_report",
    "adapt_transfer_report",
    "adapt_uncertainty_report",
    "aggregate_repeated_seeds",
    "audit_comparison_control",
    "build_benchmark_matrix",
    "build_benchmark_table",
    "build_missing_experiment_plan",
    "build_reproducibility_manifest",
    "canonical_metric_registry",
    "compile_research_report",
    "compute_campaign_coverage_summary",
    "compute_campaign_fingerprint",
    "compute_coverage_matrix",
    "compute_pairwise_comparison",
    "compute_pareto_front",
    "create_default_prism_campaign",
    "detect_evidence_gaps",
    "export_matrix_to_csv",
    "export_report_to_json",
    "export_report_to_markdown",
    "export_table_to_csv",
    "extract_representation_profile",
    "extract_tradeoff_pairs",
    "generate_research_findings",
    "group_and_aggregate",
    "synthesize_cross_architecture",
    "synthesize_cross_objective",
]
