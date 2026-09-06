"""Contracts and Pydantic schemas for research benchmarking and synthesis."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from prism.benchmarking.enums import (
    CampaignStatus,
    ChartType,
    ComparisonControlStatus,
    EvidenceStrength,
    MetricCategory,
    MetricDirection,
    ResultStatus,
)


class MetricDefinition(BaseModel):
    """Metadata and semantics for a benchmark metric."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    metric_id: str = Field(description="Unique metric identifier")
    display_name: str = Field(description="Human-readable display name")
    category: MetricCategory = Field(description="Metric domain category")
    unit: str = Field(default="", description="Measurement unit (%, nats, etc.)")
    direction: MetricDirection = Field(
        default=MetricDirection.HIGHER_IS_BETTER,
        description="Optimization direction",
    )
    valid_domains: list[str] = Field(
        default_factory=list, description="Applicable learning paradigms"
    )
    valid_tasks: list[str] = Field(
        default_factory=list, description="Applicable task types"
    )
    aggregation_policy: str = Field(
        default="mean_std", description="Repeated-seed aggregation method"
    )
    bounded_range: list[float] | None = Field(
        default=None, description="Theoretical lower and upper bounds [min, max]"
    )
    description: str = Field(default="", description="Detailed scientific definition")
    methodological_notes: str = Field(
        default="", description="Caveats, limitations, and interpretation rules"
    )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class ResearchQuestion(BaseModel):
    """Structured scientific research question with controlled factors."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    question_id: str = Field(description="Unique question identifier")
    natural_language_question: str = Field(description="Core scientific question")
    independent_variables: list[str] = Field(
        description="Intentionally varied factor names"
    )
    independent_values: list[str] = Field(
        default_factory=list, description="Evaluated factor variants"
    )
    dependent_metrics: list[str] = Field(
        description="Target outcome metric identifiers"
    )
    controlled_factors: dict[str, Any] = Field(
        default_factory=dict, description="Factors that must remain identical"
    )
    applicable_experiments: list[str] = Field(
        default_factory=list, description="Bound experiment IDs"
    )
    comparison_policy: str = Field(
        default="strictly_controlled", description="Comparison audit policy"
    )
    limitations: list[str] = Field(
        default_factory=list, description="Scope limitations"
    )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class BenchmarkCampaign(BaseModel):
    """Top-level orchestrated research benchmark campaign definition."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    campaign_id: str = Field(description="Unique campaign identifier")
    title: str = Field(description="Campaign display title")
    description: str = Field(description="Detailed scope description")
    research_questions: list[ResearchQuestion] = Field(
        default_factory=list, description="Registered research questions"
    )
    architectures: list[str] = Field(
        default_factory=list, description="Included visual architectures"
    )
    objectives: list[str] = Field(
        default_factory=list, description="Included pretraining objectives"
    )
    datasets: list[str] = Field(default_factory=list, description="Evaluated datasets")
    tasks: list[str] = Field(default_factory=list, description="Evaluated task types")
    transfer_strategies: list[str] = Field(
        default_factory=list, description="Transfer strategies"
    )
    seeds: list[int] = Field(default_factory=list, description="Evaluated RNG seeds")
    budgets: list[float] = Field(
        default_factory=list, description="Evaluated data budgets"
    )
    experiment_definitions: list[str] = Field(
        default_factory=list, description="Constituent experiment IDs"
    )
    requested_metrics: list[str] = Field(
        default_factory=list, description="Target metric IDs"
    )
    requested_reports: list[str] = Field(
        default_factory=list, description="Requested report schemas"
    )
    artifact_policy: str = Field(
        default="metadata_only", description="Artifact persistence policy"
    )
    status: CampaignStatus = Field(
        default=CampaignStatus.PLANNED, description="Campaign lifecycle status"
    )
    warnings: list[str] = Field(
        default_factory=list, description="Campaign-level warnings"
    )
    fingerprint: str = Field(
        default="", description="Deterministic SHA-256 fingerprint"
    )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def compute_campaign_fingerprint(
    campaign_id: str,
    architectures: list[str],
    objectives: list[str],
    datasets: list[str],
    tasks: list[str],
    seeds: list[int],
    budgets: list[float],
) -> str:
    """Compute deterministic SHA-256 fingerprint for a benchmark campaign."""
    payload = {
        "campaign_id": campaign_id,
        "architectures": sorted(architectures),
        "objectives": sorted(objectives),
        "datasets": sorted(datasets),
        "tasks": sorted(tasks),
        "seeds": sorted(seeds),
        "budgets": sorted(budgets),
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class BenchmarkResultCell(BaseModel):
    """Canonical atomic metric measurement cell with complete provenance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    result_id: str = Field(description="Unique result cell identifier")
    experiment_id: str = Field(description="Originating experiment ID")
    experiment_fingerprint: str = Field(description="Experiment fingerprint")
    metric_id: str = Field(description="Measured metric identifier")
    value: float | None = Field(
        default=None, description="Numerical metric value (None if missing/failed/NA)"
    )
    status: ResultStatus = Field(
        default=ResultStatus.OBSERVED, description="Cell measurement status"
    )
    seed: int | None = Field(default=None, description="Evaluation random seed")
    sample_count: int | None = Field(
        default=None, description="Number of evaluated samples"
    )
    source_report_type: str = Field(
        default="unknown", description="Source report schema"
    )
    source_run_id: str = Field(default="run_unknown", description="Originating run ID")
    source_artifact: str | None = Field(
        default=None, description="Artifact checksum / reference"
    )
    factors: dict[str, Any] = Field(
        default_factory=dict, description="Experimental factors (arch, obj, task, etc.)"
    )
    warnings: list[str] = Field(
        default_factory=list, description="Measurement warnings (single seed, etc.)"
    )
    provenance: dict[str, Any] = Field(
        default_factory=dict, description="Traceability metadata"
    )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class AggregatedBenchmarkResult(BaseModel):
    """Repeated-seed aggregation of multiple result cells."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    metric_id: str = Field(description="Metric identifier")
    group_factors: dict[str, Any] = Field(description="Shared factor configuration")
    sample_count: int = Field(ge=0, description="Number of constituent runs/seeds")
    mean: float | None = Field(default=None, description="Sample mean")
    std: float | None = Field(
        default=None, description="Sample standard deviation (None if N=1)"
    )
    min_value: float | None = Field(default=None, description="Minimum observed value")
    max_value: float | None = Field(default=None, description="Maximum observed value")
    median_value: float | None = Field(
        default=None, description="Median observed value"
    )
    member_result_ids: list[str] = Field(
        default_factory=list, description="IDs of aggregated result cells"
    )
    warnings: list[str] = Field(
        default_factory=list, description="Aggregation warnings"
    )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class ComparisonControlAudit(BaseModel):
    """Audit of experimental factor consistency for pairwise or group comparisons."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    comparison_id: str = Field(description="Unique comparison identifier")
    factors_expected_equal: list[str] = Field(description="Factors required to match")
    factors_actually_equal: list[str] = Field(description="Factors verified matching")
    varied_factors: list[str] = Field(
        default_factory=list,
        description="Factors intentionally or unintentionally varied",
    )
    mismatches: dict[str, tuple[Any, Any]] = Field(
        default_factory=dict, description="Factor mismatches {factor: (val_a, val_b)}"
    )
    status: ComparisonControlStatus = Field(description="Scientific rigor audit status")
    warnings: list[str] = Field(
        default_factory=list, description="Audit warnings and caveats"
    )

    @property
    def is_strictly_controlled(self) -> bool:
        return self.status == ComparisonControlStatus.STRICTLY_CONTROLLED

    @property
    def is_controlled(self) -> bool:
        return self.status in (
            ComparisonControlStatus.STRICTLY_CONTROLLED,
            ComparisonControlStatus.PARTIALLY_CONTROLLED,
        )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class PairwiseComparisonResult(BaseModel):
    """Structured descriptive comparison between two benchmark configurations."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    comparison_id: str = Field(description="Comparison identifier")
    metric_id: str = Field(description="Target metric identifier")
    cell_a_id: str = Field(description="Result cell A identifier")
    cell_b_id: str = Field(description="Result cell B identifier")
    value_a: float | None = Field(default=None, description="Value A")
    value_b: float | None = Field(default=None, description="Value B")
    absolute_delta: float | None = Field(
        default=None, description="Absolute difference (B - A)"
    )
    percentage_delta: float | None = Field(
        default=None, description="Percentage change relative to A"
    )
    direction: MetricDirection = Field(
        default=MetricDirection.HIGHER_IS_BETTER,
        description="Metric optimization direction",
    )
    favorable_candidate: str | None = Field(
        default=None,
        description="Candidate with more favorable metric ('A', 'B', or None)",
    )
    control_audit: ComparisonControlAudit = Field(description="Factor control audit")
    descriptive_interpretation: str = Field(
        default="", description="Descriptive, non-causal interpretation"
    )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class BenchmarkMatrix(BaseModel):
    """2D matrix view of benchmark metrics across two experimental factors."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    matrix_id: str = Field(description="Matrix identifier")
    title: str = Field(description="Matrix title")
    row_factor: str = Field(description="Row factor name")
    column_factor: str = Field(description="Column factor name")
    metric_id: str = Field(description="Metric identifier")
    row_values: list[str] = Field(description="Row factor values")
    column_values: list[str] = Field(description="Column factor values")
    cells: dict[
        str, dict[str, BenchmarkResultCell | AggregatedBenchmarkResult | None]
    ] = Field(description="2D mapping: cells[row][col] -> cell")
    warnings: list[str] = Field(default_factory=list, description="Matrix warnings")

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class BenchmarkTable(BaseModel):
    """Tabular presentation structure for scientific reporting."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    table_id: str = Field(description="Table identifier")
    title: str = Field(description="Table display title")
    research_question_id: str | None = Field(
        default=None, description="Associated research question ID"
    )
    row_factor: str = Field(description="Row factor name")
    column_factor: str = Field(description="Column factor name")
    metric_id: str = Field(description="Metric identifier")
    unit: str = Field(default="", description="Measurement unit")
    metric_direction: MetricDirection = Field(
        default=MetricDirection.HIGHER_IS_BETTER,
        description="Metric optimization direction",
    )
    rows: list[dict[str, Any]] = Field(
        description="Tabular row records with values, std, status, and seed counts"
    )
    footnotes: list[str] = Field(
        default_factory=list, description="Methodological notes and caveats"
    )
    control_status: ComparisonControlStatus = Field(
        default=ComparisonControlStatus.STRICTLY_CONTROLLED,
        description="Table-wide control status",
    )
    warnings: list[str] = Field(default_factory=list, description="Table warnings")

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class RepresentationProfile(BaseModel):
    """Multi-dimensional visual representation profile across 10 independent axes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    profile_id: str = Field(description="Profile identifier")
    architecture: str = Field(description="Visual architecture")
    objective: str = Field(description="Pretraining objective")
    semantic_performance: float | None = Field(
        default=None, description="Standard in-distribution accuracy [0, 1]"
    )
    geometry: float | None = Field(
        default=None, description="Inter-class separation / intra-class compactness"
    )
    label_efficiency: float | None = Field(
        default=None, description="Low-budget transfer accuracy (e.g. 10% labels)"
    )
    transferability: float | None = Field(
        default=None, description="Downstream linear probe accuracy"
    )
    robustness: float | None = Field(
        default=None, description="Corruption accuracy retention [0, 1]"
    )
    spatial_transfer: float | None = Field(
        default=None, description="Segmentation mIoU / detection IoU"
    )
    temporal_transfer: float | None = Field(
        default=None, description="Video classification accuracy"
    )
    calibration: float | None = Field(
        default=None, description="1.0 - ECE (higher is better calibrated)"
    )
    ood_separation: float | None = Field(
        default=None, description="OOD detection AUROC [0, 1]"
    )
    multimodal_alignment: float | None = Field(
        default=None, description="Zero-shot accuracy / retrieval R@1"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Auxiliary provenance metadata"
    )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class ParetoAnalysisResult(BaseModel):
    """Multi-objective Pareto-optimal trade-off analysis."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    analysis_id: str = Field(description="Analysis identifier")
    metric_ids: list[str] = Field(description="Evaluated trade-off metric IDs")
    candidate_experiment_ids: list[str] = Field(
        description="All evaluated candidate configurations"
    )
    non_dominated_experiment_ids: list[str] = Field(
        description="Non-dominated Pareto-optimal experiment IDs"
    )
    dominated_relationships: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Mapping: dominated_id -> list of dominating experiment IDs",
    )
    exclusions: list[str] = Field(
        default_factory=list, description="Excluded configurations"
    )
    missing_metric_warnings: list[str] = Field(
        default_factory=list,
        description="Warnings for candidates missing target metrics",
    )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class FigureSpecification(BaseModel):
    """Specification for chart rendering in research reporting."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    figure_id: str = Field(description="Figure identifier")
    title: str = Field(description="Figure title")
    chart_type: ChartType = Field(description="Chart visualization format")
    x_factor: str = Field(description="X-axis factor or metric")
    y_metric: str = Field(description="Y-axis metric")
    grouping: str | None = Field(default=None, description="Series grouping factor")
    series: list[dict[str, Any]] = Field(
        default_factory=list, description="Data series payloads"
    )
    provenance: dict[str, Any] = Field(
        default_factory=dict, description="Source experiment and cell IDs"
    )
    caption: str = Field(description="Descriptive, non-causal figure caption")
    methodological_notes: list[str] = Field(
        default_factory=list, description="Methodological limitations"
    )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class TradeoffPoint(BaseModel):
    """Paired metric data point for tradeoff analysis."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_id: str = Field(description="Experiment identifier")
    factors: dict[str, Any] = Field(default_factory=dict, description="Factor settings")
    x_metric: str = Field(description="X-axis metric")
    x_value: float = Field(description="X-axis value")
    y_metric: str = Field(description="Y-axis metric")
    y_value: float = Field(description="Y-axis value")
    note: str = Field(
        default="Descriptive tradeoff pair; does not imply causal relationship.",
        description="Descriptive note",
    )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class EvidenceGap(BaseModel):
    """Structured record of missing experimental evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    gap_id: str = Field(description="Gap identifier")
    research_question_id: str = Field(description="Associated research question ID")
    missing_factor_combination: dict[str, Any] = Field(
        description="Unexplored factor values"
    )
    missing_metric_id: str | None = Field(
        default=None, description="Missing metric measurement"
    )
    missing_seed_count: int = Field(default=1, description="Number of missing seeds")
    rationale: str = Field(description="Why this evidence is needed")

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class MissingExperimentPlan(BaseModel):
    """Plan of missing experiments needed to complete campaign coverage."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    plan_id: str = Field(description="Plan identifier")
    campaign_id: str = Field(description="Campaign identifier")
    missing_experiments: list[dict[str, Any]] = Field(
        default_factory=list, description="Required experiment specifications"
    )
    estimated_work_units: int = Field(default=0, description="Total missing run count")
    warnings: list[str] = Field(default_factory=list, description="Planning warnings")

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class ResearchFinding(BaseModel):
    """Template-generated, evidence-backed scientific finding."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    finding_id: str = Field(description="Finding identifier")
    research_question_id: str = Field(description="Associated research question ID")
    statement: str = Field(description="Evidence-backed finding statement")
    supporting_result_ids: list[str] = Field(
        description="Result cell IDs providing empirical support"
    )
    comparison_audit: ComparisonControlAudit | None = Field(
        default=None, description="Factor control audit"
    )
    effect_size_delta: float | None = Field(
        default=None, description="Measured difference / effect size"
    )
    scope: dict[str, Any] = Field(
        default_factory=dict,
        description="Explicit boundaries (dataset, task, arch, etc.)",
    )
    caveats: list[str] = Field(
        default_factory=list, description="Methodological caveats and limitations"
    )
    evidence_strength: EvidenceStrength = Field(
        description="Scientific evidence strength"
    )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class ReproducibilityManifest(BaseModel):
    """Complete provenance and configuration manifest for a benchmark campaign."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    manifest_id: str = Field(description="Manifest identifier")
    campaign_fingerprint: str = Field(description="Campaign fingerprint")
    experiment_fingerprints: dict[str, str] = Field(
        default_factory=dict, description="Mapping: exp_id -> SHA-256 fingerprint"
    )
    dataset_fingerprints: dict[str, str] = Field(
        default_factory=dict, description="Mapping: dataset_name -> fingerprint"
    )
    seeds: list[int] = Field(default_factory=list, description="Evaluated RNG seeds")
    model_specifications: dict[str, Any] = Field(
        default_factory=dict, description="Model specifications"
    )
    optimizer_specifications: dict[str, Any] = Field(
        default_factory=dict, description="Optimizer specifications"
    )
    scheduler_specifications: dict[str, Any] = Field(
        default_factory=dict, description="Scheduler specifications"
    )
    git_commit: str | None = Field(
        default=None, description="Git commit hash when available"
    )
    environment_provenance: dict[str, Any] = Field(
        default_factory=dict, description="Hardware / environment metadata"
    )
    artifact_checksums: dict[str, str] = Field(
        default_factory=dict, description="Mapping: artifact_id -> checksum"
    )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class ResearchReportSpecification(BaseModel):
    """Specification for assembling a structured PRISM research report."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    report_id: str = Field(description="Report identifier")
    title: str = Field(description="Report title")
    campaign_id: str = Field(description="Campaign identifier")
    selected_question_ids: list[str] = Field(
        default_factory=list, description="Selected research question IDs"
    )
    included_sections: list[str] = Field(
        default_factory=list, description="Included section titles"
    )
    output_formats: list[str] = Field(
        default_factory=lambda: ["json", "markdown", "csv"],
        description="Requested output formats",
    )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class PRISMResearchReport(BaseModel):
    """Unified comprehensive PRISM research synthesis report."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    report_id: str = Field(description="Report identifier")
    title: str = Field(description="Report display title")
    campaign_id: str = Field(description="Campaign identifier")
    created_at: str = Field(default="", description="ISO timestamp of report creation")
    spec: ResearchReportSpecification | None = Field(
        default=None, description="Optional report specification"
    )
    executive_summary: str = Field(description="High-level synthesis summary")
    research_questions: list[ResearchQuestion] = Field(
        default_factory=list, description="Evaluated research questions"
    )
    methodology_summary: str = Field(description="Methodological framework summary")
    tables: list[BenchmarkTable] = Field(
        default_factory=list, description="Synthesized benchmark tables"
    )
    figures: list[FigureSpecification] = Field(
        default_factory=list, description="Synthesized figure specifications"
    )
    profiles: list[RepresentationProfile] = Field(
        default_factory=list, description="Representation profiles"
    )
    findings: list[ResearchFinding] = Field(
        default_factory=list, description="Evidence-backed scientific findings"
    )
    evidence_gaps: list[EvidenceGap] = Field(
        default_factory=list, description="Identified evidence gaps"
    )
    limitations: list[str] = Field(
        default_factory=list, description="Platform-wide scientific limitations"
    )
    reproducibility_manifest: ReproducibilityManifest = Field(
        description="Complete reproducibility manifest"
    )
    warnings: list[str] = Field(
        default_factory=list, description="Diagnostic and scientific warnings"
    )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
