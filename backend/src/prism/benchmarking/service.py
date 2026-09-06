"""Service layer coordinating benchmark queries, reporting, and frontend exports."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from prism.benchmarking.contracts import (
    BenchmarkCampaign,
    BenchmarkMatrix,
    BenchmarkTable,
    EvidenceGap,
    MissingExperimentPlan,
    ParetoAnalysisResult,
    PRISMResearchReport,
    RepresentationProfile,
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
from prism.benchmarking.findings import generate_research_findings
from prism.benchmarking.matrices import build_benchmark_matrix, build_benchmark_table
from prism.benchmarking.registry import canonical_metric_registry
from prism.benchmarking.reporting import compile_research_report
from prism.benchmarking.store import BenchmarkResultStore
from prism.benchmarking.synthesis import (
    compute_pareto_front,
    extract_representation_profile,
    extract_tradeoff_pairs,
    synthesize_cross_architecture,
    synthesize_cross_objective,
)


def create_default_prism_campaign() -> BenchmarkCampaign:
    """Instantiate standard canonical PRISM Cross-Paradigm Benchmark Campaign."""
    architectures = ["cnn", "resnet", "vit"]
    objectives = [
        "supervised",
        "simclr",
        "reconstruction",
        "vision_language",
        "scratch",
    ]
    datasets = ["cifar10", "spatial_synth", "temporal_synth", "multimodal_synth"]
    tasks = [
        "classification",
        "transfer",
        "spatial",
        "temporal",
        "multimodal",
        "robustness",
    ]
    seeds = [42, 100, 2024]
    budgets = [0.01, 0.05, 0.10, 0.25, 0.50, 1.0]

    rq1 = ResearchQuestion(
        question_id="rq1_pretraining_transfer",
        natural_language_question=(
            "How does self-supervised pretraining (SimCLR vs Reconstruction) "
            "compare to Supervised pretraining on downstream linear probe accuracy "
            "across data budgets?"
        ),
        independent_variables=["pretraining_objective", "data_budget"],
        independent_values=objectives,
        dependent_metrics=["linear_probe_accuracy", "transfer_gain"],
        controlled_factors={"dataset": "cifar10", "task": "transfer"},
        limitations=[
            "Linear probe only; non-linear fine-tuning may show different dynamics."
        ],
    )

    rq2 = ResearchQuestion(
        question_id="rq2_architecture_inductive_bias",
        natural_language_question=(
            "What is the comparative inductive bias and spatial retention of "
            "CNNs, ResNets, and ViTs under clean accuracy, corruption robustness, "
            "and dense spatial transfer?"
        ),
        independent_variables=["architecture"],
        independent_values=architectures,
        dependent_metrics=[
            "accuracy",
            "robustness_accuracy_drop",
            "detection_mean_iou",
            "segmentation_miou",
        ],
        controlled_factors={
            "pretraining_objective": "supervised",
            "dataset": "cifar10",
        },
        limitations=[
            "Evaluated at standardized scale without ImageNet-22k pretraining."
        ],
    )

    rq3 = ResearchQuestion(
        question_id="rq3_calibration_ood_tradeoff",
        natural_language_question=(
            "Does higher test accuracy correlate with improved uncertainty "
            "calibration (ECE) and out-of-distribution detection AUROC across "
            "architectures and objectives?"
        ),
        independent_variables=["architecture", "pretraining_objective"],
        independent_values=architectures,
        dependent_metrics=["accuracy", "ece", "ood_auroc"],
        controlled_factors={"dataset": "cifar10", "task": "classification"},
        limitations=["Calibration evaluated via standard temperature scaling."],
    )

    fp = compute_campaign_fingerprint(
        campaign_id="prism_canonical_phase24",
        architectures=architectures,
        objectives=objectives,
        datasets=datasets,
        tasks=tasks,
        seeds=seeds,
        budgets=budgets,
    )

    return BenchmarkCampaign(
        campaign_id="prism_canonical_phase24",
        title="PRISM Cross-Paradigm Benchmark & Synthesis Campaign",
        description=(
            "Comprehensive benchmark orchestrating all PRISM learning paradigms, "
            "architectures, robustness evaluations, uncertainty calibrations, "
            "and spatial/temporal transfer probes."
        ),
        research_questions=[rq1, rq2, rq3],
        architectures=architectures,
        objectives=objectives,
        datasets=datasets,
        tasks=tasks,
        seeds=seeds,
        budgets=budgets,
        fingerprint=fp,
    )


class BenchmarkService:
    """Main query and synthesis service for PRISM benchmarking."""

    def __init__(
        self,
        store: BenchmarkResultStore | None = None,
        campaign: BenchmarkCampaign | None = None,
    ) -> None:
        self._store = store or BenchmarkResultStore()
        self._campaign = campaign or create_default_prism_campaign()

    @property
    def store(self) -> BenchmarkResultStore:
        return self._store

    @property
    def campaign(self) -> BenchmarkCampaign:
        return self._campaign

    def get_campaign(self) -> BenchmarkCampaign:
        return self._campaign

    def get_coverage_summary(self) -> CampaignCoverageSummary:
        return compute_campaign_coverage_summary(self._campaign, self._store)

    def get_coverage_matrix(
        self,
        row_factor: str = "pretraining_objective",
        column_factor: str = "architecture",
    ) -> ExperimentCoverageMatrix:
        return compute_coverage_matrix(
            self._campaign, self._store, row_factor, column_factor
        )

    def get_matrix(
        self,
        metric_id: str,
        row_factor: str = "pretraining_objective",
        column_factor: str = "architecture",
    ) -> BenchmarkMatrix:
        return build_benchmark_matrix(
            self._store,
            row_factor=row_factor,
            column_factor=column_factor,
            metric_id=metric_id,
        )

    def get_table(
        self,
        metric_id: str,
        row_factor: str = "pretraining_objective",
        column_factor: str = "architecture",
    ) -> BenchmarkTable:
        mat = self.get_matrix(
            metric_id=metric_id, row_factor=row_factor, column_factor=column_factor
        )
        m_def = (
            canonical_metric_registry.get(metric_id)
            if canonical_metric_registry.has(metric_id)
            else None
        )
        return build_benchmark_table(mat, m_def)

    def get_profile(
        self,
        architecture: str,
        objective: str = "supervised",
    ) -> RepresentationProfile:
        return extract_representation_profile(
            self._store, architecture=architecture, pretraining_objective=objective
        )

    def get_profiles(self) -> list[RepresentationProfile]:
        profiles = []
        for a in self._campaign.architectures:
            for o in self._campaign.objectives:
                prof = extract_representation_profile(self._store, a, o)
                profiles.append(prof)
        return profiles

    def get_pareto_front(
        self,
        metric_ids: Sequence[str] = ("accuracy", "robustness_accuracy_drop", "ece"),
    ) -> ParetoAnalysisResult:
        candidates = []
        for a in self._campaign.architectures:
            for o in self._campaign.objectives:
                m_vals: dict[str, float] = {}
                for mid in metric_ids:
                    matching = self._store.query(
                        metric_id=mid,
                        factors={"architecture": a, "pretraining_objective": o},
                    )
                    if matching and matching[0].value is not None:
                        m_vals[mid] = float(matching[0].value)
                if len(m_vals) == len(metric_ids):
                    candidates.append(
                        {
                            "experiment_id": f"{a}_{o}",
                            "architecture": a,
                            "pretraining_objective": o,
                            "metrics": m_vals,
                        }
                    )
        return compute_pareto_front(candidates, metric_ids)

    def get_tradeoffs(
        self,
        metric_x: str = "accuracy",
        metric_y: str = "robustness_accuracy_drop",
    ) -> list[TradeoffPoint]:
        return extract_tradeoff_pairs(self._store, metric_x, metric_y)

    def get_findings(self) -> list[ResearchFinding]:
        return generate_research_findings(self._campaign, self._store)

    def get_evidence_gaps(self) -> list[EvidenceGap]:
        return detect_evidence_gaps(self._campaign, self._store)

    def get_missing_experiment_plan(self) -> MissingExperimentPlan:
        gaps = self.get_evidence_gaps()
        return build_missing_experiment_plan(self._campaign, gaps)

    def generate_report(
        self, spec: ResearchReportSpecification | None = None
    ) -> PRISMResearchReport:
        return compile_research_report(self._campaign, self._store, spec)

    def export_dataset_for_frontend(self) -> dict[str, Any]:
        """Export benchmark payload for offline and online Next.js frontend."""
        report = self.generate_report()
        cov_matrix = self.get_coverage_matrix("pretraining_objective", "architecture")
        arch_synth = synthesize_cross_architecture(
            self._store, self._campaign.architectures
        )
        obj_synth = synthesize_cross_objective(self._store, self._campaign.objectives)
        pareto = self.get_pareto_front()
        tradeoffs = self.get_tradeoffs()

        return {
            "campaign": self._campaign.to_dict(),
            "coverage_summary": self.get_coverage_summary().to_dict(),
            "coverage_matrix": cov_matrix.to_dict(),
            "benchmark_tables": [t.to_dict() for t in report.tables],
            "profiles": [p.to_dict() for p in report.profiles],
            "pareto_analysis": pareto.to_dict(),
            "tradeoff_analysis": tradeoffs,
            "findings": [f.to_dict() for f in report.findings],
            "evidence_gaps": [g.to_dict() for g in report.evidence_gaps],
            "missing_plan": self.get_missing_experiment_plan().to_dict(),
            "report_summary": {
                "report_id": report.report_id,
                "title": report.title,
                "executive_summary": report.executive_summary,
                "methodology_summary": report.methodology_summary,
                "manifest": report.reproducibility_manifest.to_dict(),
            },
            "architecture_synthesis": {
                arch: {mid: agg.to_dict() for mid, agg in metrics_dict.items()}
                for arch, metrics_dict in arch_synth.items()
            },
            "objective_synthesis": {
                obj: {mid: agg.to_dict() for mid, agg in metrics_dict.items()}
                for obj, metrics_dict in obj_synth.items()
            },
            "all_cells": [c.to_dict() for c in self._store.all_cells()],
        }
