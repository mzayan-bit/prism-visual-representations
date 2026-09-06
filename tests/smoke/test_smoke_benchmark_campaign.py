"""Comprehensive end-to-end smoke test for PRISM Cross-Paradigm Benchmark Campaign.

Validates the full benchmark orchestration, evidence synthesis, report generation,
and publication export pipeline across all evaluated dimensions.
"""

import json

from prism.benchmarking.adapters import adapt_multimodal_alignment_report
from prism.benchmarking.aggregation import aggregate_repeated_seeds
from prism.benchmarking.comparisons import (
    audit_comparison_control,
    compute_pairwise_comparison,
)
from prism.benchmarking.contracts import (
    BenchmarkCampaign,
    BenchmarkResultCell,
    ResearchReportSpecification,
)
from prism.benchmarking.coverage import (
    build_coverage_matrix,
    compute_campaign_coverage_summary,
    detect_evidence_gaps,
    generate_missing_experiment_plan,
)
from prism.benchmarking.enums import EvidenceStrength, ResultStatus
from prism.benchmarking.export import (
    export_report_to_json,
    export_report_to_markdown,
    export_table_to_csv,
)
from prism.benchmarking.findings import generate_research_findings
from prism.benchmarking.matrices import build_benchmark_matrix, build_benchmark_table
from prism.benchmarking.reporting import compile_prism_research_report
from prism.benchmarking.runner import BenchmarkCampaignRunner
from prism.benchmarking.service import BenchmarkService
from prism.benchmarking.store import BenchmarkResultStore
from prism.benchmarking.synthesis import (
    compute_pareto_front,
    extract_representation_profile,
    extract_tradeoff_pairs,
    synthesize_cross_architecture,
    synthesize_cross_objective,
)
from prism.multimodal.contracts import (
    CrossModalRetrievalSummary,
    MultimodalCollapseSummary,
    ZeroShotClassificationSummary,
)
from prism.multimodal.reports import VisionLanguageRepresentationReport


def test_smoke_end_to_end_benchmark_campaign() -> None:
    # 1. Initialize Store and Campaign
    store = BenchmarkResultStore()
    campaign = BenchmarkCampaign(
        campaign_id="prism_smoke_campaign_v1",
        title="PRISM Cross-Paradigm Benchmark Smoke Test",
        description=(
            "End-to-end validation of the PRISM research benchmark orchestration layer"
        ),
        architectures=["resnet", "vit", "cnn"],
        objectives=["supervised", "simclr", "reconstruction", "vision_language"],
        datasets=["cifar10"],
        tasks=["classification"],
        seeds=[42, 100, 2024],
        budgets=[1.0],
    )

    # 2. Populate realistic multi-seed observations
    for arch in ("resnet", "vit", "cnn"):
        for obj in ("supervised", "simclr", "reconstruction"):
            for seed in (42, 100, 2024):
                # Primary clean accuracy
                acc_val = 0.88 if arch == "resnet" else 0.90 if arch == "vit" else 0.82
                if obj == "supervised":
                    acc_val += 0.02
                elif obj == "reconstruction":
                    acc_val -= 0.04

                store.register_cell(
                    BenchmarkResultCell(
                        result_id=f"res_{arch}_{obj}_acc_{seed}",
                        experiment_id=f"exp_{arch}_{obj}_{seed}",
                        experiment_fingerprint=f"fp_{arch}_{obj}_{seed}",
                        metric_id="accuracy",
                        value=acc_val + (seed - 100) * 0.0001,
                        status=ResultStatus.OBSERVED,
                        seed=seed,
                        source_report_type="supervised_benchmark",
                        source_run_id=f"run_{arch}_{obj}_{seed}",
                        factors={
                            "architecture": arch,
                            "pretraining_objective": obj,
                            "dataset": "cifar10",
                            "task": "classification",
                            "seed": seed,
                            "data_budget": 1.0,
                        },
                    )
                )

                # Robustness corruption drop
                drop_val = 0.12 if arch == "resnet" else 0.10 if arch == "vit" else 0.16
                if obj == "simclr":
                    drop_val -= 0.03
                store.register_cell(
                    BenchmarkResultCell(
                        result_id=f"res_{arch}_{obj}_drop_{seed}",
                        experiment_id=f"exp_{arch}_{obj}_{seed}",
                        experiment_fingerprint=f"fp_{arch}_{obj}_{seed}",
                        metric_id="robustness_accuracy_drop",
                        value=drop_val + (seed - 100) * 0.0001,
                        status=ResultStatus.OBSERVED,
                        seed=seed,
                        source_report_type="robustness_benchmark",
                        source_run_id=f"run_{arch}_{obj}_{seed}",
                        factors={
                            "architecture": arch,
                            "pretraining_objective": obj,
                            "dataset": "cifar10",
                            "task": "classification",
                            "seed": seed,
                            "data_budget": 1.0,
                        },
                    )
                )

    # 3. Adapt domain-specific report (e.g. Multimodal)
    ret_summary = CrossModalRetrievalSummary(
        image_to_text_r1=0.85,
        image_to_text_r3=0.92,
        image_to_text_r5=0.96,
        image_to_text_mrr=0.89,
        text_to_image_r1=0.82,
        text_to_image_r3=0.90,
        text_to_image_r5=0.95,
        text_to_image_mrr=0.87,
        sample_count=100,
        candidate_count=100,
    )
    zs_summary = ZeroShotClassificationSummary(
        prompt_template="a photo of a {}",
        class_count=10,
        accuracy=0.79,
        top_3_accuracy=0.93,
        per_class_accuracy={"cat": 0.8, "dog": 0.78},
        confusion_matrix=[],
        class_names=["cat", "dog"],
    )
    col_summary = MultimodalCollapseSummary(
        visual_dim_variance=1.2,
        visual_feature_std=0.8,
        visual_pairwise_similarity=0.4,
        text_dim_variance=1.1,
        text_feature_std=0.75,
        text_pairwise_similarity=0.35,
        matched_similarity=0.88,
        unmatched_similarity=0.12,
        similarity_gap=0.76,
        is_collapsed=False,
    )
    mm_report = VisionLanguageRepresentationReport(
        experiment_id="prism_vit_vl_s42",
        visual_family="vit",
        visual_architecture="vit",
        text_dim=64,
        shared_dim=64,
        temperature=0.07,
        seed=42,
        dataset_fingerprint="coco_smoke",
        final_loss=0.25,
        image_to_text_loss=0.24,
        text_to_image_loss=0.26,
        matched_similarity=0.88,
        unmatched_similarity=0.12,
        similarity_gap=0.76,
        training_history=[],
        retrieval_summary=ret_summary,
        zero_shot_summary=zs_summary,
        prompt_sensitivity={"template_variance": 0.015},
        explained_variance_ratio=[0.5, 0.3, 0.2],
        mean_paired_distance=0.3,
        mean_paired_cosine=0.85,
        centroid_alignments=[],
        collapse_summary=col_summary,
        robustness_summary={},
    )
    mm_cells = adapt_multimodal_alignment_report(mm_report)
    for c in mm_cells:
        store.register_cell(c)

    # 4. Multi-seed Statistical Aggregation
    resnet_acc_cells = store.query_cells(
        metric_id="accuracy",
        factors={"architecture": "resnet", "pretraining_objective": "supervised"},
    )
    agg_result = aggregate_repeated_seeds(resnet_acc_cells)
    assert agg_result.sample_count == 3
    assert agg_result.mean is not None
    assert agg_result.std is not None
    assert agg_result.std >= 0.0
    assert not any("SINGLE_SEED_RESULT" in w for w in agg_result.warnings)

    # 5. Controlled Factor Comparison and Audit
    audit = audit_comparison_control(
        factor_a={"architecture": "resnet", "dataset": "cifar10", "seed": 42},
        factor_b={"architecture": "vit", "dataset": "cifar10", "seed": 42},
    )
    assert audit.is_strictly_controlled is True
    assert "architecture" in audit.varied_factors

    pair_res = compute_pairwise_comparison(
        store=store,
        metric_id="accuracy",
        factor_a={"architecture": "resnet", "pretraining_objective": "supervised"},
        factor_b={"architecture": "vit", "pretraining_objective": "supervised"},
    )
    assert pair_res.absolute_delta is not None
    assert pair_res.control_audit.is_strictly_controlled is True

    # 6. Matrix & Table Generation
    mat = build_benchmark_matrix(
        store,
        metric_id="accuracy",
        row_factor="pretraining_objective",
        column_factor="architecture",
        row_values=["supervised", "simclr", "reconstruction"],
        column_values=["resnet", "vit", "cnn"],
    )
    tbl = build_benchmark_table(mat)
    assert len(tbl.rows) == 3
    csv_str = export_table_to_csv(tbl)
    assert "pretraining_objective,resnet,vit,cnn" in csv_str

    # 7. Coverage Matrix & Gap Planning
    cov_mat = build_coverage_matrix(
        store,
        row_factor="pretraining_objective",
        column_factor="architecture",
        row_values=campaign.objectives,
        column_values=campaign.architectures,
    )
    assert cov_mat.row_factor == "pretraining_objective"
    cov_summary = compute_campaign_coverage_summary(campaign, store)
    assert cov_summary.completed_experiments_count > 0
    assert cov_summary.missing_experiments_count > 0

    gaps = detect_evidence_gaps(campaign, store)
    missing_plan = generate_missing_experiment_plan(campaign, gaps)
    assert len(missing_plan.missing_experiments) > 0

    # 8. Representation Profiles & Pareto Frontiers
    profile = extract_representation_profile(
        store, architecture="resnet", objective="supervised"
    )
    assert profile.semantic_performance is not None
    assert profile.robustness is not None

    pareto = compute_pareto_front(
        store, metric_ids=["accuracy", "robustness_accuracy_drop"]
    )
    assert len(pareto.non_dominated_experiment_ids) > 0

    tradeoffs = extract_tradeoff_pairs(
        store, x_metric_id="accuracy", y_metric_id="robustness_accuracy_drop"
    )
    assert len(tradeoffs) > 0

    arch_synth = synthesize_cross_architecture(store, campaign.architectures)
    obj_synth = synthesize_cross_objective(store, campaign.objectives)
    assert "resnet" in arch_synth
    assert "supervised" in obj_synth

    # 9. Grounded Findings Generation
    findings = generate_research_findings(campaign, store)
    assert len(findings) > 0
    for f in findings:
        assert f.evidence_strength in (
            EvidenceStrength.SUPPORTED_BY_REPEATED_RUNS,
            EvidenceStrength.SUPPORTED_BY_SINGLE_RUN,
            EvidenceStrength.DESCRIPTIVE_ONLY,
            EvidenceStrength.INSUFFICIENT_EVIDENCE,
        )
        assert len(f.caveats) > 0

    # 10. Research Report Compilation and Export
    spec = ResearchReportSpecification(
        report_id="spec_smoke",
        title="PRISM End-to-End Synthesis Report",
        campaign_id=campaign.campaign_id,
    )
    report = compile_prism_research_report(spec, campaign, store)

    assert report.campaign_id == campaign.campaign_id
    assert len(report.tables) >= 2
    assert len(report.figures) >= 2
    assert len(report.findings) > 0
    assert len(report.reproducibility_manifest.seeds) == 3

    # Export to JSON and verify roundtrip
    json_out = export_report_to_json(report)
    parsed_json = json.loads(json_out)
    assert parsed_json["campaign_id"] == campaign.campaign_id
    assert parsed_json["report_id"] == report.report_id

    # Export to Markdown and verify contents
    md_out = export_report_to_markdown(report)
    assert "# PRISM End-to-End Synthesis Report" in md_out
    assert "## Executive Summary" in md_out
    assert "## Reproducibility Appendix" in md_out

    # 11. Benchmark Runner and Service API
    runner = BenchmarkCampaignRunner()
    dry_res = runner.dry_run(campaign=campaign, store=store)
    assert dry_res["is_dry_run"] is True

    service = BenchmarkService(store=store, campaign=campaign)
    fe_dataset = service.export_dataset_for_frontend()
    assert "campaign" in fe_dataset
    assert "benchmark_tables" in fe_dataset
    assert len(fe_dataset["all_cells"]) == len(store.all_cells())
