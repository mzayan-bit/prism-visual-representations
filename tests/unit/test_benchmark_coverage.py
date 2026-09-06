"""Unit tests for coverage matrix calculation, summary, and evidence gap detection."""

from prism.benchmarking.contracts import BenchmarkCampaign, BenchmarkResultCell
from prism.benchmarking.coverage import (
    build_coverage_matrix,
    compute_campaign_coverage_summary,
    detect_evidence_gaps,
    generate_missing_experiment_plan,
)
from prism.benchmarking.enums import ResultStatus
from prism.benchmarking.store import BenchmarkResultStore


def test_coverage_matrix_and_summary() -> None:
    store = BenchmarkResultStore()
    campaign = BenchmarkCampaign(
        campaign_id="camp_test",
        title="Test Campaign",
        description="Testing coverage",
        architectures=["resnet", "vit"],
        objectives=["supervised", "simclr"],
        datasets=["cifar10"],
        tasks=["classification"],
        seeds=[42, 100],
        budgets=[1.0],
    )

    # Register only resnet supervised
    cell = BenchmarkResultCell(
        result_id="res_cov_1",
        experiment_id="exp_resnet_sup",
        experiment_fingerprint="fp1",
        metric_id="accuracy",
        value=0.88,
        status=ResultStatus.OBSERVED,
        seed=42,
        source_report_type="test",
        source_run_id="run1",
        factors={
            "architecture": "resnet",
            "pretraining_objective": "supervised",
            "dataset": "cifar10",
            "task": "classification",
            "seed": 42,
            "data_budget": 1.0,
        },
    )
    store.register_cell(cell)

    matrix = build_coverage_matrix(
        store,
        row_factor="pretraining_objective",
        column_factor="architecture",
        row_values=["supervised", "simclr"],
        column_values=["resnet", "vit"],
    )

    assert "supervised" in matrix.grid
    assert "resnet" in matrix.grid["supervised"]
    assert matrix.grid["supervised"]["resnet"]["observed"] == 1
    assert matrix.grid["supervised"]["vit"]["missing"] == 1

    summary = compute_campaign_coverage_summary(campaign, store)
    assert summary.planned_experiments_count == 8  # 2 arch * 2 obj * 2 seeds
    assert summary.completed_experiments_count == 1
    assert summary.completion_fraction == 1 / 8


def test_evidence_gaps_and_missing_plan() -> None:
    store = BenchmarkResultStore()
    campaign = BenchmarkCampaign(
        campaign_id="camp_gaps",
        title="Gap Campaign",
        description="Testing gaps",
        architectures=["resnet", "vit"],
        objectives=["supervised"],
        seeds=[42, 100],
    )

    # Register single seed only for resnet
    cell = BenchmarkResultCell(
        result_id="res_gap_1",
        experiment_id="exp_resnet_sup",
        experiment_fingerprint="fp1",
        metric_id="accuracy",
        value=0.85,
        status=ResultStatus.OBSERVED,
        seed=42,
        source_report_type="test",
        source_run_id="run1",
        factors={
            "architecture": "resnet",
            "pretraining_objective": "supervised",
            "seed": 42,
        },
    )
    store.register_cell(cell)

    gaps = detect_evidence_gaps(campaign, store)
    assert len(gaps) > 0

    plan = generate_missing_experiment_plan(campaign, gaps)
    assert plan.campaign_id == campaign.campaign_id
    assert len(plan.missing_experiments) > 0
    assert plan.estimated_work_units > 0
