"""Unit tests for research benchmark foundations (Commit 1)."""

from __future__ import annotations

import pytest

from prism.benchmarking.aggregation import (
    aggregate_repeated_seeds,
)
from prism.benchmarking.comparisons import (
    audit_comparison_control,
    compute_pairwise_comparison,
)
from prism.benchmarking.contracts import (
    BenchmarkCampaign,
    BenchmarkResultCell,
    MetricDefinition,
    ResearchQuestion,
    compute_campaign_fingerprint,
)
from prism.benchmarking.enums import (
    ComparisonControlStatus,
    FactorID,
    MetricCategory,
    MetricDirection,
    ResultStatus,
)
from prism.benchmarking.matrices import (
    build_benchmark_matrix,
    build_benchmark_table,
)
from prism.benchmarking.registry import (
    FactorRegistry,
    MetricRegistry,
    canonical_metric_registry,
)
from prism.benchmarking.store import BenchmarkResultStore
from prism.core.errors import ValidationError
from prism.experiments.reporting import (
    ArchitectureComparisonReport,
)


def test_factor_and_metric_registry() -> None:
    """Verify factor and metric registries enforce canonical definitions."""
    factors = FactorRegistry.list_factors()
    assert FactorID.ARCHITECTURE in factors
    assert FactorID.PRETRAINING_OBJECTIVE in factors

    arch_info = FactorRegistry.get_factor_info(FactorID.ARCHITECTURE)
    assert "Visual Architecture" in arch_info["display_name"]

    reg = MetricRegistry()
    assert reg.has("accuracy")
    assert reg.has("ece")
    assert reg.has("linear_probe_accuracy")

    acc_def = reg.get("accuracy")
    assert acc_def.direction == MetricDirection.HIGHER_IS_BETTER
    assert acc_def.category == MetricCategory.PERFORMANCE

    # Conflicting metric redefinition should raise ValidationError
    conflicting = MetricDefinition(
        metric_id="accuracy",
        display_name="Conflicting Accuracy",
        category=MetricCategory.GEOMETRY,
        direction=MetricDirection.LOWER_IS_BETTER,
    )
    with pytest.raises(ValidationError, match="Conflicting metric"):
        reg.register(conflicting)


def test_benchmark_campaign_and_question_contracts() -> None:
    """Verify BenchmarkCampaign and ResearchQuestion schema contracts."""
    q = ResearchQuestion(
        question_id="rq1_pretraining_transfer",
        natural_language_question=(
            "How does pretraining objective affect linear probe accuracy?"
        ),
        independent_variables=["pretraining_objective"],
        independent_values=["supervised", "simclr", "reconstruction"],
        dependent_metrics=["linear_probe_accuracy"],
        controlled_factors={"architecture": "resnet", "dataset": "cifar10"},
    )
    assert q.question_id == "rq1_pretraining_transfer"

    fp = compute_campaign_fingerprint(
        campaign_id="camp_1",
        architectures=["resnet", "vit"],
        objectives=["supervised", "simclr"],
        datasets=["cifar10"],
        tasks=["classification"],
        seeds=[42, 100],
        budgets=[0.1, 1.0],
    )
    assert len(fp) == 64

    campaign = BenchmarkCampaign(
        campaign_id="camp_1",
        title="PRISM Core Pretraining Benchmark",
        description="Comprehensive evaluation across architectures and objectives",
        research_questions=[q],
        architectures=["resnet", "vit"],
        objectives=["supervised", "simclr"],
        datasets=["cifar10"],
        tasks=["classification"],
        seeds=[42, 100],
        budgets=[0.1, 1.0],
        fingerprint=fp,
    )
    assert campaign.campaign_id == "camp_1"
    serialized = campaign.to_dict()
    assert serialized["fingerprint"] == fp


def test_result_store_registration_and_deduplication() -> None:
    """Verify BenchmarkResultStore registration and deduplication."""
    store = BenchmarkResultStore()

    cell1 = BenchmarkResultCell(
        result_id="res_1",
        experiment_id="exp_resnet_sup",
        experiment_fingerprint="fp_resnet_sup",
        metric_id="accuracy",
        value=0.852,
        status=ResultStatus.OBSERVED,
        seed=42,
        factors={"architecture": "resnet", "pretraining_objective": "supervised"},
    )
    store.register_cell(cell1)
    assert store.count() == 1

    # Exact duplicate registration should silently deduplicate
    cell1_dup = BenchmarkResultCell(
        result_id="res_1_dup",
        experiment_id="exp_resnet_sup",
        experiment_fingerprint="fp_resnet_sup",
        metric_id="accuracy",
        value=0.852,
        status=ResultStatus.OBSERVED,
        seed=42,
        factors={"architecture": "resnet", "pretraining_objective": "supervised"},
    )
    store.register_cell(cell1_dup)
    assert store.count() == 1

    # Conflicting value for same identity should raise RESULT_PROVENANCE_CONFLICT
    cell1_conflict = BenchmarkResultCell(
        result_id="res_1_conflict",
        experiment_id="exp_resnet_sup",
        experiment_fingerprint="fp_resnet_sup",
        metric_id="accuracy",
        value=0.910,
        status=ResultStatus.OBSERVED,
        seed=42,
        factors={"architecture": "resnet", "pretraining_objective": "supervised"},
    )
    with pytest.raises(ValidationError, match="RESULT_PROVENANCE_CONFLICT"):
        store.register_cell(cell1_conflict)


def test_adapter_architecture_comparison_report() -> None:
    """Verify adapter correctly transforms ArchitectureComparisonReport into cells."""
    from prism.core.enums import ModelFamily, RunStatus
    from prism.experiments.architecture import ExperimentFactorAudit
    from prism.experiments.reporting import ArchitectureMetricSummary
    from prism.experiments.suites import ComparisonMode

    sum_resnet = ArchitectureMetricSummary(
        experiment_id="exp_resnet",
        model_family=ModelFamily.RESNET,
        architecture="resnet",
        parameter_count=11200000,
        test_accuracy=0.885,
        final_validation_loss=0.342,
        training_status=RunStatus.COMPLETED,
    )
    sum_vit = ArchitectureMetricSummary(
        experiment_id="exp_vit",
        model_family=ModelFamily.VISION_TRANSFORMER,
        architecture="vit",
        parameter_count=15400000,
        test_accuracy=0.871,
        final_validation_loss=0.368,
        training_status=RunStatus.COMPLETED,
    )
    report = ArchitectureComparisonReport(
        suite_id="suite_arch_test",
        suite_fingerprint="fp_suite_arch_test",
        research_question="CNN vs ResNet vs ViT",
        comparison_mode=ComparisonMode.STRICT_CONTROLLED,
        generated_at="2026-09-06T00:00:00Z",
        factor_audit=ExperimentFactorAudit(
            experiment_ids=["exp_resnet", "exp_vit"],
            constant_factors={},
            varied_factors={},
            unexpected_differences={},
            warnings=[],
            is_strictly_controlled=True,
        ),
        metric_summaries=[sum_resnet, sum_vit],
    )

    store = BenchmarkResultStore()
    cells = store.register_report(report)
    assert len(cells) == 6  # 3 metrics (acc, loss, params) * 2 models

    acc_cells = store.query(metric_id="accuracy")
    assert len(acc_cells) == 2
    assert any(
        c.factors.get("architecture") == "resnet" and c.value == 0.885
        for c in acc_cells
    )


def test_repeated_seed_aggregation_and_single_seed_warning() -> None:
    """Verify statistical aggregation across seeds with strict n=1 warning semantics."""
    # Case A: Single seed (n=1)
    cell_single = BenchmarkResultCell(
        result_id="res_s1",
        experiment_id="exp_1",
        experiment_fingerprint="fp_1",
        metric_id="accuracy",
        value=0.80,
        status=ResultStatus.OBSERVED,
        seed=42,
        factors={"architecture": "resnet", "pretraining_objective": "supervised"},
    )
    agg_single = aggregate_repeated_seeds([cell_single])
    assert agg_single.sample_count == 1
    assert agg_single.mean == 0.80
    assert agg_single.std is None  # Strict invariant: std is None for N=1
    assert any("SINGLE_SEED_RESULT" in w for w in agg_single.warnings)

    # Case B: Multi-seed (n=3)
    cells_multi = [
        BenchmarkResultCell(
            result_id=f"res_m_{seed}",
            experiment_id="exp_multi",
            experiment_fingerprint="fp_multi",
            metric_id="accuracy",
            value=val,
            status=ResultStatus.OBSERVED,
            seed=seed,
            factors={"architecture": "resnet", "pretraining_objective": "supervised"},
        )
        for seed, val in [(42, 0.80), (100, 0.82), (2024, 0.84)]
    ]
    agg_multi = aggregate_repeated_seeds(cells_multi)
    assert agg_multi.sample_count == 3
    assert agg_multi.mean == pytest.approx(0.82, abs=1e-4)
    assert agg_multi.std is not None and agg_multi.std > 0.0
    assert agg_multi.min_value == 0.80
    assert agg_multi.max_value == 0.84
    assert agg_multi.median_value == 0.82


def test_comparison_control_audit_and_pairwise() -> None:
    """Verify factor control auditing and direction-aware pairwise comparison."""
    factors_a = {
        "dataset": "cifar10",
        "task": "classification",
        "data_budget": 1.0,
        "seed": 42,
        "architecture": "resnet",
    }
    factors_b = {
        "dataset": "cifar10",
        "task": "classification",
        "data_budget": 1.0,
        "seed": 42,
        "architecture": "vit",
    }

    audit = audit_comparison_control(factors_a, factors_b)
    assert audit.status == ComparisonControlStatus.STRICTLY_CONTROLLED
    assert not audit.mismatches

    cell_a = BenchmarkResultCell(
        result_id="cell_a",
        experiment_id="exp_a",
        experiment_fingerprint="fp_a",
        metric_id="accuracy",
        value=0.88,
        status=ResultStatus.OBSERVED,
        factors=factors_a,
    )
    cell_b = BenchmarkResultCell(
        result_id="cell_b",
        experiment_id="exp_b",
        experiment_fingerprint="fp_b",
        metric_id="accuracy",
        value=0.85,
        status=ResultStatus.OBSERVED,
        factors=factors_b,
    )

    pair = compute_pairwise_comparison(
        cell_a=cell_a,
        cell_b=cell_b,
        metric_def=canonical_metric_registry.get("accuracy"),
    )
    assert pair.absolute_delta == pytest.approx(-0.03, abs=1e-4)
    assert pair.favorable_candidate == "A"
    assert pair.control_audit.status == ComparisonControlStatus.STRICTLY_CONTROLLED


def test_benchmark_matrix_and_table_generation() -> None:
    """Verify generation of BenchmarkMatrix and structured BenchmarkTable."""
    store = BenchmarkResultStore()
    objectives = ["supervised", "simclr", "reconstruction"]
    archs = ["cnn", "resnet"]

    for obj in objectives:
        for arch in archs:
            store.register_cell(
                BenchmarkResultCell(
                    result_id=f"cell_{obj}_{arch}",
                    experiment_id=f"exp_{obj}_{arch}",
                    experiment_fingerprint=f"fp_{obj}_{arch}",
                    metric_id="linear_probe_accuracy",
                    value=0.70 if obj == "supervised" else 0.65,
                    status=ResultStatus.OBSERVED,
                    factors={
                        "pretraining_objective": obj,
                        "architecture": arch,
                        "task": "transfer",
                    },
                )
            )

    matrix = build_benchmark_matrix(
        store=store,
        row_factor="pretraining_objective",
        column_factor="architecture",
        metric_id="linear_probe_accuracy",
    )
    assert len(matrix.row_values) == 3
    assert len(matrix.column_values) == 2
    assert matrix.cells["supervised"]["resnet"] is not None

    table = build_benchmark_table(
        matrix=matrix,
        metric_def=canonical_metric_registry.get("linear_probe_accuracy"),
        title="Linear Probe Accuracy Matrix",
    )
    assert table.table_id == f"tbl_{matrix.matrix_id}"
    assert len(table.rows) == 3
    assert "resnet" in table.rows[0]
