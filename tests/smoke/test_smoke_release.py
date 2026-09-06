"""High-level release smoke test validating end-to-end PRISM research workflows.

Ensures that the entire platform — from representation dataset extraction and
geometry analysis to benchmark orchestration, synthesis, report compilation,
and demo generation — executes cleanly, deterministically, on CPU without side effects.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from scripts.generate_demo import generate_demo

import prism
from prism.api.geometry_service import (
    GeometryService,
    generate_observatory_demo_data,
)
from prism.benchmarking import (
    BenchmarkResultCell,
    BenchmarkResultStore,
    BenchmarkService,
    ResearchReportSpecification,
    ResultStatus,
    create_default_prism_campaign,
    export_report_to_json,
    export_report_to_markdown,
)
from prism.representations.geometry import (
    RepresentationDataset,
)
from prism.representations.reports import analyze_representation_geometry


@pytest.mark.smoke
def test_prism_public_module_imports() -> None:
    """Verify that all core PRISM subsystems import cleanly with expected metadata."""
    import importlib

    assert hasattr(prism, "__version__")
    assert hasattr(prism, "__author__")

    submodules = [
        "prism.api",
        "prism.artifacts",
        "prism.benchmarking",
        "prism.core",
        "prism.data",
        "prism.evaluation",
        "prism.experiments",
        "prism.explainability",
        "prism.models",
        "prism.multimodal",
        "prism.reconstruction",
        "prism.representations",
        "prism.robustness",
        "prism.spatial",
        "prism.ssl",
        "prism.temporal",
        "prism.training",
        "prism.transfer",
        "prism.uncertainty",
    ]
    for submod in submodules:
        mod = importlib.import_module(submod)
        assert mod is not None


@pytest.mark.smoke
def test_end_to_end_representation_and_geometry() -> None:
    """Run an end-to-end deterministic representation geometry workflow."""
    # 1. Generate observatory demo data
    demo_payload = generate_observatory_demo_data(
        num_samples=16, num_classes=2, seed=42
    )
    assert "metadata" in demo_payload
    assert "comparison" in demo_payload
    assert "layer_profiles" in demo_payload
    assert "reports" in demo_payload

    # 2. Representation Dataset & Analysis
    dataset = RepresentationDataset(
        experiment_id="smoke_release_exp",
        model_id="smoke_release_model",
        layer_name="final_hidden",
        sample_ids=[f"s_{i}" for i in range(8)],
        labels=[0, 1, 0, 1, 0, 1, 0, 1],
        vectors=[[0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i] for i in range(8)],
        feature_dim=4,
        num_samples=8,
        num_classes=2,
    )

    report = analyze_representation_geometry(dataset, k=2)
    assert report is not None
    assert report.num_samples == 8
    assert report.pca_projection is not None
    assert len(report.pca_projection.coordinates) == 8

    # 3. Geometry Service registration and retrieval
    service = GeometryService()
    service.register_report(report, budget=1.0)
    retrieved = service.get_geometry_report(
        experiment_id="smoke_release_exp",
        model_id="smoke_release_model",
        layer_name="final_hidden",
        budget=1.0,
    )
    assert retrieved is not None
    assert retrieved.experiment_id == "smoke_release_exp"


@pytest.mark.smoke
def test_cross_architecture_comparison_and_benchmarking() -> None:
    """Verify cross-architecture comparison, benchmark orchestration, and reporting."""
    # 1. Default campaign creation
    campaign = create_default_prism_campaign()
    assert len(campaign.architectures) == 3
    assert len(campaign.objectives) == 5
    assert len(campaign.research_questions) == 3

    # 2. Benchmark Store & Service
    store = BenchmarkResultStore()
    cell = BenchmarkResultCell(
        result_id="smoke_cell_1",
        experiment_id="smoke_exp_cnn",
        experiment_fingerprint="fp_smoke_123",
        metric_id="accuracy",
        value=0.85,
        status=ResultStatus.OBSERVED,
        seed=42,
        source_report_type="smoke_test",
        source_run_id="run_smoke_1",
        factors={
            "architecture": "cnn",
            "pretraining_objective": "supervised",
            "dataset": "cifar10",
            "task": "classification",
            "seed": 42,
            "data_budget": 1.0,
        },
        provenance={"dataset": "cifar10", "hardware": "CPU"},
    )
    store.register_cell(cell)

    service = BenchmarkService(store=store, campaign=campaign)
    cov = service.get_coverage_summary()
    assert cov.planned_experiments_count > 0
    assert cov.completed_experiments_count == 1

    # 3. Research Report Compilation & Export
    spec = ResearchReportSpecification(
        report_id="smoke_report_1",
        title="Smoke Research Report",
        campaign_id=campaign.campaign_id,
        selected_question_ids=["rq1_pretraining_transfer"],
    )
    report = service.generate_report(spec)
    assert "smoke_report_1" in report.report_id

    # 4. Serialization
    with tempfile.TemporaryDirectory() as tmpdir:
        json_file = Path(tmpdir) / "report.json"
        md_file = Path(tmpdir) / "report.md"
        export_report_to_json(report, json_file)
        export_report_to_markdown(report, md_file)
        assert json_file.exists()
        assert md_file.exists()
        assert md_file.stat().st_size > 0


@pytest.mark.smoke
def test_demo_generation_script_integration() -> None:
    """Verify that generate_demo executes deterministically and exports properly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "demo_artifacts"
        res = generate_demo(output_dir=out_path, seed=42, check_only=False)

        assert res["status"] == "success"
        assert res["cells_recorded"] == 810
        assert res["findings_generated"] > 0
        assert (out_path / "prism_demo_campaign.json").exists()
        assert (out_path / "prism_demo_report.json").exists()
        assert (out_path / "prism_demo_report.md").exists()
        assert (out_path / "benchmark_matrix.csv").exists()
        assert (out_path / "benchmark_table.csv").exists()
