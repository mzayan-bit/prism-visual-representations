"""Unit tests for JSON, Markdown, and CSV export routines."""

import json
from typing import Any

from prism.benchmarking.contracts import (
    BenchmarkCampaign,
    BenchmarkResultCell,
    ResearchReportSpecification,
)
from prism.benchmarking.enums import ResultStatus
from prism.benchmarking.export import (
    export_matrix_to_csv,
    export_report_to_json,
    export_report_to_markdown,
    export_table_to_csv,
)
from prism.benchmarking.matrices import build_benchmark_matrix, build_benchmark_table
from prism.benchmarking.reporting import compile_prism_research_report
from prism.benchmarking.store import BenchmarkResultStore


def _create_sample_report() -> tuple[Any, Any, Any]:
    store = BenchmarkResultStore()
    campaign = BenchmarkCampaign(
        campaign_id="camp_exp",
        title="Export Campaign",
        description="Testing export",
        architectures=["resnet", "vit"],
        objectives=["supervised"],
        seeds=[42],
    )
    for arch in ("resnet", "vit"):
        store.register_cell(
            BenchmarkResultCell(
                result_id=f"exp_{arch}",
                experiment_id=f"exp_{arch}",
                experiment_fingerprint=f"fp_{arch}",
                metric_id="accuracy",
                value=0.88 if arch == "vit" else 0.85,
                status=ResultStatus.OBSERVED,
                seed=42,
                source_report_type="test",
                source_run_id=f"run_{arch}",
                factors={
                    "architecture": arch,
                    "pretraining_objective": "supervised",
                    "seed": 42,
                },
            )
        )
    spec = ResearchReportSpecification(
        report_id="spec_export",
        title="Export Test Report",
        campaign_id=campaign.campaign_id,
    )
    report = compile_prism_research_report(spec, campaign, store)
    return report, store, campaign


def test_export_report_to_json() -> None:
    report, _, _ = _create_sample_report()
    json_str = export_report_to_json(report)
    parsed = json.loads(json_str)

    assert parsed["report_id"] == report.report_id
    assert parsed["title"] == report.title
    assert "executive_summary" in parsed


def test_export_report_to_markdown() -> None:
    report, _, _ = _create_sample_report()
    md_str = export_report_to_markdown(report)

    assert md_str.startswith(f"# {report.title}")
    assert "## Executive Summary" in md_str
    assert "## Experimental Methodology" in md_str
    assert "## Reproducibility Appendix" in md_str


def test_export_table_and_matrix_to_csv() -> None:
    _, store, _ = _create_sample_report()
    mat = build_benchmark_matrix(
        store,
        metric_id="accuracy",
        row_factor="pretraining_objective",
        column_factor="architecture",
        row_values=["supervised"],
        column_values=["resnet", "vit"],
    )
    tbl = build_benchmark_table(mat)

    csv_tbl = export_table_to_csv(tbl)
    assert "pretraining_objective,resnet,vit" in csv_tbl
    assert "supervised,0.8500,0.8800" in csv_tbl

    csv_mat = export_matrix_to_csv(mat)
    assert "pretraining_objective,resnet,vit" in csv_mat
    assert "supervised,0.85,0.88" in csv_mat
