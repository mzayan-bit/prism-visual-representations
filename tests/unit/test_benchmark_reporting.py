"""Unit tests for publication-grade research report compiler and manifest builder."""

from prism.benchmarking.contracts import (
    BenchmarkCampaign,
    BenchmarkResultCell,
    ResearchReportSpecification,
)
from prism.benchmarking.enums import ResultStatus
from prism.benchmarking.reporting import (
    build_reproducibility_manifest,
    compile_prism_research_report,
)
from prism.benchmarking.store import BenchmarkResultStore


def test_compile_prism_research_report() -> None:
    store = BenchmarkResultStore()
    campaign = BenchmarkCampaign(
        campaign_id="camp_rep",
        title="Report Campaign",
        description="Testing report compilation",
        architectures=["resnet", "vit"],
        objectives=["supervised", "simclr"],
        seeds=[42, 100],
    )

    for arch in ("resnet", "vit"):
        for obj in ("supervised", "simclr"):
            for seed in (42, 100):
                store.register_cell(
                    BenchmarkResultCell(
                        result_id=f"rep_{arch}_{obj}_{seed}",
                        experiment_id=f"exp_{arch}_{obj}_{seed}",
                        experiment_fingerprint=f"fp_{arch}_{obj}_{seed}",
                        metric_id="accuracy",
                        value=0.85,
                        status=ResultStatus.OBSERVED,
                        seed=seed,
                        source_report_type="test",
                        source_run_id=f"run_{arch}_{obj}_{seed}",
                        factors={
                            "architecture": arch,
                            "pretraining_objective": obj,
                            "seed": seed,
                        },
                    )
                )

    spec = ResearchReportSpecification(
        report_id="spec_1",
        title="PRISM Benchmark Synthesis Report",
        campaign_id=campaign.campaign_id,
    )

    report = compile_prism_research_report(spec, campaign, store)

    assert report.report_id.startswith("rep_")
    assert report.campaign_id == campaign.campaign_id
    assert len(report.tables) > 0
    assert len(report.figures) > 0
    assert len(report.reproducibility_manifest.experiment_fingerprints) > 0
    assert 42 in report.reproducibility_manifest.seeds


def test_build_reproducibility_manifest() -> None:
    store = BenchmarkResultStore()
    campaign = BenchmarkCampaign(
        campaign_id="camp_mf",
        title="Manifest Campaign",
        description="Testing manifest",
        seeds=[42, 100],
    )
    cell = BenchmarkResultCell(
        result_id="mf_1",
        experiment_id="exp_1",
        experiment_fingerprint="fp_abc",
        metric_id="accuracy",
        value=0.9,
        status=ResultStatus.OBSERVED,
        seed=42,
        source_report_type="test",
        source_run_id="run1",
        factors={"seed": 42},
    )
    store.register_cell(cell)

    manifest = build_reproducibility_manifest(campaign, store)
    assert manifest.campaign_fingerprint == campaign.fingerprint
    assert manifest.experiment_fingerprints.get("exp_1") == "fp_abc"
    assert 42 in manifest.seeds
