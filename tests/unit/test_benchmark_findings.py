"""Unit tests for scientific findings generator and evidence strength grading."""

from prism.benchmarking.contracts import (
    BenchmarkCampaign,
    BenchmarkResultCell,
    ResearchQuestion,
)
from prism.benchmarking.enums import EvidenceStrength, ResultStatus
from prism.benchmarking.findings import generate_research_findings
from prism.benchmarking.store import BenchmarkResultStore


def test_generate_research_findings() -> None:
    store = BenchmarkResultStore()
    rq = ResearchQuestion(
        question_id="rq1_arch_acc",
        natural_language_question=(
            "Does ViT outperform ResNet in classification accuracy?"
        ),
        independent_variables=["architecture"],
        independent_values=["vit", "resnet"],
        dependent_metrics=["accuracy"],
        controlled_factors={"pretraining_objective": "supervised"},
    )
    campaign = BenchmarkCampaign(
        campaign_id="camp_findings",
        title="Findings Campaign",
        description="Testing findings",
        research_questions=[rq],
        architectures=["resnet", "vit"],
        objectives=["supervised"],
        seeds=[42, 100, 2024],
    )

    for arch, base_val in [("resnet", 0.85), ("vit", 0.89)]:
        for s in (42, 100, 2024):
            store.register_cell(
                BenchmarkResultCell(
                    result_id=f"f_{arch}_{s}",
                    experiment_id=f"exp_{arch}_{s}",
                    experiment_fingerprint=f"fp_{arch}_{s}",
                    metric_id="accuracy",
                    value=base_val + (s - 100) * 0.0001,
                    status=ResultStatus.OBSERVED,
                    seed=s,
                    source_report_type="test",
                    source_run_id=f"run_{arch}_{s}",
                    factors={
                        "architecture": arch,
                        "pretraining_objective": "supervised",
                        "seed": s,
                    },
                )
            )

    findings = generate_research_findings(campaign, store)
    assert len(findings) > 0
    f1 = findings[0]
    assert f1.research_question_id == rq.question_id
    assert f1.evidence_strength == EvidenceStrength.SUPPORTED_BY_REPEATED_RUNS
    assert f1.effect_size_delta is not None
    assert f1.effect_size_delta > 0
    assert len(f1.supporting_result_ids) == 6
    assert len(f1.caveats) > 0
