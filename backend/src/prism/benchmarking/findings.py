"""Research findings generation grounded in observed evidence."""

from __future__ import annotations

from typing import Any

from prism.benchmarking.aggregation import aggregate_repeated_seeds
from prism.benchmarking.comparisons import audit_comparison_control
from prism.benchmarking.contracts import (
    BenchmarkCampaign,
    MetricDefinition,
    ResearchFinding,
)
from prism.benchmarking.enums import (
    ComparisonControlStatus,
    EvidenceStrength,
    MetricDirection,
    ResultStatus,
)
from prism.benchmarking.registry import canonical_metric_registry
from prism.benchmarking.store import BenchmarkResultStore


def generate_research_findings(
    campaign: BenchmarkCampaign,
    store: BenchmarkResultStore,
    metric_registry: Any = canonical_metric_registry,
) -> list[ResearchFinding]:
    """Generate structured scientific findings grounded in store cells."""
    findings: list[ResearchFinding] = []
    finding_idx = 0

    rqs = campaign.research_questions
    if not rqs:
        from prism.benchmarking.contracts import ResearchQuestion

        rqs = [
            ResearchQuestion(
                question_id="rq_arch_accuracy",
                natural_language_question=(
                    "How do visual representation architectures"
                    " compare on task accuracy?"
                ),
                independent_variables=["architecture"],
                dependent_metrics=["accuracy"],
                controlled_factors={"dataset": "cifar10", "task": "classification"},
            ),
            ResearchQuestion(
                question_id="rq_obj_robustness",
                natural_language_question=(
                    "How do pretraining objectives affect representation robustness?"
                ),
                independent_variables=["pretraining_objective"],
                dependent_metrics=["robustness_accuracy_drop"],
                controlled_factors={"dataset": "cifar10", "task": "classification"},
            ),
        ]

    for q in rqs:
        for metric_id in q.dependent_metrics:
            m_def = (
                metric_registry.get(metric_id)
                if metric_registry.has(metric_id)
                else MetricDefinition(
                    metric_id=metric_id,
                    display_name=metric_id.replace("_", " ").title(),
                    category=MetricDefinition.model_fields["category"].default,
                )
            )

            cells = store.query(metric_id=metric_id)
            if not cells:
                finding_idx += 1
                findings.append(
                    ResearchFinding(
                        finding_id=f"find_{finding_idx}",
                        research_question_id=q.question_id,
                        statement=(
                            f"No experimental observations currently exist in the "
                            f"benchmark store for metric '{metric_id}'."
                        ),
                        supporting_result_ids=[],
                        comparison_audit=None,
                        effect_size_delta=None,
                        scope={
                            "research_question_id": q.question_id,
                            "metric_id": metric_id,
                        },
                        caveats=["Cannot evaluate claim without completed runs."],
                        evidence_strength=EvidenceStrength.INSUFFICIENT_EVIDENCE,
                    )
                )
                continue

            var_name = (
                q.independent_variables[0]
                if q.independent_variables
                else "architecture"
            )
            groups: dict[str, list[Any]] = {}
            for c in cells:
                if c.status == ResultStatus.OBSERVED and c.value is not None:
                    val_str = str(c.factors.get(var_name, "default"))
                    groups.setdefault(val_str, []).append(c)

            if len(groups) >= 2:
                agg_groups: dict[str, Any] = {
                    k: aggregate_repeated_seeds(v) for k, v in groups.items()
                }
                sorted_by_mean = sorted(
                    agg_groups.items(),
                    key=lambda x: (
                        -x[1].mean
                        if m_def.direction == MetricDirection.HIGHER_IS_BETTER
                        else x[1].mean
                    ),
                )

                best_variant, best_agg = sorted_by_mean[0]
                second_variant, second_agg = sorted_by_mean[1]
                delta = abs(best_agg.mean - second_agg.mean)

                all_single = all(agg.sample_count == 1 for agg in agg_groups.values())
                all_multi = all(agg.sample_count >= 3 for agg in agg_groups.values())

                sample_cell_a = groups[best_variant][0]
                sample_cell_b = groups[second_variant][0]
                ctrl_factors = [
                    k for k in sample_cell_a.factors if k != var_name and k != "seed"
                ]
                audit = audit_comparison_control(
                    sample_cell_a.factors,
                    sample_cell_b.factors,
                    expected_equal=ctrl_factors,
                )

                if (
                    audit.status == ComparisonControlStatus.STRICTLY_CONTROLLED
                    and all_multi
                ):
                    strength = EvidenceStrength.SUPPORTED_BY_REPEATED_RUNS
                elif (
                    audit.status == ComparisonControlStatus.STRICTLY_CONTROLLED
                    and all_single
                ):
                    strength = EvidenceStrength.SUPPORTED_BY_SINGLE_RUN
                elif audit.status == ComparisonControlStatus.DESCRIPTIVE_ONLY:
                    strength = EvidenceStrength.DESCRIPTIVE_ONLY
                else:
                    strength = EvidenceStrength.SUPPORTED_BY_SINGLE_RUN

                finding_idx += 1
                supp_ids = [
                    c.result_id for var_cells in groups.values() for c in var_cells
                ]

                claim_text = (
                    f"Under controlled evaluation of {var_name}, '{best_variant}' "
                    f"achieved the most favorable {m_def.display_name} "
                    f"({best_agg.mean:.3f}) compared to '{second_variant}' "
                    f"({second_agg.mean:.3f}), with a delta of {delta:.3f}."
                )

                limitations_list = [
                    (
                        "Grounding restricted to evaluated dataset: "
                        f"{sample_cell_a.factors.get('dataset', 'synthetic')}."
                    ),
                    m_def.methodological_notes
                    or "Evaluated in controlled PRISM environment.",
                ]
                if all_single:
                    limitations_list.append(
                        "SINGLE_SEED_WARNING: Comparisons reflect single-seed "
                        "evaluations (N=1)."
                    )

                findings.append(
                    ResearchFinding(
                        finding_id=f"find_{finding_idx}",
                        research_question_id=q.question_id,
                        statement=claim_text,
                        supporting_result_ids=supp_ids,
                        comparison_audit=audit,
                        effect_size_delta=delta,
                        scope={
                            "research_question_id": q.question_id,
                            "independent_variable": var_name,
                            "metric_id": metric_id,
                            "variants_compared": list(groups.keys()),
                            "sample_counts": {
                                k: agg.sample_count for k, agg in agg_groups.items()
                            },
                        },
                        caveats=limitations_list,
                        evidence_strength=strength,
                    )
                )

    return findings
