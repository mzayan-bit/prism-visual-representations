"""Experiment coverage matrices, gap analysis, and missing experiment planning."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from prism.benchmarking.contracts import (
    BenchmarkCampaign,
    EvidenceGap,
    MissingExperimentPlan,
)
from prism.benchmarking.enums import ResultStatus
from prism.benchmarking.store import BenchmarkResultStore


class ExperimentCoverageMatrix(BaseModel):
    """2D grid breakdown of experimental completion across factor combinations."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    matrix_id: str
    row_factor: str
    column_factor: str
    row_values: list[str]
    column_values: list[str]
    grid: dict[str, dict[str, dict[str, int]]] = Field(
        description="Nested counts: grid[row][col][status_str] -> int"
    )
    warnings: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class CampaignCoverageSummary(BaseModel):
    """High-level completion metrics for an entire benchmark campaign."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    campaign_id: str
    planned_experiments_count: int
    completed_experiments_count: int
    partial_experiments_count: int
    failed_experiments_count: int
    missing_experiments_count: int
    not_applicable_count: int
    completion_fraction: float = Field(
        description="Completed / (Planned - NotApplicable) fraction [0.0, 1.0]"
    )
    evaluated_seeds_count: int
    warnings: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def compute_coverage_matrix(
    campaign: BenchmarkCampaign,
    store: BenchmarkResultStore,
    row_factor: str = "pretraining_objective",
    column_factor: str = "architecture",
) -> ExperimentCoverageMatrix:
    """Compute 2D coverage status breakdown for planned factor combinations."""
    row_vals = sorted(
        getattr(campaign, f"{row_factor}s", [])
        or getattr(campaign, row_factor, [])
        or ["supervised", "simclr", "reconstruction"]
    )
    col_vals = sorted(
        getattr(campaign, f"{column_factor}s", [])
        or getattr(campaign, column_factor, [])
        or ["cnn", "resnet", "vit"]
    )

    grid: dict[str, dict[str, dict[str, int]]] = {}

    for r in row_vals:
        grid[r] = {}
        for c in col_vals:
            cells = store.query(factors={row_factor: r, column_factor: c})
            status_counts: dict[str, int] = {
                ResultStatus.OBSERVED.value: 0,
                ResultStatus.AGGREGATED.value: 0,
                ResultStatus.MISSING.value: 0,
                ResultStatus.FAILED.value: 0,
                ResultStatus.NOT_APPLICABLE.value: 0,
            }
            if cells:
                for cell in cells:
                    status_counts[cell.status.value] = (
                        status_counts.get(cell.status.value, 0) + 1
                    )
            else:
                status_counts[ResultStatus.MISSING.value] = 1

            grid[r][c] = status_counts

    return ExperimentCoverageMatrix(
        matrix_id=f"cov_{row_factor}_{column_factor}_{campaign.campaign_id}",
        row_factor=row_factor,
        column_factor=column_factor,
        row_values=row_vals,
        column_values=col_vals,
        grid=grid,
    )


def compute_campaign_coverage_summary(
    campaign: BenchmarkCampaign,
    store: BenchmarkResultStore,
) -> CampaignCoverageSummary:
    """Compute summary completion statistics across campaign plan vs store."""
    planned_combos: list[dict[str, Any]] = []

    archs = campaign.architectures or ["resnet"]
    objs = campaign.objectives or ["supervised"]
    datasets = campaign.datasets or ["cifar10"]
    tasks = campaign.tasks or ["classification"]
    seeds = campaign.seeds or [42]
    budgets = campaign.budgets or [1.0]

    for arch in archs:
        for obj in objs:
            for ds in datasets:
                for task in tasks:
                    for seed in seeds:
                        for budget in budgets:
                            planned_combos.append(
                                {
                                    "architecture": arch,
                                    "pretraining_objective": obj,
                                    "dataset": ds,
                                    "task": task,
                                    "seed": seed,
                                    "data_budget": budget,
                                }
                            )

    planned_count = len(planned_combos)
    completed_count = 0
    partial_count = 0
    failed_count = 0
    missing_count = 0
    na_count = 0
    unique_seeds_observed: set[int] = set()

    for combo in planned_combos:
        # Check if combination is conceptually not applicable
        if combo["task"] == "multimodal" and combo["pretraining_objective"] not in (
            "vision_language",
            "multimodal",
        ):
            na_count += 1
            continue

        matched_cells = store.query(factors=combo)
        if not matched_cells:
            missing_count += 1
            continue

        statuses = {c.status for c in matched_cells}
        for c in matched_cells:
            if c.seed is not None:
                unique_seeds_observed.add(c.seed)

        if ResultStatus.FAILED in statuses:
            failed_count += 1
        elif all(s == ResultStatus.OBSERVED for s in statuses):
            completed_count += 1
        else:
            partial_count += 1

    denominator = max(1, planned_count - na_count)
    fraction = min(1.0, float(completed_count) / float(denominator))

    warnings: list[str] = []
    if missing_count > 0:
        warnings.append(
            f"EVIDENCE_GAP_DETECTED: {missing_count} planned experimental "
            f"factor combinations have not yet been observed."
        )

    return CampaignCoverageSummary(
        campaign_id=campaign.campaign_id,
        planned_experiments_count=planned_count,
        completed_experiments_count=completed_count,
        partial_experiments_count=partial_count,
        failed_experiments_count=failed_count,
        missing_experiments_count=missing_count,
        not_applicable_count=na_count,
        completion_fraction=fraction,
        evaluated_seeds_count=len(unique_seeds_observed),
        warnings=warnings,
    )


def detect_evidence_gaps(
    campaign: BenchmarkCampaign,
    store: BenchmarkResultStore,
) -> list[EvidenceGap]:
    """Detect missing factor combinations, single-seed cells, and missing metrics."""
    gaps: list[EvidenceGap] = []
    gap_idx = 0

    archs = campaign.architectures or ["resnet"]
    objs = campaign.objectives or ["supervised"]
    datasets = campaign.datasets or ["cifar10"]
    tasks = campaign.tasks or ["classification"]

    # 1. Missing factor combinations
    for arch in archs:
        for obj in objs:
            for ds in datasets:
                for task in tasks:
                    if task == "multimodal" and obj not in (
                        "vision_language",
                        "multimodal",
                    ):
                        continue
                    query_factors = {
                        "architecture": arch,
                        "pretraining_objective": obj,
                        "dataset": ds,
                        "task": task,
                    }
                    cells = store.query(factors=query_factors)
                    if not cells:
                        gap_idx += 1
                        q_id = (
                            campaign.research_questions[0].question_id
                            if campaign.research_questions
                            else "rq_general"
                        )
                        gaps.append(
                            EvidenceGap(
                                gap_id=f"gap_missing_{gap_idx}",
                                research_question_id=q_id,
                                missing_factor_combination=query_factors,
                                missing_metric_id=None,
                                missing_seed_count=1,
                                rationale=(
                                    f"No observations for {arch} with {obj} "
                                    f"on {ds} ({task})."
                                ),
                            )
                        )

    # 2. Single-seed observations (N=1)
    for q in campaign.research_questions:
        for metric_id in q.dependent_metrics:
            cells = store.query(metric_id=metric_id)
            if not cells:
                continue
            # Group by architecture and objective
            groups: dict[tuple[str, str], list[int]] = {}
            for c in cells:
                arch = str(c.factors.get("architecture", "unknown"))
                obj = str(c.factors.get("pretraining_objective", "unknown"))
                if c.seed is not None:
                    groups.setdefault((arch, obj), []).append(c.seed)

            for (arch, obj), seed_list in groups.items():
                if len(set(seed_list)) == 1:
                    gap_idx += 1
                    gaps.append(
                        EvidenceGap(
                            gap_id=f"gap_single_seed_{gap_idx}",
                            research_question_id=q.question_id,
                            missing_factor_combination={
                                "architecture": arch,
                                "pretraining_objective": obj,
                                "existing_seed": seed_list[0],
                            },
                            missing_metric_id=metric_id,
                            missing_seed_count=2,
                            rationale=(
                                f"Metric '{metric_id}' for ({arch}, {obj}) has only "
                                f"1 seed ({seed_list[0]}); std is undefined."
                            ),
                        )
                    )

    # 3. Missing dependent metrics for research questions
    for q in campaign.research_questions:
        for metric_id in q.dependent_metrics:
            matched = store.query(metric_id=metric_id)
            if not matched:
                gap_idx += 1
                gaps.append(
                    EvidenceGap(
                        gap_id=f"gap_metric_{gap_idx}",
                        research_question_id=q.question_id,
                        missing_factor_combination=dict(q.controlled_factors),
                        missing_metric_id=metric_id,
                        missing_seed_count=len(campaign.seeds or [42]),
                        rationale=(
                            f"Research question '{q.question_id}' requires metric "
                            f"'{metric_id}', but zero observations exist in store."
                        ),
                    )
                )

    return gaps


def build_missing_experiment_plan(
    campaign: BenchmarkCampaign,
    gaps: Sequence[EvidenceGap],
) -> MissingExperimentPlan:
    """Generate plan detailing experiments required to close identified gaps."""
    plan_items: list[dict[str, Any]] = []

    for gap in gaps:
        plan_items.append(
            {
                "gap_id": gap.gap_id,
                "research_question_id": gap.research_question_id,
                "target_factors": gap.missing_factor_combination,
                "target_metric": gap.missing_metric_id,
                "missing_seeds": gap.missing_seed_count,
                "rationale": gap.rationale,
            }
        )

    return MissingExperimentPlan(
        plan_id=f"plan_gaps_{campaign.campaign_id}",
        campaign_id=campaign.campaign_id,
        missing_experiments=plan_items,
        estimated_work_units=len(plan_items),
        warnings=(
            ["Gaps detected requiring additional experimental runs."]
            if plan_items
            else []
        ),
    )
