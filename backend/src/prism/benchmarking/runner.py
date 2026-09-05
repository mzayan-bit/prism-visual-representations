"""Benchmark campaign runner, dry-run simulator, and failure-tolerant orchestrator."""

from __future__ import annotations

import datetime
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from prism.benchmarking.contracts import BenchmarkCampaign, BenchmarkResultCell
from prism.benchmarking.enums import CampaignStatus, ResultStatus
from prism.benchmarking.store import BenchmarkResultStore


class BenchmarkExecutionFailure(BaseModel):
    """Structured failure record for an experiment in a benchmark campaign."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_id: str
    factors: dict[str, Any]
    error_type: str
    error_message: str
    timestamp: str


class BenchmarkExecutionSummary(BaseModel):
    """Execution outcome summary from a benchmark runner campaign pass."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    campaign_id: str
    status: CampaignStatus
    total_planned: int
    executed_count: int
    skipped_count: int
    failed_count: int
    failures: list[BenchmarkExecutionFailure] = Field(default_factory=list)
    started_at: str
    completed_at: str
    warnings: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class BenchmarkCampaignRunner:
    """Orchestrator for benchmark campaigns with dry-run and failure tolerance."""

    def __init__(
        self,
        experiment_executor: (
            Callable[[dict[str, Any]], list[BenchmarkResultCell]] | None
        ) = None,
    ) -> None:
        self._executor = experiment_executor

    def dry_run(
        self,
        campaign: BenchmarkCampaign,
        store: BenchmarkResultStore,
        filter_factors: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Simulate campaign execution without modifying store or running runs."""
        planned_combos = self._generate_combinations(campaign, filter_factors)
        skipped: list[dict[str, Any]] = []
        to_execute: list[dict[str, Any]] = []

        for combo in planned_combos:
            existing = store.query(factors=combo)
            if any(c.status == ResultStatus.OBSERVED for c in existing):
                skipped.append(combo)
            else:
                to_execute.append(combo)

        return {
            "campaign_id": campaign.campaign_id,
            "is_dry_run": True,
            "total_planned": len(planned_combos),
            "would_execute_count": len(to_execute),
            "would_skip_count": len(skipped),
            "planned_combinations": planned_combos,
            "to_execute_combinations": to_execute,
        }

    def run_campaign(
        self,
        campaign: BenchmarkCampaign,
        store: BenchmarkResultStore,
        filter_factors: dict[str, Any] | None = None,
        continue_on_error: bool = True,
    ) -> BenchmarkExecutionSummary:
        """Execute missing campaign experiments and register results in store."""
        start_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
        planned_combos = self._generate_combinations(campaign, filter_factors)

        executed_count = 0
        skipped_count = 0
        failures: list[BenchmarkExecutionFailure] = []

        for combo in planned_combos:
            # Check if already completed in store
            existing = store.query(factors=combo)
            if any(c.status == ResultStatus.OBSERVED for c in existing):
                skipped_count += 1
                continue

            if self._executor is not None:
                try:
                    cells = self._executor(combo)
                    store.bulk_register(cells)
                    executed_count += 1
                except Exception as exc:
                    now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
                    failure = BenchmarkExecutionFailure(
                        experiment_id=f"exp_{combo.get('architecture')}_{combo.get('pretraining_objective')}",
                        factors=combo,
                        error_type=exc.__class__.__name__,
                        error_message=str(exc),
                        timestamp=now_str,
                    )
                    failures.append(failure)
                    # Register FAILED result cell for traceability
                    store.register_cell(
                        BenchmarkResultCell(
                            result_id=f"fail_{len(failures)}_{combo.get('architecture')}",
                            experiment_id=failure.experiment_id,
                            experiment_fingerprint=f"fp_failed_{failure.experiment_id}",
                            metric_id="execution_status",
                            value=None,
                            status=ResultStatus.FAILED,
                            factors=combo,
                            warnings=[f"EXECUTION_FAILED: {exc}"],
                        )
                    )
                    if not continue_on_error:
                        break
            else:
                # No live executor attached; mark as simulated missing execution
                skipped_count += 1

        end_time = datetime.datetime.now(datetime.timezone.utc).isoformat()

        if failures and executed_count == 0:
            status = CampaignStatus.FAILED
        elif failures:
            status = CampaignStatus.PARTIAL
        elif executed_count > 0 or skipped_count == len(planned_combos):
            status = CampaignStatus.COMPLETED
        else:
            status = CampaignStatus.PLANNED

        return BenchmarkExecutionSummary(
            campaign_id=campaign.campaign_id,
            status=status,
            total_planned=len(planned_combos),
            executed_count=executed_count,
            skipped_count=skipped_count,
            failed_count=len(failures),
            failures=failures,
            started_at=start_time,
            completed_at=end_time,
            warnings=[f.error_message for f in failures],
        )

    def _generate_combinations(
        self,
        campaign: BenchmarkCampaign,
        filter_factors: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Cartesian product of campaign factors filtered by target criteria."""
        archs = campaign.architectures or ["resnet"]
        objs = campaign.objectives or ["supervised"]
        datasets = campaign.datasets or ["cifar10"]
        tasks = campaign.tasks or ["classification"]
        seeds = campaign.seeds or [42]
        budgets = campaign.budgets or [1.0]

        combos: list[dict[str, Any]] = []
        for arch in archs:
            for obj in objs:
                for ds in datasets:
                    for task in tasks:
                        for seed in seeds:
                            for budget in budgets:
                                combo = {
                                    "architecture": arch,
                                    "pretraining_objective": obj,
                                    "dataset": ds,
                                    "task": task,
                                    "seed": seed,
                                    "data_budget": budget,
                                }
                                # Apply filter if given
                                if filter_factors:
                                    match = True
                                    for fk, fv in filter_factors.items():
                                        if combo.get(fk) != fv:
                                            match = False
                                            break
                                    if not match:
                                        continue
                                combos.append(combo)
        return combos
