from typing import Any

from prism.benchmarking.contracts import BenchmarkCampaign, BenchmarkResultCell
from prism.benchmarking.enums import CampaignStatus, ResultStatus
from prism.benchmarking.runner import BenchmarkCampaignRunner
from prism.benchmarking.store import BenchmarkResultStore


def test_runner_dry_run() -> None:
    store = BenchmarkResultStore()
    campaign = BenchmarkCampaign(
        campaign_id="camp_dry",
        title="Dry Run Campaign",
        description="Testing dry run",
        architectures=["resnet", "vit"],
        objectives=["supervised"],
        seeds=[42],
    )
    runner = BenchmarkCampaignRunner()
    result = runner.dry_run(campaign=campaign, store=store)

    assert result["is_dry_run"] is True
    assert result["total_planned"] == 2
    assert result["would_execute_count"] == 2


def test_runner_execution_and_skip() -> None:
    store = BenchmarkResultStore()
    campaign = BenchmarkCampaign(
        campaign_id="camp_exec",
        title="Exec Campaign",
        description="Testing execution",
        architectures=["resnet"],
        objectives=["supervised"],
        seeds=[42],
    )

    def dummy_executor(factors: dict[str, Any]) -> list[BenchmarkResultCell]:
        return [
            BenchmarkResultCell(
                result_id=f"res_{factors.get('architecture')}_{factors.get('seed')}",
                experiment_id=f"exp_{factors.get('architecture')}",
                experiment_fingerprint="fp1",
                metric_id="accuracy",
                value=0.88,
                status=ResultStatus.OBSERVED,
                seed=factors.get("seed", 42),
                source_report_type="test",
                source_run_id="run1",
                factors=factors,
            )
        ]

    runner = BenchmarkCampaignRunner(experiment_executor=dummy_executor)

    # First execution: executes and registers cell
    res1 = runner.run_campaign(campaign=campaign, store=store)
    assert res1.executed_count == 1
    assert res1.skipped_count == 0
    assert len(store.all_cells()) > 0

    # Second execution: skips completed experiment
    res2 = runner.run_campaign(campaign=campaign, store=store)
    assert res2.executed_count == 0
    assert res2.skipped_count == 1


def test_runner_failure_tracking() -> None:
    store = BenchmarkResultStore()
    campaign = BenchmarkCampaign(
        campaign_id="camp_fail",
        title="Fail Campaign",
        description="Testing failure",
        architectures=["resnet"],
        objectives=["supervised"],
        seeds=[42],
    )

    def failing_fn(factors: dict[str, Any]) -> list[BenchmarkResultCell]:
        raise ValueError("Simulated hardware error")

    runner = BenchmarkCampaignRunner(experiment_executor=failing_fn)
    res = runner.run_campaign(campaign=campaign, store=store)

    assert res.status == CampaignStatus.FAILED
    assert res.failed_count == 1
    assert len(res.failures) == 1
    assert "Simulated hardware error" in res.failures[0].error_message
