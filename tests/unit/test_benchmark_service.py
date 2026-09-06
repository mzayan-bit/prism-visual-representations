"""Unit tests for BenchmarkService querying and dataset export."""

from prism.benchmarking.contracts import BenchmarkResultCell
from prism.benchmarking.enums import ResultStatus
from prism.benchmarking.service import BenchmarkService, create_default_prism_campaign
from prism.benchmarking.store import BenchmarkResultStore


def test_benchmark_service_api() -> None:
    store = BenchmarkResultStore()
    campaign = create_default_prism_campaign()

    # Add dummy cell
    cell = BenchmarkResultCell(
        result_id="srv_1",
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
            "seed": 42,
        },
    )
    store.register_cell(cell)

    service = BenchmarkService(store=store, campaign=campaign)

    mat = service.get_matrix("accuracy")
    assert mat.metric_id == "accuracy"

    profile = service.get_profile("resnet", "supervised")
    assert profile.architecture == "resnet"

    pareto = service.get_pareto_front(["accuracy", "robustness_accuracy_drop"])
    assert "accuracy" in pareto.metric_ids

    dataset_dict = service.export_dataset_for_frontend()
    assert "campaign" in dataset_dict
    assert "coverage_summary" in dataset_dict
    assert "benchmark_tables" in dataset_dict
    assert "profiles" in dataset_dict
    assert "findings" in dataset_dict
