"""Unit tests for representation profiles, Pareto analysis, and tradeoff extraction."""

from prism.benchmarking.contracts import BenchmarkResultCell
from prism.benchmarking.enums import MetricDirection, ResultStatus
from prism.benchmarking.store import BenchmarkResultStore
from prism.benchmarking.synthesis import (
    compute_pareto_front,
    extract_representation_profile,
    extract_tradeoff_pairs,
    synthesize_cross_architecture,
    synthesize_cross_objective,
)


def _populate_test_store() -> BenchmarkResultStore:
    store = BenchmarkResultStore()
    archs = ["resnet", "vit", "cnn"]
    objs = ["supervised", "simclr"]
    metrics = {
        "accuracy": {"resnet": 0.88, "vit": 0.90, "cnn": 0.82},
        "robustness_accuracy_drop": {"resnet": 0.12, "vit": 0.10, "cnn": 0.15},
        "ood_auroc": {"resnet": 0.91, "vit": 0.94, "cnn": 0.86},
        "retrieval_r1": {"resnet": 0.45, "vit": 0.50, "cnn": 0.35},
    }

    for arch in archs:
        for obj in objs:
            for seed in (42, 100):
                for mid, arch_vals in metrics.items():
                    val = arch_vals[arch] + (0.02 if obj == "supervised" else 0.0)
                    cell = BenchmarkResultCell(
                        result_id=f"c_{arch}_{obj}_{mid}_{seed}",
                        experiment_id=f"exp_{arch}_{obj}",
                        experiment_fingerprint="fp123",
                        metric_id=mid,
                        value=val,
                        status=ResultStatus.OBSERVED,
                        seed=seed,
                        source_report_type="test",
                        source_run_id="run1",
                        factors={
                            "architecture": arch,
                            "pretraining_objective": obj,
                            "seed": seed,
                        },
                    )
                    store.register_cell(cell)
    return store


def test_extract_representation_profile() -> None:
    store = _populate_test_store()
    profile = extract_representation_profile(
        store, architecture="resnet", objective="supervised"
    )

    assert profile.architecture == "resnet"
    assert profile.objective == "supervised"
    assert profile.semantic_performance is not None
    assert profile.semantic_performance > 0.8
    assert profile.robustness is not None
    assert profile.ood_separation is not None


def test_compute_pareto_front() -> None:
    store = _populate_test_store()
    pareto = compute_pareto_front(
        store,
        metric_ids=["accuracy", "robustness_accuracy_drop"],
        metric_directions={
            "accuracy": MetricDirection.HIGHER_IS_BETTER,
            "robustness_accuracy_drop": MetricDirection.LOWER_IS_BETTER,
        },
    )

    assert len(pareto.candidate_experiment_ids) > 0
    assert len(pareto.non_dominated_experiment_ids) > 0
    for non_dom in pareto.non_dominated_experiment_ids:
        assert non_dom in pareto.candidate_experiment_ids


def test_extract_tradeoff_pairs() -> None:
    store = _populate_test_store()
    points = extract_tradeoff_pairs(
        store,
        x_metric_id="accuracy",
        y_metric_id="robustness_accuracy_drop",
    )

    assert len(points) > 0
    first = points[0]
    assert first.x_metric == "accuracy"
    assert first.y_metric == "robustness_accuracy_drop"
    assert first.x_value > 0
    assert first.y_value > 0


def test_cross_architecture_and_objective_synthesis() -> None:
    store = _populate_test_store()
    arch_synth = synthesize_cross_architecture(
        store, architectures=["resnet", "vit", "cnn"]
    )
    assert "resnet" in arch_synth
    assert "vit" in arch_synth
    assert "accuracy" in arch_synth["resnet"]

    obj_synth = synthesize_cross_objective(store, objectives=["supervised", "simclr"])
    assert "supervised" in obj_synth
    assert "simclr" in obj_synth
