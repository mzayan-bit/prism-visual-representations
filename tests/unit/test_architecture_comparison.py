"""Unit coverage for Phase 13 architecture-suite contracts."""

import pytest

from prism.core.enums import ModelFamily, RunStatus, TaskType
from prism.core.errors import ValidationError
from prism.data.manifests import DatasetManifest, SplitSpecification
from prism.evaluation.configuration import EvaluationConfiguration, MetricSpecification
from prism.experiments.architecture import (
    ComparisonMode,
    audit_experiment_factors,
    count_model_parameters,
    create_architecture_comparison_suite,
)
from prism.experiments.definitions import ExperimentDefinition
from prism.experiments.reporting import (
    ArchitectureRunResult,
    ExperimentSuiteRunner,
    RepeatedSeedPlan,
    SampleEfficiencyPlan,
    SampleEfficiencyRecord,
    SampleEfficiencySummary,
    aggregate_repeated_metric,
)
from prism.models.specifications import ModelSpecification
from prism.training.configuration import OptimizerSpecification, TrainingConfiguration
from prism.training.results import TrainingResult


def _experiment(
    exp_id: str, family: ModelFamily, batch_size: int = 2
) -> ExperimentDefinition:
    dataset = DatasetManifest(
        dataset_id="ds-suite-test",
        name="Suite test data",
        splits=[SplitSpecification(split_name="train")],
        num_classes=2,
    )
    model = ModelSpecification(
        model_id=f"model-{exp_id}",
        name=exp_id,
        family=family,
        architecture=family.value,
        input_shape=(1, 4, 4),
        num_classes=2,
    )
    return ExperimentDefinition(
        experiment_id=exp_id,
        name=exp_id,
        task_type=TaskType.CLASSIFICATION,
        dataset=dataset,
        model=model,
        training=TrainingConfiguration(
            epochs=2,
            batch_size=batch_size,
            optimizer=OptimizerSpecification(type="sgd", lr=0.1),
        ),
        evaluation=EvaluationConfiguration(
            target_splits=["test"],
            metrics=[MetricSpecification(name="top1_accuracy")],
        ),
    )


@pytest.mark.unit
def test_factor_audit_detects_undeclared_batch_difference() -> None:
    cnn = _experiment("exp-cnn", ModelFamily.CNN)
    resnet = _experiment("exp-resnet", ModelFamily.RESNET, batch_size=4)
    audit = audit_experiment_factors([cnn, resnet], ["model_family"])
    assert "training.batch_size" in audit.unexpected_differences
    assert audit.is_strictly_controlled is False


@pytest.mark.unit
def test_suite_strict_validation_and_roundtrip() -> None:
    experiments = [
        _experiment("exp-cnn", ModelFamily.CNN),
        _experiment("exp-resnet", ModelFamily.RESNET, batch_size=3),
    ]
    suite = create_architecture_comparison_suite(
        "suite-test",
        "Architecture suite",
        "How do the models behave?",
        experiments,
        intentionally_varied_factors=["model_family", "model_architecture"],
        required_families=[],
    )
    with pytest.raises(ValidationError):
        suite.validate_factors()
    appropriate = suite.model_copy(
        update={"comparison_mode": ComparisonMode.ARCHITECTURE_APPROPRIATE}
    )
    assert appropriate.validate_factors().unexpected_differences
    assert (
        appropriate.from_json(appropriate.to_json()).compute_fingerprint()
        == appropriate.compute_fingerprint()
    )
    recreated = create_architecture_comparison_suite(
        "suite-test",
        "Architecture suite",
        "How do the models behave?",
        [
            _experiment("exp-cnn", ModelFamily.CNN),
            _experiment("exp-resnet", ModelFamily.RESNET, batch_size=3),
        ],
        intentionally_varied_factors=["model_family", "model_architecture"],
        required_families=[],
    )
    assert recreated.compute_fingerprint() == suite.compute_fingerprint()


@pytest.mark.unit
def test_parameter_count_is_exact_and_breakdown_sums() -> None:
    class FakeModel:
        model_id = "fake"

        def get_parameters(self) -> dict[str, object]:
            return {"stem.weights": [[1.0, 2.0], [3.0, 4.0]], "head.bias": [1.0, 2.0]}

    audit = count_model_parameters(FakeModel())
    assert audit.total_trainable_parameters == 6
    assert sum(audit.component_counts.values()) == audit.total_trainable_parameters
    assert audit.parameter_shapes["stem.weights"] == (2, 2)


@pytest.mark.unit
def test_runner_isolates_failures_and_repeated_aggregation() -> None:
    experiments = [
        _experiment("exp-cnn", ModelFamily.CNN),
        _experiment("exp-resnet", ModelFamily.RESNET),
    ]
    suite = create_architecture_comparison_suite(
        "suite-runner",
        "Runner",
        "Question",
        experiments,
        intentionally_varied_factors=["model_family", "model_architecture"],
        required_families=[],
    )

    def execute(experiment: ExperimentDefinition) -> ArchitectureRunResult:
        if experiment.model.family == ModelFamily.RESNET:
            raise RuntimeError("synthetic failure")
        result = TrainingResult(
            run_id="run-cnn",
            experiment_id=experiment.experiment_id,
            status=RunStatus.COMPLETED,
            epochs_completed=1,
            total_batches=1,
            total_examples=2,
            final_train_loss=0.5,
            final_train_accuracy=0.5,
        )
        return ArchitectureRunResult(
            experiment_id=experiment.experiment_id, training_result=result
        )

    report = ExperimentSuiteRunner().run(suite, execute)
    assert report.completed_experiment_ids == ["exp-cnn"]
    assert report.failed_experiment_ids == ["exp-resnet"]
    assert "synthetic failure" in report.warnings[-1]
    assert report.from_json(report.to_json()).suite_id == "suite-runner"
    repeated = aggregate_repeated_metric("accuracy", [0.5, 0.75])
    assert repeated.standard_deviation is not None
    assert repeated.standard_deviation > 0
    assert aggregate_repeated_metric("accuracy", [0.5]).standard_deviation == 0.0
    assert RepeatedSeedPlan(seeds=[3, 1]).seeds == [1, 3]


@pytest.mark.unit
def test_sample_efficiency_orders_records_and_marks_missing() -> None:
    plan = SampleEfficiencyPlan.create([1.0, 0.1, 0.5])
    summary = SampleEfficiencySummary.from_records(
        plan,
        [SampleEfficiencyRecord(model_family=ModelFamily.CNN, data_budget=0.5)],
    )
    assert summary.records[0].data_budget == 0.5
    assert summary.missing_budgets == [0.1, 1.0]
