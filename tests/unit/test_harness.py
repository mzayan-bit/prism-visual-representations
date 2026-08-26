"""Unit tests for ExperimentExecutionHarness."""

from pathlib import Path

import pytest

from prism.core.enums import ModelFamily, RunStatus, TaskType
from prism.core.errors import LifecycleError, ValidationError
from prism.data.manifests import DatasetManifest, SplitSpecification
from prism.evaluation.configuration import (
    EvaluationConfiguration,
    MetricSpecification,
)
from prism.experiments.context import PreparedExecution
from prism.experiments.definitions import ExperimentDefinition
from prism.experiments.harness import ExperimentExecutionHarness
from prism.experiments.reproducibility import ReproducibilityConfiguration
from prism.experiments.runs import ExperimentRun
from prism.models.specifications import ModelSpecification
from prism.training.configuration import (
    OptimizerSpecification,
    TrainingConfiguration,
)


@pytest.fixture
def valid_experiment() -> ExperimentDefinition:
    dataset = DatasetManifest(
        dataset_id="ds-cifar10",
        name="CIFAR-10",
        compatible_tasks=[TaskType.CLASSIFICATION],
        splits=[
            SplitSpecification(split_name="train"),
            SplitSpecification(split_name="test"),
        ],
        num_classes=10,
    )
    model = ModelSpecification(
        model_id="model-resnet18",
        name="ResNet-18",
        family=ModelFamily.RESNET,
        architecture="resnet18",
        compatible_tasks=[TaskType.CLASSIFICATION],
        num_classes=10,
    )
    training = TrainingConfiguration(
        epochs=50,
        batch_size=64,
        optimizer=OptimizerSpecification(type="adamw", lr=1e-3),
    )
    evaluation = EvaluationConfiguration(
        target_splits=["test"],
        metrics=[MetricSpecification(name="top1_accuracy")],
    )
    return ExperimentDefinition(
        experiment_id="exp-cifar10-resnet18",
        name="CIFAR-10 ResNet-18 Baseline",
        task_type=TaskType.CLASSIFICATION,
        dataset=dataset,
        model=model,
        training=training,
        evaluation=evaluation,
        reproducibility=ReproducibilityConfiguration(seed=42, deterministic=False),
    )


@pytest.mark.unit
def test_harness_prepare_automatic_run(
    valid_experiment: ExperimentDefinition,
) -> None:
    """Verify harness automatically creates planned run and binds metadata."""
    harness = ExperimentExecutionHarness()
    run, prepared = harness.prepare(valid_experiment)

    assert isinstance(run, ExperimentRun)
    assert isinstance(prepared, PreparedExecution)
    assert run.experiment_id == valid_experiment.experiment_id
    assert run.status == RunStatus.PLANNED
    assert run.configuration_fingerprint == valid_experiment.compute_fingerprint()
    assert run.code_revision is not None
    assert run.environment is not None

    assert prepared.experiment_id == valid_experiment.experiment_id
    assert prepared.run_id == run.run_id
    assert prepared.configuration_fingerprint == run.configuration_fingerprint
    assert prepared.seeding_result.requested_seed == 42


@pytest.mark.unit
def test_harness_prepare_with_existing_planned_run(
    valid_experiment: ExperimentDefinition,
) -> None:
    """Verify harness binds context to a provided pre-created PLANNED run."""
    harness = ExperimentExecutionHarness()
    custom_run = ExperimentRun(
        run_id="run-custom-trial-01",
        experiment_id=valid_experiment.experiment_id,
        status=RunStatus.PLANNED,
    )

    run, prepared = harness.prepare(valid_experiment, run=custom_run)

    assert run.run_id == "run-custom-trial-01"
    assert run.configuration_fingerprint == valid_experiment.compute_fingerprint()
    assert prepared.run_id == "run-custom-trial-01"


@pytest.mark.unit
def test_harness_rejects_non_planned_run(
    valid_experiment: ExperimentDefinition,
) -> None:
    """Verify harness rejects runs that are not in PLANNED state."""
    harness = ExperimentExecutionHarness()
    running_run = ExperimentRun(
        run_id="run-already-running",
        experiment_id=valid_experiment.experiment_id,
        status=RunStatus.RUNNING,
    )

    with pytest.raises(LifecycleError, match="Runs must be in PLANNED state"):
        harness.prepare(valid_experiment, run=running_run)


@pytest.mark.unit
def test_harness_rejects_mismatched_experiment_id(
    valid_experiment: ExperimentDefinition,
) -> None:
    """Verify harness rejects runs with different experiment_id."""
    harness = ExperimentExecutionHarness()
    mismatched_run = ExperimentRun(
        run_id="run-mismatched",
        experiment_id="exp-different-experiment",
        status=RunStatus.PLANNED,
    )

    with pytest.raises(ValidationError, match="does not match definition"):
        harness.prepare(valid_experiment, run=mismatched_run)


@pytest.mark.unit
def test_harness_repeated_preparation_distinct_runs(
    valid_experiment: ExperimentDefinition,
) -> None:
    """Verify repeated preparation generates distinct run IDs."""
    harness = ExperimentExecutionHarness()
    run1, prep1 = harness.prepare(valid_experiment)
    run2, prep2 = harness.prepare(valid_experiment)

    assert run1.run_id != run2.run_id
    assert prep1.run_id != prep2.run_id
    assert prep1.configuration_fingerprint == prep2.configuration_fingerprint


@pytest.mark.unit
def test_harness_with_mock_git_failures(
    valid_experiment: ExperimentDefinition,
) -> None:
    """Verify non-critical git failures do not prevent execution preparation."""

    def failing_git_runner(
        args: list[str],
        cwd: Path | str | None,
        timeout: float,
    ) -> tuple[int, str, str]:
        return -1, "", "git executable not found"

    harness = ExperimentExecutionHarness()
    run, prepared = harness.prepare(valid_experiment, git_runner=failing_git_runner)

    assert run.status == RunStatus.PLANNED
    assert any("not installed" in w for w in prepared.warnings)
