"""Smoke test demonstrating the Phase 3 reproducibility runtime preparation flow.

NOTE: This is a pure contract and runtime initialization verification.
No real model training or dataset loading is executed in this test.
"""

from pathlib import Path

import pytest

from prism.core.enums import (
    MetricDirection,
    ModelFamily,
    PrecisionMode,
    RunStatus,
    TaskType,
)
from prism.data.manifests import (
    AugmentationPolicy,
    DatasetManifest,
    PreprocessingPolicy,
    SplitSpecification,
)
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
    SchedulerSpecification,
    TrainingConfiguration,
)


@pytest.mark.smoke
def test_smoke_reproducibility_preparation_flow() -> None:
    """Demonstrate experiment preparation and runtime context binding."""

    # 1. Define synthetic classification experiment
    dataset = DatasetManifest(
        dataset_id="ds-synthetic-vision",
        name="Synthetic Vision Benchmark",
        version="1.0.0",
        compatible_tasks=[TaskType.CLASSIFICATION],
        splits=[
            SplitSpecification(split_name="train", num_samples=1000),
            SplitSpecification(split_name="test", num_samples=200),
        ],
        classes=["class_a", "class_b"],
        num_classes=2,
        preprocessing=PreprocessingPolicy(resize=(64, 64)),
        augmentation=AugmentationPolicy(enabled=False),
    )

    model = ModelSpecification(
        model_id="model-synthetic-cnn",
        name="Synthetic ConvNet",
        family=ModelFamily.CNN,
        architecture="custom_tiny_cnn",
        compatible_tasks=[TaskType.CLASSIFICATION],
        input_shape=(3, 64, 64),
        num_classes=2,
    )

    training = TrainingConfiguration(
        epochs=10,
        batch_size=32,
        optimizer=OptimizerSpecification(type="adam", lr=1e-3),
        scheduler=SchedulerSpecification(type="none"),
        precision=PrecisionMode.FP32,
    )

    evaluation = EvaluationConfiguration(
        target_splits=["test"],
        metrics=[
            MetricSpecification(
                name="top1_accuracy",
                direction=MetricDirection.MAXIMIZE,
                target_split="test",
            ),
        ],
    )

    reproducibility = ReproducibilityConfiguration(
        seed=42,
        deterministic=False,
        capture_code_revision=True,
        capture_environment=True,
    )

    experiment = ExperimentDefinition(
        experiment_id="exp-synthetic-repro-01",
        name="Synthetic Reproducibility Harness Demonstration",
        task_type=TaskType.CLASSIFICATION,
        dataset=dataset,
        model=model,
        training=training,
        evaluation=evaluation,
        reproducibility=reproducibility,
    )

    # 2. Mock Git runner for predictable sandbox test execution
    def mock_git(
        args: list[str],
        cwd: Path | str | None,
        timeout: float,
    ) -> tuple[int, str, str]:
        cmd = " ".join(args)
        if cmd == "rev-parse --is-inside-work-tree":
            return 0, "true", ""
        if cmd == "rev-parse HEAD":
            return 0, "c" * 40, ""
        if cmd == "rev-parse --short HEAD":
            return 0, "ccccccc", ""
        if cmd == "rev-parse --abbrev-ref HEAD":
            return 0, "main", ""
        if cmd == "status --porcelain -uno":
            return 0, "", ""
        if cmd == "config --get remote.origin.url":
            return 0, "https://github.com/mzayan-bit/prism.git", ""
        return 0, "", ""

    # 3. Prepare execution via harness
    harness = ExperimentExecutionHarness()
    run, prepared = harness.prepare(
        experiment,
        git_runner=mock_git,
    )

    # 4. Verify runtime state and linkages
    assert isinstance(run, ExperimentRun)
    assert isinstance(prepared, PreparedExecution)
    assert run.experiment_id == experiment.experiment_id
    assert run.status == RunStatus.PLANNED
    assert run.configuration_fingerprint == experiment.compute_fingerprint()
    assert run.code_revision is not None
    assert run.code_revision.git_commit == "c" * 40
    assert run.environment is not None

    assert prepared.experiment_id == experiment.experiment_id
    assert prepared.run_id == run.run_id
    assert prepared.configuration_fingerprint == run.configuration_fingerprint
    assert prepared.seeding_result.requested_seed == 42
    assert prepared.seeding_result.python_seeded is True
    assert prepared.hardware.cpu_count is not None
    assert prepared.code_revision.git_commit == "c" * 40

    # 5. Verify capabilities report generation and serialization
    report = prepared.get_reproducibility_report()
    assert report["experiment_id"] == experiment.experiment_id
    assert report["requested"]["seed"] == 42

    serialized = prepared.to_json()
    assert experiment.experiment_id in serialized
    assert run.run_id in serialized
