"""Smoke test demonstrating the end-to-end experiment contract flow.

NOTE: This is a pure domain contract demonstration using synthetic test values.
No real model training or dataset loading is executed in this test.
"""

import pytest

from prism.artifacts.contracts import ArtifactReference
from prism.core.enums import (
    ArtifactType,
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
from prism.evaluation.reports import EvaluationReport
from prism.experiments.definitions import ExperimentDefinition
from prism.experiments.metrics import MetricRecord
from prism.experiments.reproducibility import ReproducibilityConfiguration
from prism.experiments.runs import ExperimentRun
from prism.models.specifications import ModelSpecification
from prism.training.configuration import (
    OptimizerSpecification,
    SchedulerSpecification,
    TrainingConfiguration,
)


def _get_status(run: ExperimentRun) -> RunStatus:
    """Helper to retrieve current run status without mypy literal narrowing."""
    return run.status


@pytest.mark.smoke
def test_smoke_end_to_end_contract_flow() -> None:
    """Demonstrate the full lifecycle of an experiment from definition to report."""

    # 1. Define Dataset Manifest
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
        preprocessing=PreprocessingPolicy(
            resize=(64, 64),
            normalization_mean=(0.5, 0.5, 0.5),
            normalization_std=(0.5, 0.5, 0.5),
        ),
        augmentation=AugmentationPolicy(enabled=False),
    )

    # 2. Define Model Specification
    model = ModelSpecification(
        model_id="model-synthetic-cnn",
        name="Synthetic ConvNet",
        family=ModelFamily.CNN,
        architecture="custom_tiny_cnn",
        compatible_tasks=[TaskType.CLASSIFICATION],
        input_shape=(3, 64, 64),
        num_classes=2,
        hyperparameters={"channels": [16, 32]},
    )

    # 3. Define Training and Evaluation Configurations
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
            MetricSpecification(
                name="loss",
                direction=MetricDirection.MINIMIZE,
                target_split="test",
            ),
        ],
    )

    # 4. Create immutable ExperimentDefinition
    experiment = ExperimentDefinition(
        experiment_id="exp-synthetic-demo-01",
        name="Synthetic End-to-End Contract Demonstration",
        description="Verifies the contract lifecycle across PRISM abstractions",
        task_type=TaskType.CLASSIFICATION,
        hypothesis="Validates contract dataflow integrity across domain boundaries",
        dataset=dataset,
        model=model,
        training=training,
        evaluation=evaluation,
        reproducibility=ReproducibilityConfiguration(seed=42, deterministic=True),
        tags=["demo", "contract_test"],
    )

    # 5. Serialize and compute fingerprint
    fingerprint = experiment.compute_fingerprint()
    assert isinstance(fingerprint, str) and len(fingerprint) == 64
    serialized_json = experiment.to_json()
    assert "exp-synthetic-demo-01" in serialized_json

    # 6. Initialize an ExperimentRun from definition
    run = ExperimentRun(
        run_id="run-synthetic-demo-trial-01",
        experiment_id=experiment.experiment_id,
        status=RunStatus.PLANNED,
        configuration_fingerprint=fingerprint,
        reproducibility=experiment.reproducibility,
    )

    assert _get_status(run) == RunStatus.PLANNED

    # 7. Transition run through lifecycle
    run.start()
    assert _get_status(run) == RunStatus.RUNNING
    assert run.started_at is not None

    # 8. Record synthetic metrics and register artifact
    metric_acc = MetricRecord(
        metric_name="top1_accuracy",
        value=0.92,
        split="test",
        epoch=10,
        direction=MetricDirection.MAXIMIZE,
    )
    metric_loss = MetricRecord(
        metric_name="loss",
        value=0.18,
        split="test",
        epoch=10,
        direction=MetricDirection.MINIMIZE,
    )
    run.add_metric(metric_acc)
    run.add_metric(metric_loss)

    checkpoint_artifact = ArtifactReference(
        artifact_id="art-synthetic-ckpt-final",
        artifact_type=ArtifactType.CHECKPOINT,
        logical_name="synthetic_final_weights",
        uri="artifacts/checkpoints/synthetic_final.pt",
        checksum="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        producing_run_id=run.run_id,
    )
    run.add_artifact(checkpoint_artifact)

    # 9. Complete the run with summary metrics
    run.complete(summary_metrics={"top1_accuracy": 0.92, "loss": 0.18})
    assert _get_status(run) == RunStatus.COMPLETED
    assert run.completed_at is not None

    # 10. Generate EvaluationReport
    report = EvaluationReport(
        report_id="rep-synthetic-demo-01",
        experiment_id=experiment.experiment_id,
        run_id=run.run_id,
        evaluation_config=experiment.evaluation,
        metric_records=run.metric_records,
        artifact_references=run.artifact_references,
        summary_metrics=run.summary_metrics,
    )

    assert report.experiment_id == experiment.experiment_id
    assert report.run_id == run.run_id
    assert report.summary_metrics["top1_accuracy"] == 0.92
    assert len(report.metric_records) == 2
    assert len(report.artifact_references) == 1

    # 11. Verify serialization of report
    report_json = report.to_json()
    restored_report = EvaluationReport.from_json(report_json)
    assert restored_report.report_id == report.report_id
