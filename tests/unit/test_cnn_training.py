"""Unit tests for CNN training lifecycle and metrics in TrainingEngine."""

import pytest

from prism.core.enums import (
    ModelFamily,
    OrderingStrategy,
    PrecisionMode,
    RunStatus,
    TaskType,
)
from prism.data.batching import DeterministicBatchLoader
from prism.data.manifests import DatasetManifest, SplitSpecification
from prism.data.materialized import MaterializedDataset, MaterializedSample
from prism.evaluation.configuration import (
    EvaluationConfiguration,
    MetricSpecification,
)
from prism.experiments.definitions import ExperimentDefinition
from prism.experiments.harness import ExperimentExecutionHarness
from prism.experiments.reproducibility import ReproducibilityConfiguration
from prism.models.specifications import ModelSpecification
from prism.training.configuration import (
    OptimizerSpecification,
    SchedulerSpecification,
    TrainingConfiguration,
)
from prism.training.engine import TrainingEngine
from prism.training.results import TrainingResult


@pytest.fixture
def synthetic_cnn_dataset() -> tuple[
    MaterializedDataset,
    DeterministicBatchLoader,
    MaterializedDataset,
    DeterministicBatchLoader,
]:
    # 2-class separable 3x8x8 images:
    # Class 0 has bright top-left corner, Class 1 has bright bottom-right corner
    train_samples: list[MaterializedSample] = []
    for i in range(20):
        target = i % 2
        img = [[[0.0] * 8 for _ in range(8)] for _ in range(3)]
        if target == 0:
            for c in range(3):
                img[c][0][0] = 5.0
                img[c][0][1] = 5.0
        else:
            for c in range(3):
                img[c][7][7] = 5.0
                img[c][7][6] = 5.0

        train_samples.append(
            MaterializedSample(
                sample_id=f"cnn_synth/train/{i:04d}",
                source_split="train",
                source_index=i,
                data=img,
                target=target,
            )
        )

    test_samples: list[MaterializedSample] = []
    for i in range(6):
        target = i % 2
        img = [[[0.0] * 8 for _ in range(8)] for _ in range(3)]
        if target == 0:
            for c in range(3):
                img[c][0][0] = 5.0
        else:
            for c in range(3):
                img[c][7][7] = 5.0

        test_samples.append(
            MaterializedSample(
                sample_id=f"cnn_synth/test/{i:04d}",
                source_split="test",
                source_index=i,
                data=img,
                target=target,
            )
        )

    train_ds = MaterializedDataset(dataset_id="ds-cnn-synth", samples=train_samples)
    test_ds = MaterializedDataset(dataset_id="ds-cnn-synth", samples=test_samples)

    train_loader = DeterministicBatchLoader(
        dataset=train_ds,
        batch_size=4,
        ordering_strategy=OrderingStrategy.EPOCH_AWARE_SHUFFLE,
        seed=42,
    )
    test_loader = DeterministicBatchLoader(
        dataset=test_ds,
        batch_size=6,
        ordering_strategy=OrderingStrategy.SEQUENTIAL,
        seed=42,
    )
    return train_ds, train_loader, test_ds, test_loader


@pytest.fixture
def cnn_experiment() -> ExperimentDefinition:
    dataset_manifest = DatasetManifest(
        dataset_id="ds-cnn-synth",
        name="Synthetic CNN Dataset",
        version="1.0.0",
        num_classes=2,
        splits=[
            SplitSpecification(split_name="train", num_samples=20),
            SplitSpecification(split_name="test", num_samples=6),
        ],
    )
    model = ModelSpecification(
        model_id="model-cnn-unit-train",
        name="Unit Test CNN",
        family=ModelFamily.CNN,
        architecture="cnn",
        compatible_tasks=[TaskType.CLASSIFICATION],
        input_shape=(3, 8, 8),
        num_classes=2,
        hyperparameters={
            "conv_channels": [8, 16],
            "kernel_sizes": 3,
            "strides": 1,
            "paddings": 1,
            "pool_sizes": 2,
            "pool_strides": 2,
            "activation": "relu",
            "classifier_hidden_dims": [],
            "dropout": 0.0,
        },
    )
    training = TrainingConfiguration(
        epochs=3,
        batch_size=4,
        optimizer=OptimizerSpecification(type="sgd", lr=0.05, momentum=0.9),
        scheduler=SchedulerSpecification(type="cosine", min_lr=0.005),
        precision=PrecisionMode.FP32,
    )
    evaluation = EvaluationConfiguration(
        target_splits=["test"],
        metrics=[MetricSpecification(name="top1_accuracy")],
    )
    return ExperimentDefinition(
        experiment_id="exp-cnn-unit-train",
        name="CNN Unit Train Experiment",
        task_type=TaskType.CLASSIFICATION,
        dataset=dataset_manifest,
        model=model,
        training=training,
        evaluation=evaluation,
        reproducibility=ReproducibilityConfiguration(seed=42, deterministic=True),
    )


@pytest.mark.unit
def test_cnn_training_engine_lifecycle(
    cnn_experiment: ExperimentDefinition,
    synthetic_cnn_dataset: tuple[
        MaterializedDataset,
        DeterministicBatchLoader,
        MaterializedDataset,
        DeterministicBatchLoader,
    ],
) -> None:
    """Verify TrainingEngine trains CNN across epochs, logs telemetry, and evaluates."""
    train_ds, train_loader, test_ds, test_loader = synthetic_cnn_dataset
    harness = ExperimentExecutionHarness()
    run, prepared_exec = harness.prepare(cnn_experiment)

    engine = TrainingEngine()
    result = engine.train(
        experiment=cnn_experiment,
        prepared_execution=prepared_exec,
        train_dataset=train_ds,
        train_loader=train_loader,
        test_dataset=test_ds,
        test_loader=test_loader,
        run=run,
    )

    assert isinstance(result, TrainingResult)
    assert result.status == RunStatus.COMPLETED
    assert result.epochs_completed == 3
    assert result.total_batches == 15  # 3 epochs * 5 batches
    assert result.final_train_loss > 0.0
    assert len(result.evaluation_reports) == 1

    # Verify metrics
    lr_records = [r for r in run.metric_records if r.metric_name == "learning_rate"]
    assert len(lr_records) == 3
    assert lr_records[0].value == pytest.approx(0.05)
    assert lr_records[-1].value < lr_records[0].value
