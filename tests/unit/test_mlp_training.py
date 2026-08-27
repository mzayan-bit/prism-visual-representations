"""Unit tests for MultiLayerPerceptron training lifecycle, scheduler, and metrics."""

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
def synthetic_train_test_data() -> tuple[
    MaterializedDataset,
    DeterministicBatchLoader,
    MaterializedDataset,
    DeterministicBatchLoader,
]:
    # Separable synthetic dataset: class 0 has pos feature, class 1 has neg feature
    train_samples = [
        MaterializedSample(
            sample_id=f"synth/train/{i:04d}",
            source_split="train",
            source_index=i,
            data=[2.0, -2.0] if i % 2 == 0 else [-2.0, 2.0],
            target=i % 2,
        )
        for i in range(40)
    ]
    test_samples = [
        MaterializedSample(
            sample_id=f"synth/test/{i:04d}",
            source_split="test",
            source_index=i,
            data=[2.0, -2.0] if i % 2 == 0 else [-2.0, 2.0],
            target=i % 2,
        )
        for i in range(10)
    ]

    train_ds = MaterializedDataset(dataset_id="ds-mlp-synth", samples=train_samples)
    test_ds = MaterializedDataset(dataset_id="ds-mlp-synth", samples=test_samples)

    train_loader = DeterministicBatchLoader(
        dataset=train_ds,
        batch_size=4,
        ordering_strategy=OrderingStrategy.EPOCH_AWARE_SHUFFLE,
        seed=42,
    )
    test_loader = DeterministicBatchLoader(
        dataset=test_ds,
        batch_size=5,
        ordering_strategy=OrderingStrategy.SEQUENTIAL,
        seed=42,
    )

    return train_ds, train_loader, test_ds, test_loader


@pytest.fixture
def mlp_experiment() -> ExperimentDefinition:
    dataset_manifest = DatasetManifest(
        dataset_id="ds-mlp-synth",
        name="Synthetic MLP Benchmark",
        version="1.0.0",
        num_classes=2,
        splits=[
            SplitSpecification(split_name="train", num_samples=40),
            SplitSpecification(split_name="test", num_samples=10),
        ],
    )
    model = ModelSpecification(
        model_id="model-mlp-train-unit",
        name="MLP Unit Train Model",
        family=ModelFamily.MLP,
        architecture="mlp",
        compatible_tasks=[TaskType.CLASSIFICATION],
        input_shape=(2,),
        num_classes=2,
        hyperparameters={
            "hidden_dims": [16, 8],
            "activation": "relu",
            "dropout": 0.1,
        },
    )
    training = TrainingConfiguration(
        epochs=4,
        batch_size=4,
        optimizer=OptimizerSpecification(type="sgd", lr=0.1, momentum=0.9),
        scheduler=SchedulerSpecification(type="cosine", min_lr=0.01),
        precision=PrecisionMode.FP32,
    )
    evaluation = EvaluationConfiguration(
        target_splits=["test"],
        metrics=[MetricSpecification(name="top1_accuracy")],
    )
    return ExperimentDefinition(
        experiment_id="exp-mlp-unit-train",
        name="MLP Unit Training Experiment",
        task_type=TaskType.CLASSIFICATION,
        dataset=dataset_manifest,
        model=model,
        training=training,
        evaluation=evaluation,
        reproducibility=ReproducibilityConfiguration(seed=42, deterministic=True),
    )


@pytest.mark.unit
def test_mlp_training_engine_full_lifecycle(
    mlp_experiment: ExperimentDefinition,
    synthetic_train_test_data: tuple[
        MaterializedDataset,
        DeterministicBatchLoader,
        MaterializedDataset,
        DeterministicBatchLoader,
    ],
) -> None:
    """Verify TrainingEngine trains MLP across epochs, logs LR and metrics."""
    train_ds, train_loader, test_ds, test_loader = synthetic_train_test_data
    harness = ExperimentExecutionHarness()
    run, prepared_exec = harness.prepare(mlp_experiment)

    engine = TrainingEngine()
    result = engine.train(
        experiment=mlp_experiment,
        prepared_execution=prepared_exec,
        train_dataset=train_ds,
        train_loader=train_loader,
        test_dataset=test_ds,
        test_loader=test_loader,
        run=run,
    )

    assert isinstance(result, TrainingResult)
    assert result.status == RunStatus.COMPLETED
    assert result.epochs_completed == 4
    assert result.total_batches == 40  # 4 epochs * 10 batches per epoch
    assert result.final_train_loss < 0.5  # Model learned separable data
    assert result.final_train_accuracy > 0.8
    assert len(result.evaluation_reports) == 1

    # Verify learning_rate records exist in run.metric_records
    lr_records = [r for r in run.metric_records if r.metric_name == "learning_rate"]
    assert len(lr_records) == 4
    # Learning rate smoothly decreased from base_lr (0.1) toward min_lr (0.01)
    assert lr_records[0].value == pytest.approx(0.1, abs=1e-3)
    assert lr_records[-1].value < lr_records[0].value
