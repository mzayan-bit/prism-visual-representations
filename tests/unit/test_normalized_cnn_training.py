"""Unit tests for Normalized CNN training lifecycle with TrainingEngine."""

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
from prism.models.cnn import ConvolutionalNeuralNetwork
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
                sample_id=f"cnn_norm_synth/train/{i:04d}",
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
                sample_id=f"cnn_norm_synth/test/{i:04d}",
                source_split="test",
                source_index=i,
                data=img,
                target=target,
            )
        )

    train_ds = MaterializedDataset(
        dataset_id="ds-cnn-norm-synth", samples=train_samples
    )
    test_ds = MaterializedDataset(dataset_id="ds-cnn-norm-synth", samples=test_samples)

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
def normalized_cnn_experiment() -> ExperimentDefinition:
    dataset_manifest = DatasetManifest(
        dataset_id="ds-cnn-norm-synth",
        name="Synthetic Normalized CNN Dataset",
        version="1.0.0",
        num_classes=2,
        splits=[
            SplitSpecification(split_name="train", num_samples=20),
            SplitSpecification(split_name="test", num_samples=6),
        ],
    )
    model = ModelSpecification(
        model_id="model-cnn-norm-unit-train",
        name="Unit Test Normalized CNN",
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
            "normalization": "batch_norm",
            "norm_eps": 1e-5,
            "norm_momentum": 0.1,
            "norm_affine": True,
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
        experiment_id="exp-cnn-norm-unit-train",
        name="Normalized CNN Unit Train Experiment",
        task_type=TaskType.CLASSIFICATION,
        dataset=dataset_manifest,
        model=model,
        training=training,
        evaluation=evaluation,
        reproducibility=ReproducibilityConfiguration(seed=42, deterministic=True),
    )


@pytest.mark.unit
def test_normalized_cnn_training_lifecycle(
    normalized_cnn_experiment: ExperimentDefinition,
    synthetic_cnn_dataset: tuple[
        MaterializedDataset,
        DeterministicBatchLoader,
        MaterializedDataset,
        DeterministicBatchLoader,
    ],
) -> None:
    """Verify TrainingEngine trains Normalized CNN and updates running stats."""
    train_ds, train_loader, test_ds, test_loader = synthetic_cnn_dataset
    harness = ExperimentExecutionHarness()
    run, prepared_exec = harness.prepare(normalized_cnn_experiment)

    # Initialize model to inspect parameters before and after
    model = ConvolutionalNeuralNetwork(spec=normalized_cnn_experiment.model, seed=42)
    init_params = model.get_parameters()
    init_state = model.get_state()

    engine = TrainingEngine()
    result = engine.train(
        experiment=normalized_cnn_experiment,
        prepared_execution=prepared_exec,
        train_dataset=train_ds,
        train_loader=train_loader,
        test_dataset=test_ds,
        test_loader=test_loader,
        run=run,
        model=model,
    )

    assert isinstance(result, TrainingResult)
    assert result.status == RunStatus.COMPLETED
    assert result.epochs_completed == 3
    assert result.total_batches == 15

    # Verify parameters updated
    trained_params = model.get_parameters()
    trained_state = model.get_state()

    assert trained_params["conv_0_weights"] != init_params["conv_0_weights"]
    assert trained_params["norm_0_gamma"] != init_params["norm_0_gamma"]
    assert trained_state["norm_0_running_mean"] != init_state["norm_0_running_mean"]
    assert trained_state["norm_0_num_batches_tracked"] == 15
