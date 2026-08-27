"""Unit tests for TrainingEngine and training lifecycle transitions."""

import pytest

from prism.core.enums import (
    ModelFamily,
    OrderingStrategy,
    PrecisionMode,
    RunStatus,
    TaskType,
)
from prism.core.errors import (
    TrainingError,
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
    TrainingConfiguration,
)
from prism.training.engine import TrainingEngine
from prism.training.results import TrainingResult


@pytest.fixture
def synthetic_train_data() -> tuple[MaterializedDataset, DeterministicBatchLoader]:
    samples = [
        MaterializedSample(
            sample_id=f"synth/train/{i:04d}",
            source_split="train",
            source_index=i,
            data=[1.0, -1.0] if i % 2 == 0 else [-1.0, 1.0],
            target=i % 2,
        )
        for i in range(20)
    ]
    ds = MaterializedDataset(dataset_id="ds-synth-train", samples=samples)
    loader = DeterministicBatchLoader(
        dataset=ds,
        batch_size=4,
        ordering_strategy=OrderingStrategy.EPOCH_AWARE_SHUFFLE,
        seed=42,
    )
    return ds, loader


@pytest.fixture
def base_experiment() -> ExperimentDefinition:
    dataset_manifest = DatasetManifest(
        dataset_id="ds-synth-train",
        name="Synthetic Train Dataset",
        version="1.0.0",
        num_classes=2,
        splits=[
            SplitSpecification(split_name="train", num_samples=20),
            SplitSpecification(split_name="test", num_samples=10),
        ],
    )
    model = ModelSpecification(
        model_id="model-linear-train",
        name="Linear Train Model",
        family=ModelFamily.LINEAR,
        architecture="linear_softmax",
        compatible_tasks=[TaskType.CLASSIFICATION],
        input_shape=(2,),
        num_classes=2,
    )
    training = TrainingConfiguration(
        epochs=3,
        batch_size=4,
        optimizer=OptimizerSpecification(type="sgd", lr=0.1),
        precision=PrecisionMode.FP32,
    )
    evaluation = EvaluationConfiguration(
        target_splits=["test"],
        metrics=[MetricSpecification(name="top1_accuracy")],
    )
    return ExperimentDefinition(
        experiment_id="exp-linear-train-001",
        name="Linear Train Test Experiment",
        task_type=TaskType.CLASSIFICATION,
        dataset=dataset_manifest,
        model=model,
        training=training,
        evaluation=evaluation,
        reproducibility=ReproducibilityConfiguration(seed=42),
    )


@pytest.mark.unit
def test_training_engine_multi_epoch(
    base_experiment: ExperimentDefinition,
    synthetic_train_data: tuple[MaterializedDataset, DeterministicBatchLoader],
) -> None:
    """Verify TrainingEngine executes configured epochs and returns TrainingResult."""
    train_ds, train_loader = synthetic_train_data
    harness = ExperimentExecutionHarness()
    run, prepared_exec = harness.prepare(base_experiment)

    engine = TrainingEngine()
    result = engine.train(
        experiment=base_experiment,
        prepared_execution=prepared_exec,
        train_dataset=train_ds,
        train_loader=train_loader,
        run=run,
    )

    assert isinstance(result, TrainingResult)
    assert result.status == RunStatus.COMPLETED
    assert result.epochs_completed == 3
    assert result.total_batches == 15  # 3 epochs * 5 batches per epoch
    assert result.total_examples == 60
    assert result.final_train_loss > 0.0
    assert 0.0 <= result.final_train_accuracy <= 1.0

    # Ensure MetricRecords were appended to run
    train_loss_records = [
        r for r in run.metric_records if r.metric_name == "train_loss"
    ]
    assert len(train_loss_records) == 3


@pytest.mark.unit
def test_training_engine_handles_failure(
    base_experiment: ExperimentDefinition,
) -> None:
    """Verify TrainingEngine transitions run to FAILED on exception."""
    # Corrupt sample data to cause dimension mismatch during forward pass
    corrupt_samples = [
        MaterializedSample(
            sample_id="synth/train/corrupt",
            source_split="train",
            source_index=0,
            data=[1.0, 2.0, 3.0, 4.0, 5.0],  # expected dim is 2
            target=0,
        )
    ]
    corrupt_ds = MaterializedDataset(
        dataset_id="ds-synth-corrupt", samples=corrupt_samples
    )
    corrupt_loader = DeterministicBatchLoader(dataset=corrupt_ds, batch_size=1)

    harness = ExperimentExecutionHarness()
    run, prepared_exec = harness.prepare(base_experiment)

    engine = TrainingEngine()
    with pytest.raises(TrainingError):
        engine.train(
            experiment=base_experiment,
            prepared_execution=prepared_exec,
            train_dataset=corrupt_ds,
            train_loader=corrupt_loader,
            run=run,
        )

    assert run.status == RunStatus.FAILED
    assert run.failure_info is not None
    assert run.failure_info.error_type == "ValidationError"
