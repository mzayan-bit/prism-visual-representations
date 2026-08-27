"""Unit tests for training reproducibility and deterministic convergence."""

import pytest

from prism.core.enums import (
    ModelFamily,
    OrderingStrategy,
    PrecisionMode,
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
    TrainingConfiguration,
)
from prism.training.engine import TrainingEngine


def build_test_experiment(seed: int) -> ExperimentDefinition:
    dataset_manifest = DatasetManifest(
        dataset_id="ds-synth-repro",
        name="Synthetic Repro Dataset",
        version="1.0.0",
        num_classes=2,
        splits=[
            SplitSpecification(split_name="train", num_samples=20),
            SplitSpecification(split_name="test", num_samples=10),
        ],
    )
    model = ModelSpecification(
        model_id="model-linear-repro",
        name="Linear Repro Model",
        family=ModelFamily.LINEAR,
        architecture="linear_softmax",
        compatible_tasks=[TaskType.CLASSIFICATION],
        input_shape=(2,),
        num_classes=2,
    )
    training = TrainingConfiguration(
        epochs=2,
        batch_size=4,
        optimizer=OptimizerSpecification(type="sgd", lr=0.1),
        precision=PrecisionMode.FP32,
    )
    evaluation = EvaluationConfiguration(
        target_splits=["test"],
        metrics=[MetricSpecification(name="top1_accuracy")],
    )
    return ExperimentDefinition(
        experiment_id="exp-repro-001",
        name="Reproducibility Test Experiment",
        task_type=TaskType.CLASSIFICATION,
        dataset=dataset_manifest,
        model=model,
        training=training,
        evaluation=evaluation,
        reproducibility=ReproducibilityConfiguration(seed=seed),
    )


@pytest.fixture
def repro_dataset() -> MaterializedDataset:
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
    return MaterializedDataset(dataset_id="ds-synth-repro", samples=samples)


@pytest.mark.unit
def test_training_reproducibility_same_seed(
    repro_dataset: MaterializedDataset,
) -> None:
    """Verify that same seed produces identical training progression and final loss."""
    exp1 = build_test_experiment(seed=42)
    loader1 = DeterministicBatchLoader(
        dataset=repro_dataset,
        batch_size=4,
        ordering_strategy=OrderingStrategy.EPOCH_AWARE_SHUFFLE,
        seed=42,
    )
    harness1 = ExperimentExecutionHarness()
    run1, prepared1 = harness1.prepare(exp1)

    engine1 = TrainingEngine()
    result1 = engine1.train(
        experiment=exp1,
        prepared_execution=prepared1,
        train_dataset=repro_dataset,
        train_loader=loader1,
        run=run1,
    )

    exp2 = build_test_experiment(seed=42)
    loader2 = DeterministicBatchLoader(
        dataset=repro_dataset,
        batch_size=4,
        ordering_strategy=OrderingStrategy.EPOCH_AWARE_SHUFFLE,
        seed=42,
    )
    harness2 = ExperimentExecutionHarness()
    run2, prepared2 = harness2.prepare(exp2)

    engine2 = TrainingEngine()
    result2 = engine2.train(
        experiment=exp2,
        prepared_execution=prepared2,
        train_dataset=repro_dataset,
        train_loader=loader2,
        run=run2,
    )

    assert result1.final_train_loss == pytest.approx(result2.final_train_loss)
    assert result1.final_train_accuracy == pytest.approx(result2.final_train_accuracy)


@pytest.mark.unit
def test_training_reproducibility_diff_seeds(
    repro_dataset: MaterializedDataset,
) -> None:
    """Verify that different seeds produce divergent trajectories."""
    exp1 = build_test_experiment(seed=42)
    loader1 = DeterministicBatchLoader(
        dataset=repro_dataset,
        batch_size=4,
        ordering_strategy=OrderingStrategy.EPOCH_AWARE_SHUFFLE,
        seed=42,
    )
    harness1 = ExperimentExecutionHarness()
    run1, prepared1 = harness1.prepare(exp1)
    result1 = TrainingEngine().train(
        experiment=exp1,
        prepared_execution=prepared1,
        train_dataset=repro_dataset,
        train_loader=loader1,
        run=run1,
    )

    exp2 = build_test_experiment(seed=999)
    loader2 = DeterministicBatchLoader(
        dataset=repro_dataset,
        batch_size=4,
        ordering_strategy=OrderingStrategy.EPOCH_AWARE_SHUFFLE,
        seed=999,
    )
    harness2 = ExperimentExecutionHarness()
    run2, prepared2 = harness2.prepare(exp2)
    result2 = TrainingEngine().train(
        experiment=exp2,
        prepared_execution=prepared2,
        train_dataset=repro_dataset,
        train_loader=loader2,
        run=run2,
    )

    assert result1.final_train_loss != result2.final_train_loss
