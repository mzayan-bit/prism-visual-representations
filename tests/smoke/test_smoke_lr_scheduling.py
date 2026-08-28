"""Smoke test for end-to-end learning rate scheduling lifecycle."""

import pytest

from prism.core.enums import (
    DevicePreference,
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


@pytest.mark.smoke
def test_smoke_lr_scheduling_lifecycle() -> None:
    """Validate full end-to-end training and evaluation with scheduled learning rate."""
    # 1. Generate synthetic dataset
    train_samples = [
        MaterializedSample(
            sample_id=f"lr_smoke/train/{i:04d}",
            source_split="train",
            source_index=i,
            data=[0.1 * (i % 5), -0.2 * (i % 3), 0.3 * (i % 4), 0.4 * (i % 2)],
            target=i % 2,
        )
        for i in range(24)
    ]
    test_samples = [
        MaterializedSample(
            sample_id=f"lr_smoke/test/{i:04d}",
            source_split="test",
            source_index=i,
            data=[0.15 * (i % 5), -0.25 * (i % 3), 0.35 * (i % 4), 0.45 * (i % 2)],
            target=i % 2,
        )
        for i in range(8)
    ]

    train_ds = MaterializedDataset(dataset_id="ds-lr-smoke", samples=train_samples)
    test_ds = MaterializedDataset(dataset_id="ds-lr-smoke", samples=test_samples)

    train_loader = DeterministicBatchLoader(
        dataset=train_ds,
        batch_size=6,
        ordering_strategy=OrderingStrategy.EPOCH_AWARE_SHUFFLE,
        seed=42,
    )
    test_loader = DeterministicBatchLoader(
        dataset=test_ds,
        batch_size=4,
        ordering_strategy=OrderingStrategy.SEQUENTIAL,
        seed=42,
    )

    dataset_manifest = DatasetManifest(
        dataset_id="ds-lr-smoke",
        name="Synthetic LR Smoke Dataset",
        version="1.0.0",
        num_classes=2,
        splits=[
            SplitSpecification(split_name="train", num_samples=24),
            SplitSpecification(split_name="test", num_samples=8),
        ],
    )

    # 2. Experiment Definition with Warmup + Cosine LR Schedule
    model_spec = ModelSpecification(
        model_id="model-mlp-lr-smoke",
        name="LR Smoke MLP Model",
        family=ModelFamily.MLP,
        architecture="mlp",
        compatible_tasks=[TaskType.CLASSIFICATION],
        input_shape=(4,),
        num_classes=2,
        hyperparameters={"hidden_dims": [12], "activation": "relu"},
    )
    training_config = TrainingConfiguration(
        epochs=8,
        batch_size=6,
        optimizer=OptimizerSpecification(type="sgd", lr=0.08, momentum=0.9),
        scheduler=SchedulerSpecification(
            type="cosine",
            warmup_epochs=2,
            min_lr=0.005,
        ),
        precision=PrecisionMode.FP32,
        device=DevicePreference.CPU,
    )
    evaluation_config = EvaluationConfiguration(
        target_splits=["test"],
        metrics=[MetricSpecification(name="top1_accuracy")],
    )
    experiment = ExperimentDefinition(
        experiment_id="exp-smoke-lr-scheduling",
        name="Smoke LR Scheduling Run",
        task_type=TaskType.CLASSIFICATION,
        dataset=dataset_manifest,
        model=model_spec,
        training=training_config,
        evaluation=evaluation_config,
        reproducibility=ReproducibilityConfiguration(seed=42, deterministic=True),
    )

    # 3. Execute through Harness
    harness = ExperimentExecutionHarness()
    run, prepared = harness.prepare(experiment)

    engine = TrainingEngine()
    result = engine.train(
        experiment=experiment,
        prepared_execution=prepared,
        train_dataset=train_ds,
        train_loader=train_loader,
        test_dataset=test_ds,
        test_loader=test_loader,
        run=run,
    )

    # 4. Verify Training Results
    assert isinstance(result, TrainingResult)
    assert result.status == RunStatus.COMPLETED
    assert result.epochs_completed == 8
    assert result.total_examples == 8 * 24
    assert result.final_train_loss > 0.0
    assert 0.0 <= result.final_train_accuracy <= 1.0

    # Verify LR telemetry metrics
    lr_metrics = [m for m in run.metric_records if m.metric_name == "learning_rate"]
    assert len(lr_metrics) == 8
    # Epoch 0: warmup step 0 (min_lr or start_lr)
    # Epoch 1: warmup step 1 (0.0425)
    # Epoch 2: peak at 0.08
    # Epoch 7: near min_lr = 0.005
    assert lr_metrics[2].value == pytest.approx(0.08, abs=1e-6)
    assert lr_metrics[7].value < 0.02

    # Verify summary metrics
    assert "final_learning_rate" in result.summary_metrics
    assert result.summary_metrics["final_learning_rate"] < 0.08

    # Verify Evaluation report
    assert len(result.evaluation_reports) == 1
    eval_rep = result.evaluation_reports[0]
    assert eval_rep.metadata.get("split_name") == "test"
    assert "test_top1_accuracy" in eval_rep.summary_metrics
