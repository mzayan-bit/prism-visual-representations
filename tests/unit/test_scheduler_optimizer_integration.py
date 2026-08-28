"""Unit tests for scheduler-optimizer integration and TrainingEngine lifecycle."""

import pytest

from prism.core.enums import (
    DevicePreference,
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
from prism.models.mlp import MultiLayerPerceptron
from prism.models.specifications import ModelSpecification
from prism.training.configuration import (
    OptimizerSpecification,
    SchedulerSpecification,
    TrainingConfiguration,
)
from prism.training.engine import TrainingEngine
from prism.training.optimizers import SGDOptimizer
from prism.training.schedulers import (
    CosineAnnealingLRScheduler,
    StepLRScheduler,
    WarmupScheduler,
    create_scheduler,
)


@pytest.mark.unit
def test_create_scheduler_factory_variants() -> None:
    """Verify create_scheduler maps specifications to appropriate class structures."""
    # 1. Constant
    spec_const = SchedulerSpecification(type="constant")
    sched_const = create_scheduler(spec_const, base_lr=0.1, total_epochs=10)
    assert sched_const._get_schedule_type_name() == "constant"

    # 2. Pure Cosine
    spec_cos = SchedulerSpecification(type="cosine", min_lr=0.01)
    sched_cos = create_scheduler(spec_cos, base_lr=0.1, total_epochs=10)
    assert isinstance(sched_cos, CosineAnnealingLRScheduler)

    # 3. Warmup + Cosine
    spec_warmup_cos = SchedulerSpecification(
        type="cosine", warmup_epochs=3, min_lr=0.01
    )
    sched_warmup_cos = create_scheduler(spec_warmup_cos, base_lr=0.1, total_epochs=10)
    assert isinstance(sched_warmup_cos, WarmupScheduler)
    assert isinstance(sched_warmup_cos.after_scheduler, CosineAnnealingLRScheduler)

    # 4. Step
    spec_step = SchedulerSpecification(type="step", step_size=5, gamma=0.5)
    sched_step = create_scheduler(spec_step, base_lr=0.1, total_epochs=10)
    assert isinstance(sched_step, StepLRScheduler)


@pytest.mark.unit
def test_scheduler_optimizer_lr_update_sync() -> None:
    """Verify optimizer lr responds directly to scheduler outputs."""
    spec = ModelSpecification(
        model_id="model-mlp-test-sync",
        name="MLP Sync Test",
        family=ModelFamily.MLP,
        architecture="mlp",
        compatible_tasks=[TaskType.CLASSIFICATION],
        input_shape=(4,),
        num_classes=2,
        hyperparameters={"hidden_dims": [8], "activation": "relu"},
    )
    model = MultiLayerPerceptron(spec=spec, seed=42)
    optimizer = SGDOptimizer(model=model, lr=0.1)

    sched = StepLRScheduler(base_lr=0.1, total_epochs=10, step_size=2, gamma=0.5)

    for epoch in range(4):
        new_lr = sched.step(epoch=epoch)
        optimizer.lr = new_lr
        assert optimizer.lr == new_lr

    assert optimizer.lr == 0.05


@pytest.mark.unit
def test_training_engine_records_scheduled_learning_rate() -> None:
    """Verify TrainingEngine logs exact scheduled learning rate trajectory."""
    samples = [
        MaterializedSample(
            sample_id=f"lr_test/train/{i:04d}",
            source_split="train",
            source_index=i,
            data=[0.5, -0.2, 0.1, 0.9],
            target=i % 2,
        )
        for i in range(16)
    ]
    dataset = MaterializedDataset(dataset_id="ds-test-lr", samples=samples)
    train_loader = DeterministicBatchLoader(
        dataset=dataset,
        batch_size=4,
        ordering_strategy=OrderingStrategy.EPOCH_AWARE_SHUFFLE,
        seed=42,
    )

    dataset_manifest = DatasetManifest(
        dataset_id="ds-test-lr",
        name="Test LR Dataset",
        version="1.0.0",
        num_classes=2,
        splits=[SplitSpecification(split_name="train", num_samples=16)],
    )

    model_spec = ModelSpecification(
        model_id="model-mlp-lr-test",
        name="MLP Test LR Model",
        family=ModelFamily.MLP,
        architecture="mlp",
        compatible_tasks=[TaskType.CLASSIFICATION],
        input_shape=(4,),
        num_classes=2,
        hyperparameters={"hidden_dims": [8], "activation": "relu"},
    )
    training_cfg = TrainingConfiguration(
        epochs=6,
        batch_size=4,
        optimizer=OptimizerSpecification(type="sgd", lr=0.1),
        scheduler=SchedulerSpecification(
            type="cosine",
            warmup_epochs=2,
            min_lr=0.01,
        ),
        precision=PrecisionMode.FP32,
        device=DevicePreference.CPU,
    )
    eval_cfg = EvaluationConfiguration(
        target_splits=["train"],
        metrics=[MetricSpecification(name="top1_accuracy")],
    )
    experiment = ExperimentDefinition(
        experiment_id="exp-lr-scheduled",
        name="Scheduled LR Experiment",
        task_type=TaskType.CLASSIFICATION,
        dataset=dataset_manifest,
        model=model_spec,
        training=training_cfg,
        evaluation=eval_cfg,
        reproducibility=ReproducibilityConfiguration(seed=42, deterministic=True),
    )

    harness = ExperimentExecutionHarness()
    run, prepared = harness.prepare(experiment)

    engine = TrainingEngine()
    result = engine.train(
        experiment=experiment,
        prepared_execution=prepared,
        train_dataset=dataset,
        train_loader=train_loader,
        run=run,
    )

    assert result.epochs_completed == 6
    assert "final_learning_rate" in result.summary_metrics
    # Warmup for 2 epochs -> Cosine decay for 4 epochs
    # Epoch 0: warmup step 0 (lr = 0.0)
    # Epoch 1: warmup step 1 (lr = 0.05)
    # Epoch 2: cosine start (lr = 0.1)
    # Epoch 5 (final epoch = effective cosine step 3/4): lr ~ 0.023169
    assert result.summary_metrics["final_learning_rate"] < 0.1
    assert result.summary_metrics["final_learning_rate"] >= 0.01
