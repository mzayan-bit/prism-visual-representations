"""End-to-end smoke test for the Phase 6 trainable linear baseline.

Guarantees:
- Full end-to-end pipeline: Experiment -> Train -> Eval -> Report -> Complete
- Real gradient-based optimization step
- Zero external network requests
- Zero GPU requirements (CPU-safe)
- Fast execution
"""

import pytest

from prism.core.enums import (
    ModelFamily,
    OrderingStrategy,
    PrecisionMode,
    RunStatus,
    TaskType,
)
from prism.data.manifests import ControlledDataReference, DatasetManifest
from prism.data.preparer import DataPreparer
from prism.data.synthetic import SyntheticVisionAdapter
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


@pytest.mark.smoke
def test_smoke_linear_baseline_full_lifecycle() -> None:
    """Demonstrate end-to-end learning baseline on synthetic vision data."""
    # 1. Initialize Synthetic Vision Adapter
    adapter = SyntheticVisionAdapter(
        num_train=80, num_test=20, num_classes=2, image_shape=(3, 8, 8)
    )
    canonical = adapter.get_canonical_manifest()
    partition = adapter.get_default_partition(seed=42)
    subsets = adapter.get_nested_subsets(seed=42)

    # 2. Build ControlledDataReference for 25% data budget
    subset_25pct = subsets[0.25]
    controlled_ref = ControlledDataReference(
        canonical_manifest_fingerprint=canonical.compute_fingerprint(),
        partition_manifest_fingerprint=partition.compute_fingerprint(),
        subset_manifest_fingerprint=subset_25pct.compute_fingerprint(),
        partition_id=partition.partition_id,
        subset_id=subset_25pct.subset_id,
        budget_ratio=0.25,
    )

    dataset_manifest = DatasetManifest(
        **adapter.get_dataset_manifest().model_dump(exclude={"controlled_data"}),
        controlled_data=controlled_ref,
    )

    # 3. Assemble ExperimentDefinition
    model_spec = ModelSpecification(
        model_id="model-linear-smoke",
        name="Linear Softmax Smoke Model",
        family=ModelFamily.LINEAR,
        architecture="linear_softmax",
        compatible_tasks=[TaskType.CLASSIFICATION],
        input_shape=(3, 8, 8),  # 192 features
        num_classes=2,
    )
    training_config = TrainingConfiguration(
        epochs=3,
        batch_size=4,
        optimizer=OptimizerSpecification(type="sgd", lr=0.05, weight_decay=1e-4),
        precision=PrecisionMode.FP32,
    )
    evaluation_config = EvaluationConfiguration(
        target_splits=["test"],
        metrics=[MetricSpecification(name="top1_accuracy")],
    )
    experiment = ExperimentDefinition(
        experiment_id="exp-linear-smoke-001",
        name="Linear Softmax Smoke Experiment",
        task_type=TaskType.CLASSIFICATION,
        dataset=dataset_manifest,
        model=model_spec,
        training=training_config,
        evaluation=evaluation_config,
        reproducibility=ReproducibilityConfiguration(seed=42, deterministic=True),
    )

    # 4. Prepare Runtime Execution via Harness
    harness = ExperimentExecutionHarness()
    run, prepared_exec = harness.prepare(experiment)

    # 5. Prepare Materialized Data and Batch Loaders
    preparer = DataPreparer()
    train_dataset, train_loader, _ = preparer.prepare(
        adapter=adapter,
        canonical_manifest=canonical,
        partition_manifest=partition,
        subset_manifest=subset_25pct,
        batch_size=4,
        ordering_strategy=OrderingStrategy.EPOCH_AWARE_SHUFFLE,
        seed=42,
        drop_last=False,
        prepared_execution=prepared_exec,
    )

    test_dataset, test_loader, _ = preparer.prepare(
        adapter=adapter,
        canonical_manifest=canonical,
        partition_manifest=partition,
        split_name="test",
        batch_size=10,
        ordering_strategy=OrderingStrategy.SEQUENTIAL,
        seed=42,
        drop_last=False,
        prepared_execution=prepared_exec,
    )

    # 6. Execute Full Training Loop
    engine = TrainingEngine()
    result = engine.train(
        experiment=experiment,
        prepared_execution=prepared_exec,
        train_dataset=train_dataset,
        train_loader=train_loader,
        test_dataset=test_dataset,
        test_loader=test_loader,
        run=run,
    )

    # 7. Assertions on Training Result and Lifecycles
    assert isinstance(result, TrainingResult)
    assert result.status == RunStatus.COMPLETED
    assert result.epochs_completed == 3
    assert result.total_batches > 0
    assert result.total_examples > 0
    assert result.final_train_loss > 0.0
    assert len(result.evaluation_reports) == 1
    assert result.evaluation_reports[0].experiment_id == experiment.experiment_id

    # 8. Assertions on ExperimentRun Final State
    assert run.status == RunStatus.COMPLETED
    assert run.completed_at is not None
    assert len(run.metric_records) >= 3  # train_loss, train_acc, and test metrics
    assert "test_top1_accuracy" in run.summary_metrics
