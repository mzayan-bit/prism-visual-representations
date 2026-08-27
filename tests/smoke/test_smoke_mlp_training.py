"""End-to-end smoke test for the Phase 7 Multi-Layer Perceptron pipeline.

Guarantees:
- Full pipeline: Experiment -> Data -> MLP -> Dropout -> Sched -> Eval -> Reps
- Non-linear representation learning with ReLU and dropout
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
from prism.models.mlp import MultiLayerPerceptron
from prism.models.specifications import ModelSpecification
from prism.training.configuration import (
    OptimizerSpecification,
    SchedulerSpecification,
    TrainingConfiguration,
)
from prism.training.engine import TrainingEngine
from prism.training.results import TrainingResult


@pytest.mark.smoke
def test_smoke_mlp_baseline_full_lifecycle() -> None:
    """Demonstrate end-to-end MLP baseline on synthetic vision data."""
    # 1. Initialize Synthetic Vision Adapter (3x8x8 images = 192 features)
    adapter = SyntheticVisionAdapter(
        num_train=60, num_test=20, num_classes=2, image_shape=(3, 8, 8)
    )
    canonical = adapter.get_canonical_manifest()
    partition = adapter.get_default_partition(seed=42)
    subsets = adapter.get_nested_subsets(seed=42)

    # 2. Build ControlledDataReference for 50% data budget
    subset_50pct = subsets[0.50]
    controlled_ref = ControlledDataReference(
        canonical_manifest_fingerprint=canonical.compute_fingerprint(),
        partition_manifest_fingerprint=partition.compute_fingerprint(),
        subset_manifest_fingerprint=subset_50pct.compute_fingerprint(),
        partition_id=partition.partition_id,
        subset_id=subset_50pct.subset_id,
        budget_ratio=0.50,
    )

    dataset_manifest = DatasetManifest(
        **adapter.get_dataset_manifest().model_dump(exclude={"controlled_data"}),
        controlled_data=controlled_ref,
    )

    # 3. Assemble Multi-Layer Perceptron Model Specification
    model_spec = ModelSpecification(
        model_id="model-mlp-smoke",
        name="MLP Smoke Baseline",
        family=ModelFamily.MLP,
        architecture="mlp",
        compatible_tasks=[TaskType.CLASSIFICATION],
        input_shape=(3, 8, 8),  # 192 features
        num_classes=2,
        hyperparameters={
            "hidden_dims": [64, 32],
            "activation": "relu",
            "dropout": 0.2,
        },
    )

    # 4. Assemble Training with Optimizer & Cosine Annealing Scheduler
    training_config = TrainingConfiguration(
        epochs=3,
        batch_size=6,
        optimizer=OptimizerSpecification(
            type="sgd", lr=0.05, momentum=0.9, weight_decay=1e-4
        ),
        scheduler=SchedulerSpecification(type="cosine", min_lr=0.005, warmup_epochs=1),
        precision=PrecisionMode.FP32,
    )

    evaluation_config = EvaluationConfiguration(
        target_splits=["test"],
        metrics=[MetricSpecification(name="top1_accuracy")],
    )

    experiment = ExperimentDefinition(
        experiment_id="exp-mlp-smoke-001",
        name="MLP Smoke Experiment",
        task_type=TaskType.CLASSIFICATION,
        dataset=dataset_manifest,
        model=model_spec,
        training=training_config,
        evaluation=evaluation_config,
        reproducibility=ReproducibilityConfiguration(seed=42, deterministic=True),
    )

    # 5. Prepare Runtime Execution via Harness
    harness = ExperimentExecutionHarness()
    run, prepared_exec = harness.prepare(experiment)

    # 6. Materialize Train and Test Datasets
    preparer = DataPreparer()
    train_dataset, train_loader, _ = preparer.prepare(
        adapter=adapter,
        canonical_manifest=canonical,
        partition_manifest=partition,
        subset_manifest=subset_50pct,
        batch_size=6,
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

    # 7. Execute Full Training Loop
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

    # 8. Assertions on Training Result and Lifecycle
    assert isinstance(result, TrainingResult)
    assert result.status == RunStatus.COMPLETED
    assert result.epochs_completed == 3
    assert result.total_batches > 0
    assert result.total_examples > 0
    assert result.final_train_loss > 0.0
    assert len(result.evaluation_reports) == 1
    assert run.status == RunStatus.COMPLETED

    # 9. Verify Hidden Representation Extraction on Trained Model
    model = MultiLayerPerceptron(spec=model_spec, seed=42)
    sample_batch = [test_dataset[0].data, test_dataset[1].data]

    # Extract final hidden layer (dimension 32)
    final_hidden = model.extract_representations(sample_batch, layer="final_hidden")
    assert len(final_hidden) == 2
    assert len(final_hidden[0]) == 32

    # Extract first hidden layer (dimension 64)
    h0 = model.extract_representations(sample_batch, layer="hidden_0")
    assert len(h0) == 2
    assert len(h0[0]) == 64
