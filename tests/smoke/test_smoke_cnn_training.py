"""End-to-end smoke test for the Phase 8 CNN pipeline.

Guarantees:
- Full pipeline: Experiment -> Data -> CNN -> Spatial Reps -> Scheduler -> Eval
- Multi-channel 2D convolutions, spatial pooling, and hierarchical features
- Extraction of spatial feature maps and vector representations
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
from prism.models.cnn import ConvolutionalNeuralNetwork
from prism.models.specifications import ModelSpecification
from prism.representations.contracts import (
    RepresentationBatch,
    RepresentationDescriptor,
)
from prism.training.configuration import (
    OptimizerSpecification,
    SchedulerSpecification,
    TrainingConfiguration,
)
from prism.training.engine import TrainingEngine
from prism.training.results import TrainingResult


@pytest.mark.smoke
def test_smoke_cnn_baseline_full_lifecycle() -> None:
    """Demonstrate end-to-end CNN baseline on synthetic vision data."""
    # 1. Initialize Synthetic Vision Adapter (3x8x8 images)
    adapter = SyntheticVisionAdapter(
        num_train=40, num_test=12, num_classes=2, image_shape=(3, 8, 8)
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

    # 3. Assemble Convolutional Neural Network Specification
    model_spec = ModelSpecification(
        model_id="model-cnn-smoke",
        name="CNN Smoke Baseline",
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
            "dropout": 0.1,
        },
    )

    # 4. Assemble Training with Optimizer & Cosine Annealing Scheduler
    training_config = TrainingConfiguration(
        epochs=2,
        batch_size=4,
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
        experiment_id="exp-cnn-smoke-001",
        name="CNN Smoke Experiment",
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
        batch_size=6,
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
    assert result.epochs_completed == 2
    assert result.total_batches > 0
    assert result.total_examples > 0
    assert result.final_train_loss > 0.0
    assert len(result.evaluation_reports) == 1
    assert run.status == RunStatus.COMPLETED

    # 9. Verify Spatial Feature Map and Vector Extraction
    model = ConvolutionalNeuralNetwork(spec=model_spec, seed=42)
    sample_batch = [test_dataset[0].data, test_dataset[1].data]

    # Extract final spatial feature map: [2, 16, 2, 2]
    spatial_map = model.extract_representations(sample_batch, layer="final_spatial")
    assert len(spatial_map) == 2
    assert len(spatial_map[0]) == 16
    assert len(spatial_map[0][0]) == 2
    assert len(spatial_map[0][0][0]) == 2

    # Extract final vector representation: [2, 64]
    vector_rep = model.extract_representations(sample_batch, layer="final_hidden")
    assert len(vector_rep) == 2
    assert len(vector_rep[0]) == 64

    # Build Representation Batch contract
    desc = RepresentationDescriptor(
        layer_name="final_spatial",
        feature_dim=64,
        num_samples=2,
        representation_kind="spatial",
        spatial_shape=(16, 2, 2),
        receptive_field=model.receptive_field,
        sample_ids=[test_dataset[0].sample_id, test_dataset[1].sample_id],
        model_id=model_spec.model_id,
        is_training_mode=False,
    )
    rep_batch = RepresentationBatch(descriptor=desc, embeddings=spatial_map)
    assert rep_batch.descriptor.is_spatial is True
    assert rep_batch.descriptor.receptive_field == 10
