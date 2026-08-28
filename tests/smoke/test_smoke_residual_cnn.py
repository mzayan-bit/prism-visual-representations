"""End-to-end smoke test for Phase 10 Residual CNN pipeline.

Guarantees:
- Full pipeline: Experiment -> Data -> ResNet -> GradFlow -> Reps -> Eval
- Dual-branch residual addition with analytical backpropagation
- Extraction of residual, shortcut, and post-addition representations
- Feature distribution summaries and stability comparisons
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
from prism.experiments.comparisons import create_residual_comparison
from prism.experiments.definitions import ExperimentDefinition
from prism.experiments.harness import ExperimentExecutionHarness
from prism.experiments.reproducibility import ReproducibilityConfiguration
from prism.models.resnet import ResidualNeuralNetwork
from prism.models.specifications import ModelSpecification
from prism.representations.contracts import (
    RepresentationBatch,
    RepresentationDescriptor,
)
from prism.representations.summaries import (
    compare_distribution_summaries,
    compute_distribution_summary,
)
from prism.training.configuration import (
    OptimizerSpecification,
    SchedulerSpecification,
    TrainingConfiguration,
)
from prism.training.engine import TrainingEngine
from prism.training.gradient_flow import (
    ModelGradientFlowSummary,
    compute_gradient_flow_summary,
)
from prism.training.results import TrainingResult


@pytest.mark.smoke
def test_smoke_residual_cnn_full_lifecycle() -> None:
    """Demonstrate end-to-end Residual CNN baseline with gradient flow tracking."""
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

    # 3. Assemble Residual Neural Network Specification
    model_spec = ModelSpecification(
        model_id="model-resnet-smoke",
        name="Residual CNN Smoke Baseline",
        family=ModelFamily.RESNET,
        architecture="resnet",
        compatible_tasks=[TaskType.CLASSIFICATION],
        input_shape=(3, 8, 8),
        num_classes=2,
        hyperparameters={
            "stem_channels": 8,
            "stage_widths": [8, 16],
            "blocks_per_stage": [1, 1],
            "strides": [1, 2],
            "activation": "relu",
            "normalization": "batch_norm",
            "norm_eps": 1e-5,
            "norm_momentum": 0.1,
            "norm_affine": True,
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
        experiment_id="exp-resnet-smoke-001",
        name="Residual CNN Smoke Experiment",
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
    assert result.final_train_loss > 0.0
    assert len(result.evaluation_reports) == 1
    assert run.status == RunStatus.COMPLETED

    # 9. Compute Model Gradient Flow Summary
    model = ResidualNeuralNetwork(spec=model_spec, seed=42)
    sample_batch = [test_dataset[0].data, test_dataset[1].data]
    _ = model.forward(sample_batch)
    model.backward([[0.5, -0.5], [-0.5, 0.5]])

    grad_summary = compute_gradient_flow_summary(model)
    assert isinstance(grad_summary, ModelGradientFlowSummary)
    assert grad_summary.global_grad_norm_l2 > 0.0
    assert grad_summary.is_finite is True
    assert len(grad_summary.parameter_summaries) > 0

    # 10. Extract Intermediate Residual, Shortcut, and Post-Addition Representations
    res_map = model.extract_representations(
        sample_batch, layer="stage_0_block_0_residual"
    )
    sc_map = model.extract_representations(
        sample_batch, layer="stage_0_block_0_shortcut"
    )
    post_add_map = model.extract_representations(
        sample_batch, layer="stage_0_block_0_post_add"
    )

    assert len(res_map) == 2
    assert len(sc_map) == 2
    assert len(post_add_map) == 2

    # 11. Compute and Compare Feature Distribution Summaries across Branches
    sum_res = compute_distribution_summary(res_map)
    sum_sc = compute_distribution_summary(sc_map)
    sum_add = compute_distribution_summary(post_add_map)

    assert sum_res.is_finite is True
    assert sum_sc.is_finite is True
    assert sum_add.is_finite is True

    branch_stability = compare_distribution_summaries(sum_res, sum_sc)
    assert "mean_shift" in branch_stability

    # 12. Controlled Comparison Helper
    comp = create_residual_comparison(
        comparison_id="comp-smoke-resnet",
        name="Smoke Plain vs Residual Comparison",
        baseline_experiment_id="exp-cnn-smoke-001",
        candidate_experiment_id="exp-resnet-smoke-001",
        dataset_fingerprint=canonical.compute_fingerprint(),
        seed=42,
    )
    assert comp.compute_fingerprint() is not None

    # 13. Build Representation Batch container
    desc = RepresentationDescriptor(
        layer_name="stage_0_block_0_post_add",
        feature_dim=64,
        num_samples=2,
        representation_kind="spatial",
        spatial_shape=(8, 8, 8),
        receptive_field=7,
        sample_ids=[test_dataset[0].sample_id, test_dataset[1].sample_id],
        model_id=model_spec.model_id,
        is_training_mode=False,
    )
    rep_batch = RepresentationBatch(descriptor=desc, embeddings=post_add_map)
    assert rep_batch.descriptor.is_spatial is True
