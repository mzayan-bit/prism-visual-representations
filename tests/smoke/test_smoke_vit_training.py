"""End-to-end smoke test for VisionTransformer lifecycle and profiling."""

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
from prism.models.transformer import VisionTransformer
from prism.representations.attention import compute_transformer_attention_profile
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
def test_smoke_vit_full_pipeline() -> None:
    """Execute complete end-to-end ViT training and profiling pipeline."""
    # 1. Synthetic Vision Adapter (1x8x8 grayscale or 3x8x8 RGB)
    adapter = SyntheticVisionAdapter(
        num_train=24, num_test=8, num_classes=2, image_shape=(3, 8, 8)
    )
    canonical = adapter.get_canonical_manifest()
    partition = adapter.get_default_partition(seed=42)
    subsets = adapter.get_nested_subsets(seed=42)

    # 2. Controlled Data Reference
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

    # 3. Vision Transformer Specification
    model_spec = ModelSpecification(
        model_id="model-vit-smoke",
        name="Vision Transformer Smoke Baseline",
        family=ModelFamily.VISION_TRANSFORMER,
        architecture="vit_tiny",
        compatible_tasks=[TaskType.CLASSIFICATION],
        input_shape=(3, 8, 8),
        num_classes=2,
        hyperparameters={
            "patch_size": 4,
            "embed_dim": 8,
            "num_heads": 2,
            "depth": 2,
            "mlp_ratio": 2.0,
            "norm_eps": 1e-5,
            "activation": "gelu",
        },
    )

    # 4. Training Configuration with Cosine Annealing
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
        experiment_id="exp-vit-smoke-001",
        name="ViT Smoke Experiment",
        task_type=TaskType.CLASSIFICATION,
        dataset=dataset_manifest,
        model=model_spec,
        training=training_config,
        evaluation=evaluation_config,
        reproducibility=ReproducibilityConfiguration(seed=42, deterministic=True),
    )

    # 5. Harness preparation
    harness = ExperimentExecutionHarness()
    run, prepared_exec = harness.prepare(experiment)

    # 6. Data Materialization
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
        batch_size=4,
        ordering_strategy=OrderingStrategy.SEQUENTIAL,
        seed=42,
        drop_last=False,
        prepared_execution=prepared_exec,
    )

    # 7. Training Engine Run
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

    assert isinstance(result, TrainingResult)
    assert result.status == RunStatus.COMPLETED
    assert result.epochs_completed == 2
    assert result.total_batches > 0
    assert result.final_train_loss > 0.0
    assert len(result.evaluation_reports) == 1
    assert run.status == RunStatus.COMPLETED

    # 8. Gradient Flow Summary
    model = VisionTransformer(spec=model_spec, seed=42)
    sample_batch = [test_dataset[0].data, test_dataset[1].data]
    _ = model.forward(sample_batch)
    model.backward([[0.5, -0.5], [-0.5, 0.5]])

    grad_summary = compute_gradient_flow_summary(model)
    assert isinstance(grad_summary, ModelGradientFlowSummary)
    assert grad_summary.global_grad_norm_l2 > 0.0
    assert grad_summary.is_finite is True
    assert len(grad_summary.parameter_summaries) > 0

    # 9. Multi-Layer Attention Evolution Profile
    attn_weights = model.get_attention_weights()
    profile = compute_transformer_attention_profile(
        attn_weights, model_id=model.model_id
    )
    assert profile.depth == 2
    assert profile.num_heads == 2
    assert len(profile.layer_mean_entropies) == 2
    assert len(profile.layer_diagonal_masses) == 2
    assert all(ent >= 0.0 for ent in profile.layer_mean_entropies)
