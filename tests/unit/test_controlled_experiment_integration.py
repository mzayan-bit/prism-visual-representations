"""Unit tests for integrating controlled data manifests into ExperimentDefinition."""

import pytest

from prism.core.enums import ModelFamily, TaskType
from prism.data.adapters import CIFAR10Adapter
from prism.data.manifests import ControlledDataReference, DatasetManifest
from prism.evaluation.configuration import (
    EvaluationConfiguration,
    MetricSpecification,
)
from prism.experiments.definitions import ExperimentDefinition
from prism.experiments.harness import ExperimentExecutionHarness
from prism.models.specifications import ModelSpecification
from prism.training.configuration import (
    OptimizerSpecification,
    TrainingConfiguration,
)

ComponentTuple = tuple[
    ModelSpecification,
    TrainingConfiguration,
    EvaluationConfiguration,
]


@pytest.fixture
def base_components() -> ComponentTuple:
    model = ModelSpecification(
        model_id="model-resnet18",
        name="ResNet-18",
        family=ModelFamily.RESNET,
        architecture="resnet18",
        compatible_tasks=[TaskType.CLASSIFICATION],
        num_classes=10,
    )
    training = TrainingConfiguration(
        epochs=10,
        batch_size=32,
        optimizer=OptimizerSpecification(type="adamw", lr=1e-3),
    )
    evaluation = EvaluationConfiguration(
        target_splits=["test"],
        metrics=[MetricSpecification(name="top1_accuracy")],
    )
    return model, training, evaluation


@pytest.mark.unit
def test_experiment_definition_with_controlled_data_reference(
    base_components: ComponentTuple,
) -> None:
    """Verify ExperimentDefinition fingerprint sensitivity to controlled data."""
    model, training, evaluation = base_components
    adapter = CIFAR10Adapter()

    dataset_base = adapter.get_dataset_manifest()
    canonical = adapter.get_canonical_manifest()
    partition = adapter.get_default_partition(seed=42)
    subsets = adapter.get_nested_subsets(seed=42)

    # 1. Base experiment definition without controlled data
    exp_base = ExperimentDefinition(
        experiment_id="exp-cifar10-base",
        name="CIFAR-10 Standard Baseline",
        task_type=TaskType.CLASSIFICATION,
        dataset=dataset_base,
        model=model,
        training=training,
        evaluation=evaluation,
    )

    # 2. Experiment with full 100% partition reference
    ref_100 = ControlledDataReference(
        canonical_manifest_fingerprint=canonical.compute_fingerprint(),
        partition_manifest_fingerprint=partition.compute_fingerprint(),
        subset_manifest_fingerprint=subsets[1.0].compute_fingerprint(),
        partition_id=partition.partition_id,
        subset_id=subsets[1.0].subset_id,
        budget_ratio=1.0,
    )
    dataset_100 = DatasetManifest(
        **dataset_base.model_dump(exclude={"controlled_data"}),
        controlled_data=ref_100,
    )
    exp_100 = ExperimentDefinition(
        experiment_id="exp-cifar10-100pct",
        name="CIFAR-10 100% Regime",
        task_type=TaskType.CLASSIFICATION,
        dataset=dataset_100,
        model=model,
        training=training,
        evaluation=evaluation,
    )

    # 3. Experiment with 10% low-budget regime reference
    ref_10 = ControlledDataReference(
        canonical_manifest_fingerprint=canonical.compute_fingerprint(),
        partition_manifest_fingerprint=partition.compute_fingerprint(),
        subset_manifest_fingerprint=subsets[0.10].compute_fingerprint(),
        partition_id=partition.partition_id,
        subset_id=subsets[0.10].subset_id,
        budget_ratio=0.10,
    )
    dataset_10 = DatasetManifest(
        **dataset_base.model_dump(exclude={"controlled_data"}),
        controlled_data=ref_10,
    )
    exp_10 = ExperimentDefinition(
        experiment_id="exp-cifar10-10pct",
        name="CIFAR-10 10% Regime",
        task_type=TaskType.CLASSIFICATION,
        dataset=dataset_10,
        model=model,
        training=training,
        evaluation=evaluation,
    )

    # Verify fingerprints distinguish regimes
    assert exp_base.compute_fingerprint() != exp_100.compute_fingerprint()
    assert exp_100.compute_fingerprint() != exp_10.compute_fingerprint()

    # 4. Verify execution harness prepares controlled experiment
    harness = ExperimentExecutionHarness()
    run, prepared = harness.prepare(exp_10)

    assert run.configuration_fingerprint == exp_10.compute_fingerprint()
    assert prepared.configuration_fingerprint == exp_10.compute_fingerprint()
