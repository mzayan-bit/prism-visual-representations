"""Unit tests for deterministic configuration fingerprinting and hashing."""

import pytest

from prism.core.enums import ModelFamily, TaskType
from prism.data.manifests import DatasetManifest, SplitSpecification
from prism.evaluation.configuration import (
    EvaluationConfiguration,
    MetricSpecification,
)
from prism.experiments.definitions import ExperimentDefinition
from prism.experiments.hashing import compute_configuration_fingerprint
from prism.experiments.reproducibility import ReproducibilityConfiguration
from prism.models.specifications import ModelSpecification
from prism.training.configuration import (
    OptimizerSpecification,
    TrainingConfiguration,
)


@pytest.fixture
def base_experiment() -> ExperimentDefinition:
    dataset = DatasetManifest(
        dataset_id="ds-cifar10",
        name="CIFAR-10",
        compatible_tasks=[TaskType.CLASSIFICATION],
        splits=[
            SplitSpecification(split_name="train"),
            SplitSpecification(split_name="test"),
        ],
        num_classes=10,
    )
    model = ModelSpecification(
        model_id="model-resnet18",
        name="ResNet-18",
        family=ModelFamily.RESNET,
        architecture="resnet18",
        compatible_tasks=[TaskType.CLASSIFICATION],
        num_classes=10,
    )
    training = TrainingConfiguration(
        epochs=50,
        batch_size=64,
        optimizer=OptimizerSpecification(type="adamw", lr=1e-3),
    )
    evaluation = EvaluationConfiguration(
        target_splits=["test"],
        metrics=[MetricSpecification(name="top1_accuracy")],
    )
    return ExperimentDefinition(
        experiment_id="exp-fingerprint-test",
        name="Fingerprint Test",
        description="Initial description",
        task_type=TaskType.CLASSIFICATION,
        hypothesis="Initial hypothesis",
        dataset=dataset,
        model=model,
        training=training,
        evaluation=evaluation,
        tags=["tag1", "tag2"],
    )


@pytest.mark.unit
def test_fingerprint_stability(base_experiment: ExperimentDefinition) -> None:
    """Verify that identical configurations always yield the identical fingerprint."""
    hash1 = base_experiment.compute_fingerprint()
    hash2 = base_experiment.compute_fingerprint()
    assert isinstance(hash1, str)
    assert len(hash1) == 64  # SHA-256 hex digest length
    assert hash1 == hash2


@pytest.mark.unit
def test_non_semantic_fields_excluded_from_fingerprint(
    base_experiment: ExperimentDefinition,
) -> None:
    """Verify narrative changes do not alter the semantic fingerprint."""
    hash1 = base_experiment.compute_fingerprint()

    # Create modified experiment with altered narrative/tags only
    exp_modified = base_experiment.model_copy(
        update={
            "description": "Completely updated description for documentation",
            "hypothesis": "New wording for the scientific question",
            "tags": ["different_tag_a", "different_tag_b"],
        }
    )

    hash2 = exp_modified.compute_fingerprint()
    assert hash1 == hash2


@pytest.mark.unit
def test_semantic_changes_alter_fingerprint(
    base_experiment: ExperimentDefinition,
) -> None:
    """Verify that any meaningful training or model change alters the fingerprint."""
    hash_base = base_experiment.compute_fingerprint()

    # Changing learning rate
    exp_lr = base_experiment.model_copy(
        update={
            "training": base_experiment.training.model_copy(
                update={
                    "optimizer": base_experiment.training.optimizer.model_copy(
                        update={"lr": 5e-4}
                    )
                }
            )
        }
    )
    assert exp_lr.compute_fingerprint() != hash_base

    # Changing epochs
    exp_epochs = base_experiment.model_copy(
        update={"training": base_experiment.training.model_copy(update={"epochs": 100})}
    )
    assert exp_epochs.compute_fingerprint() != hash_base

    # Changing seed
    exp_seed = base_experiment.model_copy(
        update={"reproducibility": ReproducibilityConfiguration(seed=999)}
    )
    assert exp_seed.compute_fingerprint() != hash_base


@pytest.mark.unit
def test_compute_fingerprint_on_dict() -> None:
    """Verify fingerprint calculation works directly on dictionaries."""
    data1 = {"b": 2, "a": 1, "nested": {"y": "val", "x": [1, 2, 3]}}
    data2 = {"a": 1, "b": 2, "nested": {"x": [1, 2, 3], "y": "val"}}
    hash1 = compute_configuration_fingerprint(data1)
    hash2 = compute_configuration_fingerprint(data2)
    assert hash1 == hash2
