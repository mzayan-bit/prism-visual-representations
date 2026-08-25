"""Unit tests for ExperimentDefinition, immutability, and serialization."""

import pytest
from pydantic import ValidationError as PydanticValidationError

from prism.core.enums import ModelFamily, TaskType
from prism.core.errors import ValidationError
from prism.data.manifests import DatasetManifest, SplitSpecification
from prism.evaluation.configuration import (
    EvaluationConfiguration,
    MetricSpecification,
)
from prism.experiments.definitions import ExperimentDefinition
from prism.experiments.reproducibility import ReproducibilityConfiguration
from prism.models.specifications import ModelSpecification
from prism.training.configuration import (
    OptimizerSpecification,
    TrainingConfiguration,
)


@pytest.fixture
def valid_dataset() -> DatasetManifest:
    return DatasetManifest(
        dataset_id="ds-cifar10",
        name="CIFAR-10",
        compatible_tasks=[TaskType.CLASSIFICATION],
        splits=[
            SplitSpecification(split_name="train"),
            SplitSpecification(split_name="test"),
        ],
        num_classes=10,
    )


@pytest.fixture
def valid_model() -> ModelSpecification:
    return ModelSpecification(
        model_id="model-resnet18",
        name="ResNet-18",
        family=ModelFamily.RESNET,
        architecture="resnet18",
        compatible_tasks=[TaskType.CLASSIFICATION],
        num_classes=10,
    )


@pytest.fixture
def valid_training() -> TrainingConfiguration:
    return TrainingConfiguration(
        epochs=50,
        batch_size=64,
        optimizer=OptimizerSpecification(type="adamw", lr=1e-3),
    )


@pytest.fixture
def valid_evaluation() -> EvaluationConfiguration:
    return EvaluationConfiguration(
        target_splits=["test"],
        metrics=[MetricSpecification(name="top1_accuracy")],
    )


@pytest.mark.unit
def test_valid_experiment_definition(
    valid_dataset: DatasetManifest,
    valid_model: ModelSpecification,
    valid_training: TrainingConfiguration,
    valid_evaluation: EvaluationConfiguration,
) -> None:
    """Verify construction of a complete ExperimentDefinition."""
    exp = ExperimentDefinition(
        experiment_id="exp-cifar10-resnet18",
        name="CIFAR-10 ResNet-18 Baseline",
        description="Controlled baseline probing linear classification",
        task_type=TaskType.CLASSIFICATION,
        hypothesis="ResNet-18 achieves >90% accuracy under standard preprocessing",
        dataset=valid_dataset,
        model=valid_model,
        training=valid_training,
        evaluation=valid_evaluation,
        reproducibility=ReproducibilityConfiguration(seed=42, deterministic=True),
        tags=["baseline", "cifar10", "resnet"],
    )

    assert exp.experiment_id == "exp-cifar10-resnet18"
    assert exp.task_type == TaskType.CLASSIFICATION
    assert exp.reproducibility.seed == 42
    assert len(exp.tags) == 3


@pytest.mark.unit
def test_experiment_definition_immutability(
    valid_dataset: DatasetManifest,
    valid_model: ModelSpecification,
    valid_training: TrainingConfiguration,
    valid_evaluation: EvaluationConfiguration,
) -> None:
    """Verify that ExperimentDefinition instances are frozen and cannot be mutated."""
    exp = ExperimentDefinition(
        experiment_id="exp-frozen-test",
        name="Frozen Test",
        task_type=TaskType.CLASSIFICATION,
        dataset=valid_dataset,
        model=valid_model,
        training=valid_training,
        evaluation=valid_evaluation,
    )

    with pytest.raises((PydanticValidationError, TypeError)):
        exp.name = "Mutated Name"


@pytest.mark.unit
def test_serialization_round_trip(
    valid_dataset: DatasetManifest,
    valid_model: ModelSpecification,
    valid_training: TrainingConfiguration,
    valid_evaluation: EvaluationConfiguration,
) -> None:
    """Verify JSON and dict serialization / deserialization preserve all data."""
    exp = ExperimentDefinition(
        experiment_id="exp-round-trip",
        name="Round Trip Experiment",
        task_type=TaskType.CLASSIFICATION,
        dataset=valid_dataset,
        model=valid_model,
        training=valid_training,
        evaluation=valid_evaluation,
        tags=["test_tag"],
    )

    # Dict round trip
    exp_dict = exp.to_dict()
    restored_from_dict = ExperimentDefinition.from_dict(exp_dict)
    assert restored_from_dict.experiment_id == exp.experiment_id
    assert restored_from_dict.task_type == exp.task_type
    assert restored_from_dict.training.epochs == exp.training.epochs

    # JSON round trip
    json_str = exp.to_json()
    restored_from_json = ExperimentDefinition.from_json(json_str)
    assert restored_from_json.experiment_id == exp.experiment_id
    assert restored_from_json.model.architecture == exp.model.architecture
    assert restored_from_json.tags == ["test_tag"]


@pytest.mark.unit
def test_task_incompatibility_rejection(
    valid_model: ModelSpecification,
    valid_training: TrainingConfiguration,
    valid_evaluation: EvaluationConfiguration,
) -> None:
    """Verify that task mismatch between experiment and dataset/model is rejected."""
    # Dataset only supports OBJECT_DETECTION
    detection_dataset = DatasetManifest(
        dataset_id="ds-coco",
        name="COCO",
        compatible_tasks=[TaskType.OBJECT_DETECTION],
        splits=[SplitSpecification(split_name="val")],
    )

    with pytest.raises(ValidationError, match="does not support task"):
        ExperimentDefinition(
            experiment_id="exp-task-mismatch",
            name="Task Mismatch",
            task_type=TaskType.CLASSIFICATION,
            dataset=detection_dataset,  # incompatible task
            model=valid_model,
            training=valid_training,
            evaluation=valid_evaluation,
        )


@pytest.mark.unit
def test_class_count_mismatch_rejection(
    valid_dataset: DatasetManifest,
    valid_training: TrainingConfiguration,
    valid_evaluation: EvaluationConfiguration,
) -> None:
    """Verify class count mismatch between dataset and model is rejected."""
    # Dataset has 10 classes, model has 100 classes
    mismatched_model = ModelSpecification(
        model_id="model-resnet18-cifar100",
        name="ResNet-18 CIFAR-100",
        family=ModelFamily.RESNET,
        architecture="resnet18",
        compatible_tasks=[TaskType.CLASSIFICATION],
        num_classes=100,  # 100 vs 10
    )

    with pytest.raises(ValidationError, match="Class count mismatch"):
        ExperimentDefinition(
            experiment_id="exp-class-mismatch",
            name="Class Mismatch Experiment",
            task_type=TaskType.CLASSIFICATION,
            dataset=valid_dataset,
            model=mismatched_model,
            training=valid_training,
            evaluation=valid_evaluation,
        )
