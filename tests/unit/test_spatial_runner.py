"""Unit tests for SpatialTransferRunner, freeze plans, and drift."""

import copy

import pytest

from prism.core.enums import ModelFamily, TaskType
from prism.models.specifications import ModelSpecification
from prism.spatial.enums import (
    PretrainingObjective,
    SpatialTaskType,
    SpatialTransferStrategy,
)
from prism.spatial.runner import SpatialTransferRunner
from prism.spatial.specification import SpatialTransferSpecification
from prism.spatial.synthetic import generate_synthetic_spatial_dataset


def _create_toy_cnn_spec() -> ModelSpecification:
    return ModelSpecification(
        model_id="toy_cnn_runner",
        name="Toy CNN Runner",
        architecture="cnn_toy",
        family=ModelFamily.CNN,
        input_shape=(3, 16, 16),
        num_classes=3,
        compatible_tasks=[TaskType.CLASSIFICATION],
        hyperparameters={
            "conv_channels": [8, 16],
            "kernel_sizes": [3, 3],
            "strides": [1, 1],
            "paddings": [1, 1],
            "use_batch_norm": False,
            "hidden_dims": [16],
        },
    )


def test_runner_frozen_spatial_probe_leaves_encoder_identical():
    """Test that FROZEN_SPATIAL_PROBE leaves encoder parameters identical."""
    spec = SpatialTransferSpecification.create(
        source_objective=PretrainingObjective.SUPERVISED,
        source_experiment_id="exp_freeze_test",
        model_spec=_create_toy_cnn_spec(),
        task_type=SpatialTaskType.OBJECT_DETECTION,
        spatial_layer="final_spatial",
        transfer_strategy=SpatialTransferStrategy.FROZEN_SPATIAL_PROBE,
        num_classes=3,
        epochs=2,
        learning_rate=0.01,
        batch_size=4,
        seed=42,
    )
    runner = SpatialTransferRunner(spec)

    # Capture initial encoder parameters and head parameters
    initial_enc_params = copy.deepcopy(runner.encoder.get_parameters())
    assert runner.detection_head is not None
    initial_head_params = copy.deepcopy(runner.detection_head.get_parameters())

    det_train, _ = generate_synthetic_spatial_dataset(
        num_samples=8, image_shape=(3, 16, 16), num_classes=3, seed=1
    )
    det_val, _ = generate_synthetic_spatial_dataset(
        num_samples=4, image_shape=(3, 16, 16), num_classes=3, seed=2
    )

    report = runner.train_and_evaluate(
        train_samples=det_train,
        eval_samples=det_val,
    )

    # 1. Verify encoder parameters remained bitwise identical
    final_enc_params = runner.encoder.get_parameters()
    assert initial_enc_params == final_enc_params

    # 2. Verify head parameters updated
    final_head_params = runner.detection_head.get_parameters()
    assert initial_head_params != final_head_params

    # 3. Verify report structure
    assert (
        report.specification.transfer_strategy
        == SpatialTransferStrategy.FROZEN_SPATIAL_PROBE
    )
    assert report.trainable_fraction == pytest.approx(
        report.head_parameters / report.total_parameters, rel=1e-3
    )
    assert report.detection_metrics is not None
    assert report.detection_metrics.mean_iou is not None
    assert report.spatial_representation_drift_cosine is not None
    # Frozen probe drift should be 0.0
    assert report.spatial_representation_drift_cosine == pytest.approx(0.0)


def test_runner_full_fine_tune_updates_encoder():
    """Test that FULL_FINE_TUNE updates encoder parameters and measures drift."""
    spec = SpatialTransferSpecification.create(
        source_objective=PretrainingObjective.RECONSTRUCTION,
        source_experiment_id="exp_full_ft_test",
        model_spec=_create_toy_cnn_spec(),
        task_type=SpatialTaskType.SEMANTIC_SEGMENTATION,
        spatial_layer="final_spatial",
        transfer_strategy=SpatialTransferStrategy.FULL_FINE_TUNE,
        num_classes=3,
        epochs=2,
        learning_rate=0.05,
        batch_size=4,
        seed=42,
    )
    runner = SpatialTransferRunner(spec)
    initial_enc_params = copy.deepcopy(runner.encoder.get_parameters())

    _, seg_train = generate_synthetic_spatial_dataset(
        num_samples=8, image_shape=(3, 16, 16), num_classes=3, seed=10
    )
    _, seg_val = generate_synthetic_spatial_dataset(
        num_samples=4, image_shape=(3, 16, 16), num_classes=3, seed=20
    )

    report = runner.train_and_evaluate(
        train_samples=seg_train,
        eval_samples=seg_val,
    )

    # 1. Verify encoder parameters updated
    final_enc_params = runner.encoder.get_parameters()
    assert initial_enc_params != final_enc_params

    # 2. Verify trainable fraction is 1.0
    assert report.trainable_fraction == pytest.approx(1.0)
    assert report.segmentation_metrics is not None
    assert report.segmentation_metrics.mean_iou is not None
    assert report.segmentation_metrics.pixel_accuracy is not None


def test_runner_partial_fine_tune():
    """Test PARTIAL_FINE_TUNE strategy."""
    spec = SpatialTransferSpecification.create(
        source_objective=PretrainingObjective.SIMCLR,
        source_experiment_id="exp_partial_ft_test",
        model_spec=_create_toy_cnn_spec(),
        task_type=SpatialTaskType.OBJECT_DETECTION,
        spatial_layer="final_spatial",
        transfer_strategy=SpatialTransferStrategy.PARTIAL_FINE_TUNE,
        num_classes=3,
        epochs=1,
        learning_rate=0.01,
        batch_size=4,
        seed=42,
    )
    runner = SpatialTransferRunner(spec)
    det_train, _ = generate_synthetic_spatial_dataset(
        num_samples=4, image_shape=(3, 16, 16), num_classes=3, seed=100
    )
    det_val, _ = generate_synthetic_spatial_dataset(
        num_samples=2, image_shape=(3, 16, 16), num_classes=3, seed=200
    )

    report = runner.train_and_evaluate(
        train_samples=det_train,
        eval_samples=det_val,
    )
    assert (
        report.specification.transfer_strategy
        == SpatialTransferStrategy.PARTIAL_FINE_TUNE
    )
    assert 0.0 < report.trainable_fraction < 1.0
