"""Unit tests for TemporalTrainingRunner and transfer freezing strategies."""

import copy
from typing import Any

from prism.core.enums import ModelFamily, SplitName, TaskType
from prism.models.cnn import ConvolutionalNeuralNetwork
from prism.models.specifications import ModelSpecification
from prism.temporal.enums import (
    PretrainingObjective,
    TemporalAggregationType,
    TemporalTransferStrategy,
)
from prism.temporal.runner import TemporalTrainingRunner
from prism.temporal.specification import TemporalTransferSpecification
from prism.temporal.synthetic import SyntheticVideoGenerator


def _create_toy_cnn() -> ConvolutionalNeuralNetwork:
    spec = ModelSpecification(
        model_id="runner_cnn",
        name="Runner CNN",
        architecture="cnn_toy",
        family=ModelFamily.CNN,
        input_shape=(3, 16, 16),
        num_classes=4,
        compatible_tasks=[TaskType.CLASSIFICATION],
        hyperparameters={
            "conv_channels": [4, 8],
            "kernel_sizes": [3, 3],
            "strides": [1, 1],
            "paddings": [1, 1],
            "use_batch_norm": False,
            "hidden_dims": [16],
        },
    )
    return ConvolutionalNeuralNetwork(spec=spec, seed=42)


def _clone_params(params: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(params)


def test_runner_frozen_frame_encoder_immutability() -> None:
    gen = SyntheticVideoGenerator(num_frames=4, height=16, width=16, seed=42)
    train_samples = gen.generate_dataset(num_samples=4, split=SplitName.TRAIN)
    val_samples = gen.generate_dataset(num_samples=4, split=SplitName.VAL)

    cnn = _create_toy_cnn()
    initial_params = _clone_params(cnn.get_parameters())

    spec = TemporalTransferSpecification(
        source_objective=PretrainingObjective.SUPERVISED,
        architecture=ModelFamily.CNN,
        selected_layer="final_hidden",
        temporal_aggregator=TemporalAggregationType.LEARNED_TEMPORAL_POOLING,
        transfer_strategy=TemporalTransferStrategy.FROZEN_FRAME_ENCODER,
        epochs=3,
        learning_rate=0.1,
        seed=42,
    )

    runner = TemporalTrainingRunner(
        spec=spec,
        model=cnn,
        train_samples=train_samples,
        val_samples=val_samples,
    )

    report = runner.run_transfer()

    # Verify encoder parameters remained 100% bitwise identical
    final_params = cnn.get_parameters()
    for k, init_val in initial_params.items():
        assert final_params[k] == init_val, f"Encoder parameter {k} changed!"

    # Verify report is populated
    assert report.video_accuracy >= 0.0
    assert len(report.drift_curve) == 4
    assert len(report.robustness_summaries) > 0


def test_runner_full_fine_tune_updates_encoder() -> None:
    gen = SyntheticVideoGenerator(num_frames=4, height=16, width=16, seed=42)
    train_samples = gen.generate_dataset(num_samples=4, split=SplitName.TRAIN)
    val_samples = gen.generate_dataset(num_samples=4, split=SplitName.VAL)

    cnn = _create_toy_cnn()
    initial_params = _clone_params(cnn.get_parameters())

    spec = TemporalTransferSpecification(
        source_objective=PretrainingObjective.SCRATCH,
        architecture=ModelFamily.CNN,
        selected_layer="final_hidden",
        temporal_aggregator=TemporalAggregationType.MEAN_POOL,
        transfer_strategy=TemporalTransferStrategy.FULL_FINE_TUNE,
        epochs=3,
        learning_rate=0.1,
        seed=42,
    )

    runner = TemporalTrainingRunner(
        spec=spec,
        model=cnn,
        train_samples=train_samples,
        val_samples=val_samples,
    )

    report = runner.run_transfer()
    assert report.trainable_fraction == 1.0

    # Verify encoder parameters updated
    final_params = cnn.get_parameters()
    has_changed = False
    for k, init_val in initial_params.items():
        if final_params[k] != init_val:
            has_changed = True
            break
    assert has_changed, "Encoder parameters should update during full fine-tuning!"


def test_runner_report_serialization() -> None:
    gen = SyntheticVideoGenerator(num_frames=4, height=16, width=16, seed=42)
    train_samples = gen.generate_dataset(num_samples=2, split=SplitName.TRAIN)
    val_samples = gen.generate_dataset(num_samples=2, split=SplitName.VAL)

    cnn = _create_toy_cnn()
    spec = TemporalTransferSpecification(
        source_objective=PretrainingObjective.RECONSTRUCTION,
        architecture=ModelFamily.CNN,
        selected_layer="final_hidden",
        temporal_aggregator=TemporalAggregationType.SIMPLE_RNN,
        transfer_strategy=TemporalTransferStrategy.FROZEN_FRAME_ENCODER,
        epochs=1,
        seed=42,
    )
    runner = TemporalTrainingRunner(
        spec=spec,
        model=cnn,
        train_samples=train_samples,
        val_samples=val_samples,
    )
    report = runner.run_transfer()
    d = report.to_dict()
    assert "spec" in d
    assert "video_accuracy" in d
    assert "robustness_summaries" in d
