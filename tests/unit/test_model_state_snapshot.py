"""Unit tests for ModelStateSnapshot creation, validation, and restoration."""

import pytest

from prism.core.enums import ModelFamily
from prism.core.errors import ValidationError
from prism.models.cnn import ConvolutionalNeuralNetwork
from prism.models.specifications import ModelSpecification
from prism.transfer.snapshot import (
    create_model_state_snapshot,
    restore_model_from_snapshot,
)


def test_model_state_snapshot_cnn_roundtrip() -> None:
    """Test creating, validating, and restoring a CNN model snapshot."""
    c, h, w, num_classes = 3, 8, 8, 4
    spec = ModelSpecification(
        model_id="cnn_snap_test",
        name="CNN Snapshot Test",
        family=ModelFamily.CNN,
        architecture="cnn_simple",
        input_shape=(c, h, w),
        num_classes=num_classes,
        hyperparameters={
            "conv_channels": [4],
            "kernel_sizes": [3],
            "activation": "relu",
            "use_batch_norm": True,
            "pooling": "none",
            "hidden_dims": [8],
        },
    )
    model = ConvolutionalNeuralNetwork(spec, seed=42)
    model.eval()

    snapshot = create_model_state_snapshot(model, source_experiment_id="exp_cnn_src")
    assert snapshot.source_experiment_id == "exp_cnn_src"
    assert snapshot.verify_integrity() is True
    assert len(snapshot.parameters) > 0

    # Restore into new instance
    restored = restore_model_from_snapshot(snapshot, seed=99)
    assert restored.spec.model_id == "cnn_snap_test"
    assert restored.get_parameters() == model.get_parameters()


def test_model_state_snapshot_mismatched_family_rejection() -> None:
    """Test that restoring snapshot into a mismatched architecture family fails."""
    c, h, w, num_classes = 3, 8, 8, 4
    cnn_spec = ModelSpecification(
        model_id="cnn_snap",
        name="CNN",
        family=ModelFamily.CNN,
        architecture="cnn_simple",
        input_shape=(c, h, w),
        num_classes=num_classes,
        hyperparameters={"conv_channels": [4], "kernel_sizes": [3]},
    )
    cnn_m = ConvolutionalNeuralNetwork(cnn_spec, seed=42)
    snapshot = create_model_state_snapshot(cnn_m)

    vit_spec = ModelSpecification(
        model_id="vit_target",
        name="ViT",
        family=ModelFamily.VISION_TRANSFORMER,
        architecture="vit_simple",
        input_shape=(c, h, w),
        num_classes=num_classes,
        hyperparameters={"patch_size": 2, "embed_dim": 8, "depth": 1, "num_heads": 2},
    )

    with pytest.raises(ValidationError, match="family"):
        restore_model_from_snapshot(snapshot, target_spec=vit_spec)
