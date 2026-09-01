"""Unit tests for parameter freeze plans and trainable parameter selection."""

from prism.core.enums import ModelFamily
from prism.models.resnet import ResidualNeuralNetwork
from prism.models.specifications import ModelSpecification
from prism.models.transformer import VisionTransformer
from prism.transfer.freezing import (
    create_freeze_plan,
)
from prism.transfer.specification import TransferStrategy


def test_linear_probe_freeze_plan_resnet() -> None:
    """Test LINEAR_PROBE freezes backbone parameters and keeps classifier trainable."""
    c, h, w, num_classes = 3, 8, 8, 2
    spec = ModelSpecification(
        model_id="resnet_freeze_test",
        name="ResNet Freeze Test",
        family=ModelFamily.RESNET,
        architecture="resnet_simple",
        input_shape=(c, h, w),
        num_classes=num_classes,
        hyperparameters={
            "stem_channels": 4,
            "stage_channels": [4, 8],
            "stage_blocks": [1, 1],
            "activation": "relu",
            "use_batch_norm": False,
        },
    )
    model = ResidualNeuralNetwork(spec, seed=42)

    plan = create_freeze_plan(model, strategy=TransferStrategy.LINEAR_PROBE)
    assert plan.trainable_fraction < 0.5
    assert plan.frozen_tensors > 0
    assert plan.trainable_tensors > 0

    # Ensure classifier weights are trainable and stem/stage weights are frozen
    for p in plan.trainable_parameters:
        assert "classifier" in p or "fc" in p

    for p in plan.frozen_parameters:
        assert "stem" in p or "stage_" in p


def test_full_fine_tune_freeze_plan_vit() -> None:
    """Test that FULL_FINE_TUNE leaves all parameters trainable."""
    c, h, w, num_classes = 3, 8, 8, 2
    spec = ModelSpecification(
        model_id="vit_freeze_test",
        name="ViT Freeze Test",
        family=ModelFamily.VISION_TRANSFORMER,
        architecture="vit_simple",
        input_shape=(c, h, w),
        num_classes=num_classes,
        hyperparameters={
            "patch_size": 2,
            "embed_dim": 8,
            "depth": 2,
            "num_heads": 2,
            "mlp_ratio": 2.0,
            "activation": "gelu",
        },
    )
    model = VisionTransformer(spec, seed=44)

    plan = create_freeze_plan(model, strategy=TransferStrategy.FULL_FINE_TUNE)
    assert plan.trainable_fraction == 1.0
    assert plan.frozen_tensors == 0
    assert plan.trainable_tensors == plan.total_tensors
