"""Unit tests for classification head replacement and optimizer freezing."""

from prism.core.enums import ModelFamily
from prism.models.cnn import ConvolutionalNeuralNetwork
from prism.models.specifications import ModelSpecification
from prism.models.transformer import VisionTransformer
from prism.training.optimizers import SGDOptimizer
from prism.transfer.freezing import create_freeze_plan
from prism.transfer.head import replace_classifier_head
from prism.transfer.specification import TransferStrategy


def test_replace_classifier_head_cnn_and_vit() -> None:
    """Test replacing classification head from 2 classes to 5 classes."""
    c, h, w = 3, 8, 8

    # 1. CNN
    cnn_spec = ModelSpecification(
        model_id="cnn_head_test",
        name="CNN Head Test",
        family=ModelFamily.CNN,
        architecture="cnn_simple",
        input_shape=(c, h, w),
        num_classes=2,
        hyperparameters={"conv_channels": [4], "kernel_sizes": [3]},
    )
    cnn_m = ConvolutionalNeuralNetwork(cnn_spec, seed=42)
    assert cnn_m.num_classes == 2

    replace_classifier_head(cnn_m, num_classes=5, seed=123)
    assert cnn_m.num_classes == 5

    # Test forward pass with new 5-class logits
    sample_img = [[[0.5 for _ in range(w)] for _ in range(h)] for _ in range(c)]
    logits = cnn_m.forward([sample_img])
    assert len(logits[0]) == 5

    # 2. ViT
    vit_spec = ModelSpecification(
        model_id="vit_head_test",
        name="ViT Head Test",
        family=ModelFamily.VISION_TRANSFORMER,
        architecture="vit_simple",
        input_shape=(c, h, w),
        num_classes=2,
        hyperparameters={
            "patch_size": 2,
            "embed_dim": 8,
            "depth": 1,
            "num_heads": 2,
            "mlp_ratio": 2.0,
            "activation": "gelu",
        },
    )
    vit_m = VisionTransformer(vit_spec, seed=44)
    replace_classifier_head(vit_m, num_classes=5, seed=123)
    assert vit_m.num_classes == 5

    vit_logits = vit_m.forward([sample_img])
    assert len(vit_logits[0]) == 5


def test_sgd_optimizer_respects_frozen_parameters() -> None:
    """Test that SGDOptimizer does NOT update parameters when omitted from plan."""
    c, h, w = 3, 8, 8
    cnn_spec = ModelSpecification(
        model_id="cnn_opt_test",
        name="CNN Opt Test",
        family=ModelFamily.CNN,
        architecture="cnn_simple",
        input_shape=(c, h, w),
        num_classes=3,
        hyperparameters={"conv_channels": [4], "kernel_sizes": [3]},
    )
    model = ConvolutionalNeuralNetwork(cnn_spec, seed=42)
    plan = create_freeze_plan(model, strategy=TransferStrategy.LINEAR_PROBE)

    initial_params = model.get_parameters()
    conv_w_initial = initial_params["conv_0_weights"]

    optimizer = SGDOptimizer(
        model=model,
        lr=0.1,
        trainable_parameters=plan.trainable_parameters,
    )

    # Forward + Backward
    sample_img = [[[0.5 for _ in range(w)] for _ in range(h)] for _ in range(c)]
    _ = model.forward([sample_img])
    model.backward([[1.0, 0.0, 0.0]])

    optimizer.step()

    updated_params = model.get_parameters()
    # Conv weights must be completely unchanged (frozen)
    assert updated_params["conv_0_weights"] == conv_w_initial
    # Classifier weights must have updated (trainable)
    assert updated_params["classifier_weights"] != initial_params["classifier_weights"]
