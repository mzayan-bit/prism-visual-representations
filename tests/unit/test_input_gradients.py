"""Unit tests for input gradient saliency and Gradient x Input."""

import pytest

from prism.core.enums import ModelFamily
from prism.explainability.attribution import TargetClassMode
from prism.explainability.gradients import (
    compute_gradient_x_input,
    compute_input_gradient_saliency,
)
from prism.models.cnn import ConvolutionalNeuralNetwork
from prism.models.resnet import ResidualNeuralNetwork
from prism.models.specifications import ModelSpecification
from prism.models.transformer import VisionTransformer


@pytest.fixture
def test_models() -> tuple[
    ConvolutionalNeuralNetwork, ResidualNeuralNetwork, VisionTransformer
]:
    """Build small CNN, ResNet, and ViT models for gradient attribution tests."""
    c, h, w, num_classes = 3, 8, 8, 3

    cnn_spec = ModelSpecification(
        model_id="cnn_test",
        name="CNN Test",
        family=ModelFamily.CNN,
        architecture="cnn_simple",
        input_shape=(c, h, w),
        num_classes=num_classes,
        hyperparameters={
            "conv_channels": [4],
            "kernel_sizes": [3],
            "activation": "relu",
            "use_batch_norm": False,
            "pooling": "none",
            "hidden_dims": [8],
        },
    )
    cnn_m = ConvolutionalNeuralNetwork(cnn_spec, seed=42)
    cnn_m.eval()

    resnet_spec = ModelSpecification(
        model_id="resnet_test",
        name="ResNet Test",
        family=ModelFamily.RESNET,
        architecture="resnet_simple",
        input_shape=(c, h, w),
        num_classes=num_classes,
        hyperparameters={
            "stem_channels": 4,
            "stage_channels": [4],
            "stage_blocks": [1],
            "activation": "relu",
            "use_batch_norm": False,
            "hidden_dims": [8],
        },
    )
    resnet_m = ResidualNeuralNetwork(resnet_spec, seed=43)
    resnet_m.eval()

    vit_spec = ModelSpecification(
        model_id="vit_test",
        name="ViT Test",
        family=ModelFamily.VISION_TRANSFORMER,
        architecture="vit_simple",
        input_shape=(c, h, w),
        num_classes=num_classes,
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
    vit_m.eval()

    return cnn_m, resnet_m, vit_m


def test_input_gradient_saliency_across_architectures(
    test_models: tuple[
        ConvolutionalNeuralNetwork, ResidualNeuralNetwork, VisionTransformer
    ],
) -> None:
    """Test input gradient computation produces matching spatial dimensions."""
    cnn_m, resnet_m, vit_m = test_models
    c, h, w = 3, 8, 8
    image = [[[0.5 for _ in range(w)] for _ in range(h)] for _ in range(c)]

    for model in [cnn_m, resnet_m, vit_m]:
        res = compute_input_gradient_saliency(
            model=model,
            image=image,
            target_mode=TargetClassMode.PREDICTED_CLASS,
        )
        assert res.attribution_shape == [h, w]
        assert len(res.normalized_attribution_map) == h
        assert len(res.normalized_attribution_map[0]) == w
        assert res.statistics.is_finite is True
        assert res.statistics.total_absolute_mass >= 0.0


def test_gradient_x_input_computation(
    test_models: tuple[
        ConvolutionalNeuralNetwork, ResidualNeuralNetwork, VisionTransformer
    ],
) -> None:
    """Test Gradient x Input scales gradients with input pixel intensities."""
    cnn_m, _, _ = test_models
    c, h, w = 3, 8, 8

    # All-zero image should result in zero Gradient x Input
    zero_image = [[[0.0 for _ in range(w)] for _ in range(h)] for _ in range(c)]
    res_zero = compute_gradient_x_input(
        model=cnn_m,
        image=zero_image,
        target_mode=TargetClassMode.PREDICTED_CLASS,
    )
    assert pytest.approx(res_zero.positive_mass + res_zero.negative_mass) == 0.0

    # Non-zero image
    ones_image = [[[1.0 for _ in range(w)] for _ in range(h)] for _ in range(c)]
    res_ones = compute_gradient_x_input(
        model=cnn_m,
        image=ones_image,
        target_mode=TargetClassMode.PREDICTED_CLASS,
    )
    assert res_ones.attribution_shape == [h, w]


def test_explicit_target_class_selection(
    test_models: tuple[
        ConvolutionalNeuralNetwork, ResidualNeuralNetwork, VisionTransformer
    ],
) -> None:
    """Test target class override selects intended logit gradient."""
    cnn_m, _, _ = test_models
    c, h, w = 3, 8, 8
    image = [[[0.2 for _ in range(w)] for _ in range(h)] for _ in range(c)]

    res_c0 = compute_input_gradient_saliency(
        model=cnn_m,
        image=image,
        target_mode=TargetClassMode.EXPLICIT_CLASS,
        explicit_target_class=0,
    )
    res_c1 = compute_input_gradient_saliency(
        model=cnn_m,
        image=image,
        target_mode=TargetClassMode.EXPLICIT_CLASS,
        explicit_target_class=1,
    )

    assert res_c0.target_class == 0
    assert res_c1.target_class == 1
