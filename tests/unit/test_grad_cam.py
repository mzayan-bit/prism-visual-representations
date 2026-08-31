"""Unit tests for Grad-CAM attribution and bilinear 2D upsampling."""

import pytest

from prism.core.enums import ModelFamily
from prism.core.errors import ValidationError
from prism.explainability.attribution import TargetClassMode
from prism.explainability.grad_cam import (
    compute_grad_cam,
    upsample_bilinear_2d,
)
from prism.models.cnn import ConvolutionalNeuralNetwork
from prism.models.resnet import ResidualNeuralNetwork
from prism.models.specifications import ModelSpecification
from prism.models.transformer import VisionTransformer


def test_upsample_bilinear_2d() -> None:
    """Test deterministic bilinear 2D upsampling of low-resolution feature maps."""
    # 2x2 map
    low_res = [
        [0.0, 1.0],
        [1.0, 2.0],
    ]
    high_res = upsample_bilinear_2d(low_res, target_h=4, target_w=4)
    assert len(high_res) == 4
    assert len(high_res[0]) == 4
    assert pytest.approx(high_res[0][0]) == 0.0
    assert pytest.approx(high_res[3][3]) == 2.0
    assert high_res[1][1] > 0.0


def test_grad_cam_on_cnn_and_resnet() -> None:
    """Test Grad-CAM computation on CNN and ResNet models."""
    c, h, w, num_classes = 3, 8, 8, 3

    cnn_spec = ModelSpecification(
        model_id="cnn_cam",
        name="CNN Cam",
        family=ModelFamily.CNN,
        architecture="cnn_simple",
        input_shape=(c, h, w),
        num_classes=num_classes,
        hyperparameters={
            "conv_channels": [4, 8],
            "kernel_sizes": [3, 3],
            "activation": "relu",
            "use_batch_norm": False,
            "pooling": "none",
            "hidden_dims": [8],
        },
    )
    cnn_m = ConvolutionalNeuralNetwork(cnn_spec, seed=42)
    cnn_m.eval()

    resnet_spec = ModelSpecification(
        model_id="resnet_cam",
        name="ResNet Cam",
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
            "hidden_dims": [8],
        },
    )
    resnet_m = ResidualNeuralNetwork(resnet_spec, seed=43)
    resnet_m.eval()

    image = [[[0.5 for _ in range(w)] for _ in range(h)] for _ in range(c)]

    res_cnn = compute_grad_cam(
        model=cnn_m,
        image=image,
        target_mode=TargetClassMode.PREDICTED_CLASS,
    )
    assert res_cnn.attribution_shape == [h, w]
    assert res_cnn.statistics.is_finite is True

    res_resnet = compute_grad_cam(
        model=resnet_m,
        image=image,
        target_mode=TargetClassMode.PREDICTED_CLASS,
    )
    assert res_resnet.attribution_shape == [h, w]
    assert res_resnet.statistics.is_finite is True


def test_grad_cam_rejects_vit() -> None:
    """Test that Grad-CAM raises ValidationError when invoked on ViT."""
    c, h, w, num_classes = 3, 8, 8, 2
    vit_spec = ModelSpecification(
        model_id="vit_cam_test",
        name="ViT Cam Test",
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

    image = [[[0.5 for _ in range(w)] for _ in range(h)] for _ in range(c)]

    with pytest.raises(
        ValidationError,
        match="Grad-CAM is not applicable to VisionTransformer",
    ):
        compute_grad_cam(model=vit_m, image=image)
