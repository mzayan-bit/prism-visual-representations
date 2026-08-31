"""Unit tests for Vision Transformer CLS-to-patch attention attribution."""

import pytest

from prism.core.enums import ModelFamily
from prism.core.errors import ValidationError
from prism.explainability.attention_attribution import compute_vit_attention_attribution
from prism.explainability.attribution import ViTAttentionHeadPolicy
from prism.models.cnn import ConvolutionalNeuralNetwork
from prism.models.specifications import ModelSpecification
from prism.models.transformer import VisionTransformer


def test_vit_attention_attribution_mean_and_specific_head() -> None:
    """Test ViT CLS attention extraction under mean heads and specific head policies."""
    c, h, w, num_classes = 3, 8, 8, 3
    vit_spec = ModelSpecification(
        model_id="vit_attn_test",
        name="ViT Attn Test",
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
    vit_m = VisionTransformer(vit_spec, seed=44)
    vit_m.eval()

    image = [[[0.5 for _ in range(w)] for _ in range(h)] for _ in range(c)]

    # 1. Mean over heads
    res_mean = compute_vit_attention_attribution(
        model=vit_m,
        image=image,
        head_policy=ViTAttentionHeadPolicy.MEAN_HEADS,
        layer_index=-1,
    )
    assert res_mean.attribution_shape == [h, w]
    assert res_mean.statistics.is_finite is True
    assert res_mean.positive_mass > 0.0

    # 2. Specific head (head 0)
    res_head0 = compute_vit_attention_attribution(
        model=vit_m,
        image=image,
        head_policy=ViTAttentionHeadPolicy.SPECIFIC_HEAD,
        head_index=0,
        layer_index=-1,
    )
    assert res_head0.attribution_shape == [h, w]
    assert res_head0.method_metadata["head_index"] == 0


def test_vit_attention_rejects_non_vit() -> None:
    """Test that ViT attention attribution raises ValidationError on CNN models."""
    c, h, w, num_classes = 3, 8, 8, 2
    cnn_spec = ModelSpecification(
        model_id="cnn_attn_test",
        name="CNN Attn Test",
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
            "hidden_dims": [4],
        },
    )
    cnn_m = ConvolutionalNeuralNetwork(cnn_spec, seed=42)
    cnn_m.eval()

    image = [[[0.5 for _ in range(w)] for _ in range(h)] for _ in range(c)]

    with pytest.raises(
        ValidationError,
        match="only supported for VisionTransformer",
    ):
        compute_vit_attention_attribution(model=cnn_m, image=image)
