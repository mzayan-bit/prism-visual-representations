"""Unit tests for complete VisionTransformer model and end-to-end backpropagation."""

import random

import pytest

from prism.core.enums import ModelFamily, TaskType
from prism.core.errors import ValidationError
from prism.models.specifications import ModelSpecification
from prism.models.transformer import VisionTransformer
from prism.training.loss import SoftmaxCrossEntropyLoss
from prism.training.optimizers import SGDOptimizer


def _create_vit_spec(
    input_shape: tuple[int, ...] = (1, 8, 8),
    patch_size: int = 4,
    embed_dim: int = 16,
    num_heads: int = 2,
    depth: int = 2,
    num_classes: int = 3,
) -> ModelSpecification:
    return ModelSpecification(
        model_id="test-vit-model",
        name="Test Vision Transformer",
        family=ModelFamily.VISION_TRANSFORMER,
        architecture="vit_tiny_custom",
        compatible_tasks=[TaskType.CLASSIFICATION],
        input_shape=input_shape,
        num_classes=num_classes,
        hyperparameters={
            "patch_size": patch_size,
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "depth": depth,
            "mlp_ratio": 2.0,
            "norm_eps": 1e-5,
            "activation": "gelu",
            "bias": True,
        },
    )


def test_vit_forward_pass_and_geometry() -> None:
    """Test full ViT forward pass transforms 4D images to classification logits."""
    spec = _create_vit_spec(
        input_shape=(1, 8, 8),
        patch_size=4,
        embed_dim=16,
        num_heads=2,
        depth=2,
        num_classes=3,
    )
    model = VisionTransformer(spec=spec, seed=42)

    # 8x8 with 4x4 patches -> 2x2 = 4 patches, with CLS token -> 5 tokens
    assert model.geometry.total_patches == 4

    rng = random.Random(123)
    x = [
        [[[rng.gauss(0.0, 1.0) for _ in range(8)] for _ in range(8)]] for _ in range(2)
    ]

    logits = model.forward(x)

    assert len(logits) == 2
    assert len(logits[0]) == 3

    # Check multi-layer attention weights
    attn_weights = model.get_attention_weights()
    assert len(attn_weights) == 2
    for layer_w in attn_weights:
        # [N=2, H=2, T=5, T=5]
        assert len(layer_w) == 2
        assert len(layer_w[0]) == 2
        assert len(layer_w[0][0]) == 5
        assert len(layer_w[0][0][0]) == 5


def test_vit_backward_to_pixels() -> None:
    """Test that analytical backward propagates all the way to input pixels."""
    spec = _create_vit_spec(
        input_shape=(1, 8, 8),
        patch_size=4,
        embed_dim=8,
        num_heads=2,
        depth=2,
        num_classes=2,
    )
    model = VisionTransformer(spec=spec, seed=99)

    x = [[[[0.5 for _ in range(8)] for _ in range(8)]] for _ in range(2)]
    d_logits = [[1.0, -1.0], [-0.5, 0.5]]

    model.zero_grad()
    _ = model.forward(x)
    model.backward(d_logits)

    # Verify parameter gradients across all stages
    grads = model.get_gradients()
    assert "patch_embed.weights" in grads
    assert "cls_token.token" in grads
    assert "pos_embed.embeddings" in grads
    assert "encoder.blocks.0.attn.w_q" in grads
    assert "encoder.blocks.1.ffn.w_1" in grads
    assert "norm.gamma" in grads
    assert "classifier.weights" in grads

    # Verify gradients are non-zero
    assert any(any(v != 0.0 for v in row) for row in grads["patch_embed.weights"])
    assert any(v != 0.0 for v in grads["cls_token.token"])
    assert any(any(v != 0.0 for v in row) for row in grads["pos_embed.embeddings"])


def test_vit_optimizer_step_reduces_loss() -> None:
    """Test that optimizer step on ViT updates weights and decreases training loss."""
    spec = _create_vit_spec(
        input_shape=(1, 4, 4),
        patch_size=2,
        embed_dim=8,
        num_heads=2,
        depth=1,
        num_classes=2,
    )
    model = VisionTransformer(spec=spec, seed=42)
    loss_fn = SoftmaxCrossEntropyLoss()
    optimizer = SGDOptimizer(model=model, lr=0.05, momentum=0.0)

    # Synthetic batch
    x = [[[[1.0 for _ in range(4)] for _ in range(4)]] for _ in range(2)]
    y = [0, 1]

    # Initial forward + loss
    logits_0 = model.forward(x)
    loss_0, d_logits_0 = loss_fn(logits_0, y)

    # Backward + step
    model.zero_grad()
    model.backward(d_logits_0)
    optimizer.step()

    # New forward + loss
    logits_1 = model.forward(x)
    loss_1, _ = loss_fn(logits_1, y)

    assert loss_1 < loss_0


def test_vit_parameter_discovery_and_restoration() -> None:
    """Test discovery and parameter setting for ViT."""
    spec = _create_vit_spec()
    model = VisionTransformer(spec=spec, seed=1)

    params = model.get_parameters()
    assert len(params) > 10

    model.set_parameters(params)
    assert model.get_state() == {}


def test_vit_validation_errors() -> None:
    """Test validation errors for invalid input shapes and specifications."""
    spec = _create_vit_spec(input_shape=(1, 7, 7), patch_size=4)
    # 7 is not divisible by 4
    with pytest.raises(ValidationError):
        VisionTransformer(spec=spec)
