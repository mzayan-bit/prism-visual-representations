"""Unit tests for stacked TransformerEncoder architecture."""

import random

import pytest

from prism.core.errors import ValidationError
from prism.models.transformer import TransformerEncoder


def test_stacked_encoder_depth_and_shapes() -> None:
    """Test stacked encoder with depth 3 processes sequence correctly."""
    depth = 3
    embed_dim = 8
    num_heads = 2
    encoder = TransformerEncoder(
        depth=depth,
        embed_dim=embed_dim,
        num_heads=num_heads,
        seed=42,
    )

    assert len(encoder.blocks) == depth

    # Ensure blocks have distinct initial parameter weights
    b0_wq = encoder.blocks[0].attn.w_q
    b1_wq = encoder.blocks[1].attn.w_q
    assert b0_wq != b1_wq

    rng = random.Random(99)
    x = [
        [[rng.gauss(0.0, 1.0) for _ in range(embed_dim)] for _ in range(4)]
        for _ in range(2)
    ]

    out = encoder.forward(x)
    assert len(out) == 2
    assert len(out[0]) == 4
    assert len(out[0][0]) == embed_dim

    # Multi-layer attention weights
    attn_weights = encoder.get_attention_weights()
    assert len(attn_weights) == depth
    for layer_w in attn_weights:
        # [N=2, H=2, T=4, T=4]
        assert len(layer_w) == 2
        assert len(layer_w[0]) == num_heads
        assert len(layer_w[0][0]) == 4
        assert len(layer_w[0][0][0]) == 4


def test_stacked_encoder_backward_propagation() -> None:
    """Test backpropagation through multi-layer stacked encoder."""
    depth = 2
    embed_dim = 4
    num_heads = 2
    encoder = TransformerEncoder(
        depth=depth,
        embed_dim=embed_dim,
        num_heads=num_heads,
        seed=10,
    )

    x = [[[0.1 for _ in range(embed_dim)] for _ in range(3)] for _ in range(2)]
    d_out = [[[1.0 for _ in range(embed_dim)] for _ in range(3)] for _ in range(2)]

    encoder.zero_grad()
    _ = encoder.forward(x)
    dx = encoder.backward(d_out)

    assert len(dx) == 2
    assert len(dx[0]) == 3
    assert len(dx[0][0]) == embed_dim

    # Verify all blocks accumulated gradients
    grads = encoder.get_gradients()
    assert "blocks.0.ffn.w_1" in grads
    assert "blocks.1.ffn.w_1" in grads


def test_stacked_encoder_parameter_namespacing() -> None:
    """Test parameter loading and discovery across blocks."""
    encoder = TransformerEncoder(
        depth=2,
        embed_dim=4,
        num_heads=2,
        seed=42,
    )
    params = encoder.get_parameters()
    assert any(k.startswith("blocks.0.") for k in params)
    assert any(k.startswith("blocks.1.") for k in params)

    encoder.set_parameters(params)
    assert encoder.get_state() == {}


def test_stacked_encoder_validation() -> None:
    """Test validation errors for invalid depth."""
    with pytest.raises(ValidationError):
        TransformerEncoder(depth=0, embed_dim=4, num_heads=2)
