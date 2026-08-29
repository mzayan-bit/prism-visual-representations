"""Unit tests for pre-norm TransformerEncoderBlock and dual residual gradients."""

import random

import pytest

from prism.core.errors import ValidationError
from prism.models.transformer import TransformerEncoderBlock


def test_encoder_block_forward_and_residual_properties() -> None:
    """Test that pre-norm encoder block processes sequence and preserves residual."""
    embed_dim = 8
    num_heads = 2
    block = TransformerEncoderBlock(
        embed_dim=embed_dim,
        num_heads=num_heads,
        hidden_dim=16,
        norm_eps=1e-5,
        bias=True,
        seed=42,
    )

    rng = random.Random(123)
    x = [
        [[rng.gauss(0.0, 1.0) for _ in range(embed_dim)] for _ in range(4)]
        for _ in range(2)
    ]

    out = block.forward(x)

    assert len(out) == 2
    assert len(out[0]) == 4
    assert len(out[0][0]) == embed_dim

    # Verify intermediate representations are captured
    assert block.last_attention_weights is not None
    assert block.last_u is not None
    assert block.last_ln1_out is not None
    assert block.last_ln2_out is not None
    assert block.last_output == out


def test_encoder_block_gradient_numerical_check() -> None:
    """Validate dual residual branch gradient accumulation w.r.t finite differences."""
    embed_dim = 4
    num_heads = 2
    block = TransformerEncoderBlock(
        embed_dim=embed_dim,
        num_heads=num_heads,
        hidden_dim=8,
        norm_eps=1e-5,
        bias=True,
        seed=77,
    )

    rng = random.Random(55)
    x = [
        [[rng.gauss(0.0, 1.0) for _ in range(embed_dim)] for _ in range(3)]
        for _ in range(2)
    ]
    d_out = [
        [[rng.gauss(0.0, 1.0) for _ in range(embed_dim)] for _ in range(3)]
        for _ in range(2)
    ]

    block.zero_grad()
    _ = block.forward(x)
    dx_analytic = block.backward(d_out)

    eps = 1e-5
    for n in range(2):
        for t in range(3):
            for j in range(embed_dim):
                x_pos = [[list(tok) for tok in sample] for sample in x]
                x_neg = [[list(tok) for tok in sample] for sample in x]
                x_pos[n][t][j] += eps
                x_neg[n][t][j] -= eps

                out_pos = block.forward(x_pos)
                out_neg = block.forward(x_neg)

                obj_pos = sum(
                    out_pos[b][s][k] * d_out[b][s][k]
                    for b in range(2)
                    for s in range(3)
                    for k in range(embed_dim)
                )
                obj_neg = sum(
                    out_neg[b][s][k] * d_out[b][s][k]
                    for b in range(2)
                    for s in range(3)
                    for k in range(embed_dim)
                )
                grad_num = (obj_pos - obj_neg) / (2.0 * eps)
                assert abs(dx_analytic[n][t][j] - grad_num) < 1e-4


def test_encoder_block_parameters_discovery() -> None:
    """Test hierarchical parameter discovery and restoration across sub-layers."""
    block = TransformerEncoderBlock(embed_dim=4, num_heads=2, seed=10)
    params = block.get_parameters()

    assert "ln1.gamma" in params
    assert "ln1.beta" in params
    assert "attn.w_q" in params
    assert "attn.w_o" in params
    assert "ln2.gamma" in params
    assert "ffn.w_1" in params
    assert "ffn.w_2" in params

    block.set_parameters(params)
    block.zero_grad()
    grads = block.get_gradients()
    assert all(
        all(v == 0.0 for v in row) if isinstance(row, list) else row == 0.0
        for g in grads.values()
        for row in (g if isinstance(g, list) else [g])
    )


def test_encoder_block_validation_errors() -> None:
    """Test invalid block configuration throws ValidationError."""
    with pytest.raises(ValidationError):
        TransformerEncoderBlock(embed_dim=7, num_heads=2)  # Not divisible

    with pytest.raises(ValidationError):
        TransformerEncoderBlock(embed_dim=0, num_heads=1)

    block = TransformerEncoderBlock(embed_dim=4, num_heads=2)
    with pytest.raises(ValidationError):
        block.forward([[[1.0, 2.0, 3.0]]])  # Feature dim 3 vs 4
