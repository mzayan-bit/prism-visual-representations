"""Unit tests for MultiHeadSelfAttention forward, backward, and parameter gradients."""

import copy

import pytest

from prism.core.errors import ValidationError
from prism.models.attention import MultiHeadSelfAttention


@pytest.mark.unit
def test_multi_head_self_attention_shapes_and_weights() -> None:
    """Verify MultiHeadSelfAttention forward output shape and attention weights."""
    # N=2, L=3, D_embed=8, num_heads=2 -> D_head=4
    mhsa = MultiHeadSelfAttention(embed_dim=8, num_heads=2, bias=True, seed=42)
    x = [
        [[0.1 * (i + d) for d in range(8)] for i in range(3)],
        [[-0.1 * (i + d) for d in range(8)] for i in range(3)],
    ]
    out = mhsa.forward(x)

    assert len(out) == 2 and len(out[0]) == 3 and len(out[0][0]) == 8
    assert mhsa.last_attention_weights is not None
    # Shape of attention weights: [N, H, L, L] = [2, 2, 3, 3]
    weights = mhsa.last_attention_weights
    assert (
        len(weights) == 2
        and len(weights[0]) == 2
        and len(weights[0][0]) == 3
        and len(weights[0][0][0]) == 3
    )


@pytest.mark.unit
def test_multi_head_self_attention_validation_rejections() -> None:
    """Verify MultiHeadSelfAttention rejects non-divisible embeddings."""
    # embed_dim=7 not divisible by num_heads=2
    with pytest.raises(ValidationError, match="must be divisible by num_heads"):
        MultiHeadSelfAttention(embed_dim=7, num_heads=2)

    with pytest.raises(ValidationError, match="num_heads must be positive"):
        MultiHeadSelfAttention(embed_dim=8, num_heads=0)


@pytest.mark.unit
def test_multi_head_self_attention_numerical_input_gradient() -> None:
    """Verify MHSA analytical input gradient dX matches numerical derivative."""
    # Small dimensions for fast and accurate numerical verification
    # N=1, L=2, D_embed=4, num_heads=2 -> D_head=2
    mhsa = MultiHeadSelfAttention(embed_dim=4, num_heads=2, bias=True, seed=42)
    x = [[[0.5, -0.3, 0.8, -0.1], [0.2, 0.4, -0.6, 0.7]]]

    _ = mhsa.forward(x)
    # Scalar objective L = sum(out * d_out)
    d_out = [[[1.0, 0.5, -0.5, 0.2], [-0.3, 0.8, 0.4, 1.0]]]
    dx_analytical = mhsa.backward(d_out)

    eps = 1e-6
    for seq_idx in range(2):
        for d in range(4):
            x_plus = copy.deepcopy(x)
            x_minus = copy.deepcopy(x)
            x_plus[0][seq_idx][d] += eps
            x_minus[0][seq_idx][d] -= eps

            out_plus = mhsa.forward(x_plus)
            out_minus = mhsa.forward(x_minus)

            loss_plus = sum(
                out_plus[0][i][k] * d_out[0][i][k] for i in range(2) for k in range(4)
            )
            loss_minus = sum(
                out_minus[0][i][k] * d_out[0][i][k] for i in range(2) for k in range(4)
            )

            grad_num = (loss_plus - loss_minus) / (2.0 * eps)
            assert dx_analytical[0][seq_idx][d] == pytest.approx(
                grad_num, rel=1e-3, abs=1e-4
            )


@pytest.mark.unit
def test_multi_head_self_attention_numerical_parameter_gradients() -> None:
    """Verify MHSA parameter gradients match numerical derivatives."""
    mhsa = MultiHeadSelfAttention(embed_dim=4, num_heads=2, bias=True, seed=42)
    x = [[[0.2, -0.1, 0.4, 0.3], [-0.3, 0.5, 0.1, -0.2]]]

    _ = mhsa.forward(x)
    d_out = [[[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]]
    _ = mhsa.backward(d_out)
    grads_analytical = mhsa.get_gradients()

    eps = 1e-6
    # 1. Check W_O
    for r in range(4):
        for c in range(4):
            mhsa.w_o[r][c] += eps
            out_plus = mhsa.forward(x)
            loss_plus = sum(
                out_plus[0][i][k] * d_out[0][i][k] for i in range(2) for k in range(4)
            )
            mhsa.w_o[r][c] -= 2.0 * eps
            out_minus = mhsa.forward(x)
            loss_minus = sum(
                out_minus[0][i][k] * d_out[0][i][k] for i in range(2) for k in range(4)
            )
            mhsa.w_o[r][c] += eps  # restore

            grad_num = (loss_plus - loss_minus) / (2.0 * eps)
            assert grads_analytical["w_o"][r][c] == pytest.approx(
                grad_num, rel=1e-3, abs=1e-4
            )

    # 2. Check W_Q
    for r in range(4):
        for c in range(4):
            mhsa.w_q[r][c] += eps
            out_plus = mhsa.forward(x)
            loss_plus = sum(
                out_plus[0][i][k] * d_out[0][i][k] for i in range(2) for k in range(4)
            )
            mhsa.w_q[r][c] -= 2.0 * eps
            out_minus = mhsa.forward(x)
            loss_minus = sum(
                out_minus[0][i][k] * d_out[0][i][k] for i in range(2) for k in range(4)
            )
            mhsa.w_q[r][c] += eps  # restore

            grad_num = (loss_plus - loss_minus) / (2.0 * eps)
            assert grads_analytical["w_q"][r][c] == pytest.approx(
                grad_num, rel=1e-3, abs=1e-4
            )
