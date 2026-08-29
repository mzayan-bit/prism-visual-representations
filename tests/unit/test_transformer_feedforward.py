"""Unit tests for TransformerFeedForward network and analytical backpropagation."""

import random

import pytest

from prism.core.errors import ValidationError
from prism.models.transformer import TransformerFeedForward


def test_feedforward_forward_shape() -> None:
    """Test that feedforward network maps [N, T, D] to [N, T, D]."""
    ffn = TransformerFeedForward(
        in_features=8,
        hidden_dim=16,
        bias=True,
        activation="gelu",
        seed=42,
    )
    x = [[[0.5 for _ in range(8)] for _ in range(4)] for _ in range(2)]
    out = ffn.forward(x)

    assert len(out) == 2
    assert len(out[0]) == 4
    assert len(out[0][0]) == 8


def test_feedforward_token_independence() -> None:
    """Test that tokens are processed independently without cross-token interaction."""
    ffn = TransformerFeedForward(in_features=4, hidden_dim=8, seed=123)
    rng = random.Random(42)

    tok1 = [rng.gauss(0.0, 1.0) for _ in range(4)]
    tok2 = [rng.gauss(0.0, 1.0) for _ in range(4)]
    tok3 = [rng.gauss(0.0, 1.0) for _ in range(4)]

    # Batch with 2 tokens: [tok1, tok2]
    out_a = ffn.forward([[tok1, tok2]])
    # Batch with 2 tokens: [tok1, tok3]
    out_b = ffn.forward([[tok1, tok3]])

    # Token 1 output should be bitwise identical regardless of token 2 vs token 3
    for d in range(4):
        assert abs(out_a[0][0][d] - out_b[0][0][d]) < 1e-12


def test_feedforward_gradient_numerical_check() -> None:
    """Validate analytical gradients w.r.t numerical finite differences."""
    d_in = 3
    d_hid = 4
    ffn = TransformerFeedForward(
        in_features=d_in,
        hidden_dim=d_hid,
        bias=True,
        activation="gelu",
        seed=101,
    )

    rng = random.Random(77)
    x = [
        [[rng.gauss(0.0, 1.0) for _ in range(d_in)] for _ in range(2)] for _ in range(2)
    ]
    d_out = [
        [[rng.gauss(0.0, 1.0) for _ in range(d_in)] for _ in range(2)] for _ in range(2)
    ]

    ffn.zero_grad()
    _ = ffn.forward(x)
    dx_analytic = ffn.backward(d_out)

    eps = 1e-5

    # 1. Check Input Gradients dX
    for n in range(2):
        for t in range(2):
            for j in range(d_in):
                x_pos = [[list(tok) for tok in sample] for sample in x]
                x_neg = [[list(tok) for tok in sample] for sample in x]
                x_pos[n][t][j] += eps
                x_neg[n][t][j] -= eps

                out_pos = ffn.forward(x_pos)
                out_neg = ffn.forward(x_neg)

                obj_pos = sum(
                    out_pos[b][s][k] * d_out[b][s][k]
                    for b in range(2)
                    for s in range(2)
                    for k in range(d_in)
                )
                obj_neg = sum(
                    out_neg[b][s][k] * d_out[b][s][k]
                    for b in range(2)
                    for s in range(2)
                    for k in range(d_in)
                )
                grad_num = (obj_pos - obj_neg) / (2.0 * eps)
                assert abs(dx_analytic[n][t][j] - grad_num) < 1e-4

    # 2. Check W_1 gradients
    for i in range(d_in):
        for h in range(d_hid):
            orig = ffn.w_1[i][h]
            ffn.w_1[i][h] = orig + eps
            out_pos = ffn.forward(x)
            ffn.w_1[i][h] = orig - eps
            out_neg = ffn.forward(x)
            ffn.w_1[i][h] = orig

            obj_pos = sum(
                out_pos[b][s][k] * d_out[b][s][k]
                for b in range(2)
                for s in range(2)
                for k in range(d_in)
            )
            obj_neg = sum(
                out_neg[b][s][k] * d_out[b][s][k]
                for b in range(2)
                for s in range(2)
                for k in range(d_in)
            )
            grad_num = (obj_pos - obj_neg) / (2.0 * eps)
            assert abs(ffn.grad_w_1[i][h] - grad_num) < 1e-4

    # 3. Check W_2 gradients
    for h in range(d_hid):
        for j in range(d_in):
            orig = ffn.w_2[h][j]
            ffn.w_2[h][j] = orig + eps
            out_pos = ffn.forward(x)
            ffn.w_2[h][j] = orig - eps
            out_neg = ffn.forward(x)
            ffn.w_2[h][j] = orig

            obj_pos = sum(
                out_pos[b][s][k] * d_out[b][s][k]
                for b in range(2)
                for s in range(2)
                for k in range(d_in)
            )
            obj_neg = sum(
                out_neg[b][s][k] * d_out[b][s][k]
                for b in range(2)
                for s in range(2)
                for k in range(d_in)
            )
            grad_num = (obj_pos - obj_neg) / (2.0 * eps)
            assert abs(ffn.grad_w_2[h][j] - grad_num) < 1e-4


def test_feedforward_parameters_and_state() -> None:
    """Test get/set parameters and zero_grad."""
    ffn = TransformerFeedForward(in_features=4, hidden_dim=6, bias=True, seed=1)
    params = ffn.get_parameters()
    assert "w_1" in params
    assert "b_1" in params
    assert "w_2" in params
    assert "b_2" in params

    ffn.set_parameters(params)
    assert ffn.get_state() == {}


def test_feedforward_validation_errors() -> None:
    """Test invalid configurations and dimensions raise ValidationError."""
    with pytest.raises(ValidationError):
        TransformerFeedForward(in_features=0, hidden_dim=4)

    with pytest.raises(ValidationError):
        TransformerFeedForward(in_features=4, hidden_dim=0)

    ffn = TransformerFeedForward(in_features=4, hidden_dim=8)
    with pytest.raises(ValidationError):
        ffn.forward([[[1.0, 2.0]]])  # in_features mismatch (2 vs 4)
