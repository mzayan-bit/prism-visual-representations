"""Unit tests for attention primitives: softmax and ScaledDotProductAttention."""

import copy
import math

import pytest

from prism.models.attention import (
    ScaledDotProductAttention,
    softmax_1d,
    softmax_backward_1d,
)


@pytest.mark.unit
def test_softmax_1d_stability_and_probabilities() -> None:
    """Verify softmax_1d handles large dynamic ranges and produces probabilities."""
    # Large numbers that would overflow standard exp without max subtraction
    large_x = [1000.0, 1001.0, 1002.0]
    probs = softmax_1d(large_x)

    assert len(probs) == 3
    assert sum(probs) == pytest.approx(1.0, abs=1e-7)
    assert probs[2] > probs[1] > probs[0]
    # Check exact relative ratios: exp(0), exp(1), exp(2)
    s = 1.0 + math.e + math.e**2
    assert probs[0] == pytest.approx(1.0 / s, abs=1e-7)
    assert probs[1] == pytest.approx(math.e / s, abs=1e-7)
    assert probs[2] == pytest.approx(math.e**2 / s, abs=1e-7)


@pytest.mark.unit
def test_softmax_1d_numerical_gradient() -> None:
    """Verify softmax_backward_1d matches finite-difference numerical gradient."""
    x = [0.5, -0.2, 1.1, -0.8]
    y = softmax_1d(x)

    # Let scalar objective L = sum(y * c) for arbitrary c
    c = [1.2, -0.5, 0.8, 2.0]
    dy = c
    dx_analytical = softmax_backward_1d(y, dy)

    eps = 1e-6
    for i in range(len(x)):
        x_plus = list(x)
        x_minus = list(x)
        x_plus[i] += eps
        x_minus[i] -= eps

        y_plus = softmax_1d(x_plus)
        y_minus = softmax_1d(x_minus)

        loss_plus = sum(p * weight for p, weight in zip(y_plus, c, strict=True))
        loss_minus = sum(p * weight for p, weight in zip(y_minus, c, strict=True))

        grad_num = (loss_plus - loss_minus) / (2.0 * eps)
        assert dx_analytical[i] == pytest.approx(grad_num, rel=1e-3, abs=1e-4)


@pytest.mark.unit
def test_scaled_dot_product_attention_forward() -> None:
    """Verify ScaledDotProductAttention output dimensions and row normalization."""
    # N=2, H=2, L_q=3, L_k=3, D_h=4, D_v=4
    q = [
        [[[0.1, 0.2, 0.3, 0.4] for _ in range(3)] for _ in range(2)],
        [[[-0.1, 0.2, -0.3, 0.4] for _ in range(3)] for _ in range(2)],
    ]
    k = [
        [[[0.2, -0.1, 0.4, 0.1] for _ in range(3)] for _ in range(2)],
        [[[0.1, 0.3, -0.2, 0.5] for _ in range(3)] for _ in range(2)],
    ]
    v = [
        [[[1.0, 2.0, 3.0, 4.0] for _ in range(3)] for _ in range(2)],
        [[[0.5, 1.5, 2.5, 3.5] for _ in range(3)] for _ in range(2)],
    ]

    attn = ScaledDotProductAttention()
    out, weights = attn.forward(q, k, v)

    # Check shapes
    assert len(out) == 2 and len(out[0]) == 2 and len(out[0][0]) == 3
    assert len(weights) == 2 and len(weights[0]) == 2 and len(weights[0][0]) == 3

    # Check row normalization
    for n in range(2):
        for h in range(2):
            for i in range(3):
                row_sum = sum(weights[n][h][i])
                assert row_sum == pytest.approx(1.0, abs=1e-6)


@pytest.mark.unit
def test_scaled_dot_product_attention_numerical_gradients() -> None:
    """Verify ScaledDotProductAttention analytical dQ, dK, dV match numerical grads."""
    # N=1, H=1, L_q=2, L_k=2, D_h=2, D_v=2
    q = [[[[0.6, -0.4], [0.2, 0.8]]]]
    k = [[[[0.3, 0.5], [-0.7, 0.1]]]]
    v = [[[[1.2, -0.3], [0.4, 0.9]]]]

    attn = ScaledDotProductAttention()
    _ = attn.forward(q, k, v)

    # Objective: L = sum(out * d_out)
    d_out = [[[[1.0, 0.5], [-0.5, 1.0]]]]
    dq, dk, dv = attn.backward(d_out)

    eps = 1e-6

    # 1. Check dQ
    for i in range(2):
        for d in range(2):
            q_plus = copy.deepcopy(q)
            q_minus = copy.deepcopy(q)
            q_plus[0][0][i][d] += eps
            q_minus[0][0][i][d] -= eps

            out_plus, _ = attn.forward(q_plus, k, v)
            out_minus, _ = attn.forward(q_minus, k, v)

            loss_plus = sum(
                out_plus[0][0][r][c] * d_out[0][0][r][c]
                for r in range(2)
                for c in range(2)
            )
            loss_minus = sum(
                out_minus[0][0][r][c] * d_out[0][0][r][c]
                for r in range(2)
                for c in range(2)
            )
            grad_num = (loss_plus - loss_minus) / (2.0 * eps)
            assert dq[0][0][i][d] == pytest.approx(grad_num, rel=1e-3, abs=1e-4)

    # 2. Check dK
    for j in range(2):
        for d in range(2):
            k_plus = copy.deepcopy(k)
            k_minus = copy.deepcopy(k)
            k_plus[0][0][j][d] += eps
            k_minus[0][0][j][d] -= eps

            out_plus, _ = attn.forward(q, k_plus, v)
            out_minus, _ = attn.forward(q, k_minus, v)

            loss_plus = sum(
                out_plus[0][0][r][c] * d_out[0][0][r][c]
                for r in range(2)
                for c in range(2)
            )
            loss_minus = sum(
                out_minus[0][0][r][c] * d_out[0][0][r][c]
                for r in range(2)
                for c in range(2)
            )
            grad_num = (loss_plus - loss_minus) / (2.0 * eps)
            assert dk[0][0][j][d] == pytest.approx(grad_num, rel=1e-3, abs=1e-4)

    # 3. Check dV
    for j in range(2):
        for d in range(2):
            v_plus = copy.deepcopy(v)
            v_minus = copy.deepcopy(v)
            v_plus[0][0][j][d] += eps
            v_minus[0][0][j][d] -= eps

            out_plus, _ = attn.forward(q, k, v_plus)
            out_minus, _ = attn.forward(q, k, v_minus)

            loss_plus = sum(
                out_plus[0][0][r][c] * d_out[0][0][r][c]
                for r in range(2)
                for c in range(2)
            )
            loss_minus = sum(
                out_minus[0][0][r][c] * d_out[0][0][r][c]
                for r in range(2)
                for c in range(2)
            )
            grad_num = (loss_plus - loss_minus) / (2.0 * eps)
            assert dv[0][0][j][d] == pytest.approx(grad_num, rel=1e-3, abs=1e-4)
