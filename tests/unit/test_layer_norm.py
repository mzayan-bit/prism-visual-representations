"""Unit tests for explicit LayerNorm implementation and analytical backward."""

import random

import pytest

from prism.core.errors import ValidationError
from prism.models.normalization import LayerNorm, get_normalization


def test_layer_norm_2d_forward_normalization() -> None:
    """Test that 2D [N, D] input produces zero-mean unit-variance vectors."""
    ln = LayerNorm(normalized_shape=4, eps=1e-5, affine=False)
    x = [
        [1.0, 2.0, 3.0, 4.0],
        [10.0, 20.0, 30.0, 40.0],
    ]
    out = ln.forward(x)

    assert len(out) == 2
    assert len(out[0]) == 4

    for row in out:
        mean = sum(row) / len(row)
        var = sum((v - mean) ** 2 for v in row) / len(row)
        assert abs(mean) < 1e-6
        # Variance of normalized values should be close to 1.0
        assert abs(var - 1.0) < 1e-3


def test_layer_norm_3d_forward_normalization() -> None:
    """Test that 3D [N, T, D] input normalizes across token vectors independently."""
    ln = LayerNorm(normalized_shape=4, eps=1e-5, affine=False)
    x = [
        [
            [1.0, 2.0, 3.0, 4.0],
            [10.0, 20.0, 30.0, 40.0],
        ],
        [
            [100.0, 200.0, 300.0, 400.0],
            [-1.0, -2.0, -3.0, -4.0],
        ],
    ]
    out = ln.forward(x)

    assert len(out) == 2
    assert len(out[0]) == 2
    assert len(out[0][0]) == 4

    for sample in out:
        for token in sample:
            mean = sum(token) / len(token)
            var = sum((v - mean) ** 2 for v in token) / len(token)
            assert abs(mean) < 1e-6
            assert abs(var - 1.0) < 1e-3


def test_layer_norm_affine_transformation() -> None:
    """Test that affine parameters gamma and beta scale and shift normalized output."""
    ln = LayerNorm(normalized_shape=3, eps=1e-5, affine=True)
    ln.gamma = [2.0, 3.0, 4.0]
    ln.beta = [0.5, -0.5, 1.0]

    x = [[1.0, 2.0, 3.0]]
    out = ln.forward(x)

    # Without affine, normalized [1, 2, 3] with mean 2, var 2/3:
    # diffs: [-1, 0, 1], std = sqrt(2/3) ~ 0.81649658
    # x_hat ~ [-1.22474487, 0.0, 1.22474487]
    # out = gamma * x_hat + beta
    # out[0] ~ 2.0 * (-1.22474487) + 0.5 = -1.9494897
    # out[1] ~ 3.0 * (0.0) - 0.5 = -0.5
    # out[2] ~ 4.0 * (1.22474487) + 1.0 = 5.8989795
    assert abs(out[0][0] - (-1.9494897)) < 1e-4
    assert abs(out[0][1] - (-0.5)) < 1e-4
    assert abs(out[0][2] - 5.8989795) < 1e-4


def test_layer_norm_train_eval_equivalence() -> None:
    """Test that LayerNorm behaves identically in training and evaluation modes."""
    ln = LayerNorm(normalized_shape=4, eps=1e-5, affine=True)
    x = [[2.0, 4.0, 6.0, 8.0]]

    ln.train()
    out_train = ln.forward(x)

    ln.eval()
    out_eval = ln.forward(x)

    assert out_train == out_eval
    assert ln.get_state() == {}


def test_layer_norm_gradient_numerical_check_2d() -> None:
    """Validate analytical input gradient dX w.r.t numerical gradients."""
    d = 4
    ln = LayerNorm(normalized_shape=d, eps=1e-5, affine=True)
    ln.gamma = [1.2, 0.8, 1.5, 0.9]
    ln.beta = [0.1, -0.2, 0.3, -0.1]

    rng = random.Random(42)
    x = [[rng.gauss(0.0, 1.0) for _ in range(d)] for _ in range(2)]
    d_out = [[rng.gauss(0.0, 1.0) for _ in range(d)] for _ in range(2)]

    ln.zero_grad()
    _ = ln.forward(x)
    dx_analytic = ln.backward(d_out)

    # Numerical gradient check for x
    eps = 1e-5
    for n in range(2):
        for j in range(d):
            x_pos = [list(r) for r in x]
            x_neg = [list(r) for r in x]
            x_pos[n][j] += eps
            x_neg[n][j] -= eps

            out_pos = ln.forward(x_pos)
            out_neg = ln.forward(x_neg)

            # Scalar objective = sum(out * d_out)
            obj_pos = sum(
                out_pos[b][k] * d_out[b][k] for b in range(2) for k in range(d)
            )
            obj_neg = sum(
                out_neg[b][k] * d_out[b][k] for b in range(2) for k in range(d)
            )
            grad_num = (obj_pos - obj_neg) / (2.0 * eps)

            assert abs(dx_analytic[n][j] - grad_num) < 1e-4


def test_layer_norm_gradient_numerical_check_3d() -> None:
    """Validate analytical input and parameter gradients on 3D token sequences."""
    d = 3
    ln = LayerNorm(normalized_shape=d, eps=1e-5, affine=True)
    ln.gamma = [1.1, 0.9, 1.3]
    ln.beta = [0.2, -0.1, 0.4]

    rng = random.Random(99)
    x = [[[rng.gauss(0.0, 1.0) for _ in range(d)] for _ in range(2)] for _ in range(2)]
    d_out = [
        [[rng.gauss(0.0, 1.0) for _ in range(d)] for _ in range(2)] for _ in range(2)
    ]

    ln.zero_grad()
    _ = ln.forward(x)
    dx_analytic = ln.backward(d_out)

    eps = 1e-5
    for n in range(2):
        for t in range(2):
            for j in range(d):
                x_pos = [[list(tok) for tok in sample] for sample in x]
                x_neg = [[list(tok) for tok in sample] for sample in x]
                x_pos[n][t][j] += eps
                x_neg[n][t][j] -= eps

                out_pos = ln.forward(x_pos)
                out_neg = ln.forward(x_neg)

                obj_pos = sum(
                    out_pos[b][s][k] * d_out[b][s][k]
                    for b in range(2)
                    for s in range(2)
                    for k in range(d)
                )
                obj_neg = sum(
                    out_neg[b][s][k] * d_out[b][s][k]
                    for b in range(2)
                    for s in range(2)
                    for k in range(d)
                )
                grad_num = (obj_pos - obj_neg) / (2.0 * eps)
                assert abs(dx_analytic[n][t][j] - grad_num) < 1e-4

    # Check gamma gradients
    for j in range(d):
        ln_pos = LayerNorm(normalized_shape=d, eps=1e-5, affine=True)
        ln_pos.gamma = list(ln.gamma)
        ln_pos.beta = list(ln.beta)
        ln_pos.gamma[j] += eps

        ln_neg = LayerNorm(normalized_shape=d, eps=1e-5, affine=True)
        ln_neg.gamma = list(ln.gamma)
        ln_neg.beta = list(ln.beta)
        ln_neg.gamma[j] -= eps

        out_pos = ln_pos.forward(x)
        out_neg = ln_neg.forward(x)

        obj_pos = sum(
            out_pos[b][s][k] * d_out[b][s][k]
            for b in range(2)
            for s in range(2)
            for k in range(d)
        )
        obj_neg = sum(
            out_neg[b][s][k] * d_out[b][s][k]
            for b in range(2)
            for s in range(2)
            for k in range(d)
        )
        grad_num = (obj_pos - obj_neg) / (2.0 * eps)
        assert abs(ln.grad_gamma[j] - grad_num) < 1e-4


def test_layer_norm_factory_registration() -> None:
    """Test get_normalization factory creates LayerNorm."""
    norm = get_normalization("layer_norm", num_features=64, eps=1e-6)
    assert isinstance(norm, LayerNorm)
    assert norm.num_features == 64
    assert norm.eps == 1e-6


def test_layer_norm_validation_rejections() -> None:
    """Test invalid configurations and inputs raise ValidationError."""
    with pytest.raises(ValidationError):
        LayerNorm(normalized_shape=0)

    with pytest.raises(ValidationError):
        LayerNorm(normalized_shape=4, eps=-1.0)

    ln = LayerNorm(normalized_shape=4)
    with pytest.raises(ValidationError):
        ln.forward([])

    with pytest.raises(ValidationError):
        ln.forward([1.0, 2.0])  # 1D not allowed

    with pytest.raises(ValidationError):
        ln.forward([[1.0, float("nan"), 3.0, 4.0]])
