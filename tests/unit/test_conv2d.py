"""Unit tests for Conv2D layer forward pass, backward pass, and numerical gradients."""

import pytest

from prism.core.errors import ValidationError
from prism.models.convolution import Conv2D
from prism.models.spatial import (
    compute_conv2d_output_shape,
    compute_receptive_field,
)


@pytest.mark.unit
def test_compute_conv2d_output_shape_formula() -> None:
    """Verify output shape matches floor((in + 2p - k)/s) + 1."""
    # 32x32 input, 3x3 kernel, stride 1, pad 1 -> 32x32
    assert compute_conv2d_output_shape(32, 32, kernel_size=3, stride=1, padding=1) == (
        32,
        32,
    )

    # 32x32 input, 3x3 kernel, stride 2, pad 1 -> 16x16
    assert compute_conv2d_output_shape(32, 32, kernel_size=3, stride=2, padding=1) == (
        16,
        16,
    )

    # 32x32 input, 5x5 kernel, stride 1, pad 0 -> 28x28
    assert compute_conv2d_output_shape(32, 32, kernel_size=5, stride=1, padding=0) == (
        28,
        28,
    )

    # Invalid cases
    with pytest.raises(ValidationError, match=r"Kernel height .* exceeds"):
        compute_conv2d_output_shape(4, 4, kernel_size=6, stride=1, padding=0)


@pytest.mark.unit
def test_compute_receptive_field_tracking() -> None:
    """Verify receptive field expansion and jump across convolutional stages."""
    # Stage 1: Conv 3x3, stride 1 -> RF = 3, Jump = 1
    # Stage 2: MaxPool 2x2, stride 2 -> RF = 3 + (2-1)*1 = 4, Jump = 2
    # Stage 3: Conv 3x3, stride 1 -> RF = 4 + (3-1)*2 = 8, Jump = 2
    stages = [(3, 1), (2, 2), (3, 1)]
    rf, jump = compute_receptive_field(stages)
    assert rf == 8
    assert jump == 2


@pytest.mark.unit
def test_conv2d_forward_single_channel_known_output() -> None:
    """Verify single-channel 2D convolution against hand-calculated values."""
    conv = Conv2D(
        in_channels=1,
        out_channels=1,
        kernel_size=2,
        stride=1,
        padding=0,
        bias=True,
        seed=42,
    )
    conv.weights = [[[[1.0, 2.0], [3.0, 4.0]]]]
    conv.bias_weights = [0.5]

    # Input image 3x3
    x = [[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]]

    # Output should be 2x2:
    # top-left: 0.5 + 1*1 + 2*2 + 4*3 + 5*4 = 37.5
    # top-right: 0.5 + 2*1 + 3*2 + 5*3 + 6*4 = 47.5
    # bottom-left: 0.5 + 4*1 + 5*2 + 7*3 + 8*4 = 67.5
    # bottom-right: 0.5 + 5*1 + 6*2 + 8*3 + 9*4 = 77.5
    out = conv.forward(x)
    assert len(out) == 1
    assert len(out[0]) == 1
    assert len(out[0][0]) == 2
    assert len(out[0][0][0]) == 2

    assert out[0][0][0][0] == pytest.approx(37.5)
    assert out[0][0][0][1] == pytest.approx(47.5)
    assert out[0][0][1][0] == pytest.approx(67.5)
    assert out[0][0][1][1] == pytest.approx(77.5)


@pytest.mark.unit
def test_conv2d_forward_multichannel_multichannel_padding() -> None:
    """Verify multi-channel and multi-filter convolution with padding and stride."""
    conv = Conv2D(
        in_channels=3,
        out_channels=4,
        kernel_size=3,
        stride=2,
        padding=1,
        bias=True,
        seed=42,
    )
    # Batch of 2 samples, 3 channels, 8x8 image
    x = [
        [[[1.0] * 8 for _ in range(8)] for _ in range(3)],
        [[[0.5] * 8 for _ in range(8)] for _ in range(3)],
    ]
    out = conv.forward(x)
    # Output shape: [2, 4, 4, 4]
    assert len(out) == 2
    assert len(out[0]) == 4
    assert len(out[0][0]) == 4
    assert len(out[0][0][0]) == 4


@pytest.mark.unit
def test_conv2d_backward_gradients_and_zero_grad() -> None:
    """Verify analytic backward computes gradients and zero_grad resets."""
    conv = Conv2D(
        in_channels=2,
        out_channels=3,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True,
        seed=42,
    )
    x = [[[[0.5] * 4 for _ in range(4)] for _ in range(2)]]
    _ = conv.forward(x)

    d_out = [[[[1.0] * 4 for _ in range(4)] for _ in range(3)]]
    dx = conv.backward(d_out)

    # dx shape matches x shape: [1, 2, 4, 4]
    assert len(dx) == 1
    assert len(dx[0]) == 2
    assert len(dx[0][0]) == 4
    assert len(dx[0][0][0]) == 4

    grads = conv.get_gradients()
    assert "grad_weights" in grads
    assert "grad_bias" in grads
    assert len(grads["grad_weights"]) == 3
    assert len(grads["grad_bias"]) == 3

    # Reset gradients
    conv.zero_grad()
    cleared = conv.get_gradients()
    assert all(b == 0.0 for b in cleared["grad_bias"])


@pytest.mark.unit
def test_conv2d_numerical_gradient_check() -> None:
    """Perform numerical gradient checking for Conv2D weights, bias, and input."""
    conv = Conv2D(
        in_channels=2,
        out_channels=2,
        kernel_size=2,
        stride=1,
        padding=0,
        bias=True,
        seed=123,
    )

    x = [
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[0.5, -0.5], [1.0, -1.0]],
        ]
    ]

    # Analytic forward and backward
    _ = conv.forward(x)
    d_out = [[[[0.7]], [[-0.5]]]]  # [1, 2, 1, 1]
    dx_analytic = conv.backward(d_out)
    dw_analytic = conv.grad_weights
    db_analytic = conv.grad_bias_weights

    # Helper scalar loss: L = sum(out * d_out)
    def compute_loss(test_conv: Conv2D, test_x: list[list[list[list[float]]]]) -> float:
        y = test_conv.forward(test_x)
        loss = 0.0
        for n in range(len(y)):
            for f in range(len(y[0])):
                for i in range(len(y[0][0])):
                    for j in range(len(y[0][0][0])):
                        loss += y[n][f][i][j] * d_out[n][f][i][j]
        return loss

    eps = 1e-5

    # 1. Numerical check for weights
    for f in range(2):
        for c in range(2):
            for kh in range(2):
                for kw in range(2):
                    orig = conv.weights[f][c][kh][kw]

                    conv.weights[f][c][kh][kw] = orig + eps
                    l_plus = compute_loss(conv, x)

                    conv.weights[f][c][kh][kw] = orig - eps
                    l_minus = compute_loss(conv, x)

                    conv.weights[f][c][kh][kw] = orig

                    dw_num = (l_plus - l_minus) / (2.0 * eps)
                    dw_ana = dw_analytic[f][c][kh][kw]
                    assert dw_ana == pytest.approx(dw_num, rel=1e-3, abs=1e-4)

    # 2. Numerical check for bias
    for f in range(2):
        orig_b = conv.bias_weights[f]

        conv.bias_weights[f] = orig_b + eps
        l_plus = compute_loss(conv, x)

        conv.bias_weights[f] = orig_b - eps
        l_minus = compute_loss(conv, x)

        conv.bias_weights[f] = orig_b

        db_num = (l_plus - l_minus) / (2.0 * eps)
        db_ana = db_analytic[f]
        assert db_ana == pytest.approx(db_num, rel=1e-3, abs=1e-4)

    # 3. Numerical check for input x
    for c in range(2):
        for h in range(2):
            for w in range(2):
                orig_x = x[0][c][h][w]

                x[0][c][h][w] = orig_x + eps
                l_plus = compute_loss(conv, x)

                x[0][c][h][w] = orig_x - eps
                l_minus = compute_loss(conv, x)

                x[0][c][h][w] = orig_x

                dx_num = (l_plus - l_minus) / (2.0 * eps)
                dx_ana = dx_analytic[0][c][h][w]
                assert dx_ana == pytest.approx(dx_num, rel=1e-3, abs=1e-4)
