"""Finite-difference numerical gradient checks for ResidualBlock components."""

import pytest

from prism.models.residual import ProjectionShortcut, ResidualBlock


@pytest.mark.unit
def test_identity_residual_block_numerical_gradient_check() -> None:
    """Verify finite-difference gradients for Identity ResidualBlock."""
    block = ResidualBlock(
        in_channels=2,
        out_channels=2,
        stride=1,
        normalization="none",
        activation="relu",
        seed=42,
    )
    block.train()

    # Small 1x2x3x3 input
    x = [
        [
            [[0.5, -0.2, 0.3], [0.1, 0.4, -0.5], [-0.3, 0.2, 0.1]],
            [[-0.4, 0.3, 0.1], [0.2, -0.1, 0.5], [0.4, -0.2, 0.3]],
        ]
    ]
    d_out = [
        [
            [[0.2, -0.1, 0.3], [0.4, -0.2, 0.1], [-0.1, 0.3, -0.2]],
            [[-0.3, 0.2, 0.1], [0.1, -0.4, 0.2], [0.2, -0.1, 0.3]],
        ]
    ]

    _ = block.forward(x)
    dx_ana = block.backward(d_out)
    dw1_ana = list(block.conv1.grad_weights)
    dw2_ana = list(block.conv2.grad_weights)

    def compute_loss(
        test_block: ResidualBlock, test_x: list[list[list[list[float]]]]
    ) -> float:
        y = test_block.forward(test_x)
        loss = 0.0
        for n in range(len(y)):
            for c in range(len(y[0])):
                for h in range(len(y[0][0])):
                    for w in range(len(y[0][0][0])):
                        loss += y[n][c][h][w] * d_out[n][c][h][w]
        return loss

    eps = 1e-5

    # 1. Check Input Gradient dX (routes through both main and shortcut paths)
    for c in range(2):
        for h in range(3):
            for w in range(3):
                orig = x[0][c][h][w]
                x[0][c][h][w] = orig + eps
                l_plus = compute_loss(block, x)
                x[0][c][h][w] = orig - eps
                l_minus = compute_loss(block, x)
                x[0][c][h][w] = orig

                dx_num = (l_plus - l_minus) / (2.0 * eps)
                assert dx_ana[0][c][h][w] == pytest.approx(dx_num, rel=1e-3, abs=1e-4)

    # 2. Check Conv1 Weights Gradient
    for out_c in range(2):
        for in_c in range(2):
            for kh in range(3):
                for kw in range(3):
                    orig_w = block.conv1.weights[out_c][in_c][kh][kw]
                    block.conv1.weights[out_c][in_c][kh][kw] = orig_w + eps
                    l_plus = compute_loss(block, x)
                    block.conv1.weights[out_c][in_c][kh][kw] = orig_w - eps
                    l_minus = compute_loss(block, x)
                    block.conv1.weights[out_c][in_c][kh][kw] = orig_w

                    dw_num = (l_plus - l_minus) / (2.0 * eps)
                    assert dw1_ana[out_c][in_c][kh][kw] == pytest.approx(
                        dw_num, rel=1e-3, abs=1e-4
                    )

    # 3. Check Conv2 Weights Gradient
    for out_c in range(2):
        for in_c in range(2):
            for kh in range(3):
                for kw in range(3):
                    orig_w = block.conv2.weights[out_c][in_c][kh][kw]
                    block.conv2.weights[out_c][in_c][kh][kw] = orig_w + eps
                    l_plus = compute_loss(block, x)
                    block.conv2.weights[out_c][in_c][kh][kw] = orig_w - eps
                    l_minus = compute_loss(block, x)
                    block.conv2.weights[out_c][in_c][kh][kw] = orig_w

                    dw_num = (l_plus - l_minus) / (2.0 * eps)
                    assert dw2_ana[out_c][in_c][kh][kw] == pytest.approx(
                        dw_num, rel=1e-3, abs=1e-4
                    )


@pytest.mark.unit
def test_projection_residual_block_numerical_gradient_check() -> None:
    """Verify finite-difference gradients for Projection ResidualBlock."""
    block = ResidualBlock(
        in_channels=1,
        out_channels=2,
        stride=2,
        normalization="none",
        activation="relu",
        seed=42,
    )
    block.train()

    # Small 1x1x4x4 input
    x = [
        [
            [
                [0.5, -0.2, 0.3, 0.1],
                [0.1, 0.4, -0.5, 0.2],
                [-0.3, 0.2, 0.1, -0.4],
                [0.4, -0.1, 0.3, 0.2],
            ]
        ]
    ]
    # Output is 1x2x2x2
    d_out = [
        [
            [[0.3, -0.2], [0.1, 0.4]],
            [[-0.1, 0.3], [0.2, -0.4]],
        ]
    ]

    _ = block.forward(x)
    dx_ana = block.backward(d_out)
    assert isinstance(block.shortcut, ProjectionShortcut)
    d_proj_conv_ana = list(block.shortcut.conv.grad_weights)

    def compute_loss(
        test_block: ResidualBlock, test_x: list[list[list[list[float]]]]
    ) -> float:
        y = test_block.forward(test_x)
        loss = 0.0
        for n in range(len(y)):
            for c in range(len(y[0])):
                for h in range(len(y[0][0])):
                    for w in range(len(y[0][0][0])):
                        loss += y[n][c][h][w] * d_out[n][c][h][w]
        return loss

    eps = 1e-5

    # 1. Check Input Gradient dX
    for h in range(4):
        for w in range(4):
            orig = x[0][0][h][w]
            x[0][0][h][w] = orig + eps
            l_plus = compute_loss(block, x)
            x[0][0][h][w] = orig - eps
            l_minus = compute_loss(block, x)
            x[0][0][h][w] = orig

            dx_num = (l_plus - l_minus) / (2.0 * eps)
            assert dx_ana[0][0][h][w] == pytest.approx(dx_num, rel=1e-3, abs=1e-4)

    # 2. Check Projection Conv 1x1 Weights Gradient
    for out_c in range(2):
        for in_c in range(1):
            orig_w = block.shortcut.conv.weights[out_c][in_c][0][0]
            block.shortcut.conv.weights[out_c][in_c][0][0] = orig_w + eps
            l_plus = compute_loss(block, x)
            block.shortcut.conv.weights[out_c][in_c][0][0] = orig_w - eps
            l_minus = compute_loss(block, x)
            block.shortcut.conv.weights[out_c][in_c][0][0] = orig_w

            dw_num = (l_plus - l_minus) / (2.0 * eps)
            assert d_proj_conv_ana[out_c][in_c][0][0] == pytest.approx(
                dw_num, rel=1e-3, abs=1e-4
            )
