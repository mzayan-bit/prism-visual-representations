"""Unit tests for residual shortcuts and residual blocks."""

import pytest

from prism.core.errors import ValidationError
from prism.models.residual import (
    IdentityShortcut,
    ProjectionShortcut,
    ResidualAdd,
    ResidualBlock,
)


@pytest.mark.unit
def test_residual_add_forward_and_backward() -> None:
    """Verify elementwise addition and dual-branch gradient routing."""
    res_add = ResidualAdd()
    a = [[[[1.0, 2.0], [3.0, 4.0]]]]
    b = [[[[0.5, -1.0], [2.0, -3.0]]]]

    # Forward
    z = res_add.forward(a, b)
    assert z == [[[[1.5, 1.0], [5.0, 1.0]]]]

    # Backward: routes upstream gradient dZ identically to dA and dB
    d_out = [[[[2.0, -2.0], [1.0, 0.0]]]]
    d_a, d_b = res_add.backward(d_out)
    assert d_a == d_out
    assert d_b == d_out


@pytest.mark.unit
def test_residual_add_incompatible_shapes() -> None:
    """Verify ResidualAdd rejects tensors with mismatched shapes."""
    res_add = ResidualAdd()
    a = [[[[1.0, 2.0], [3.0, 4.0]]]]  # 1x1x2x2
    b = [[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]]  # 1x1x2x3

    with pytest.raises(ValidationError, match="Incompatible shapes"):
        res_add.forward(a, b)


@pytest.mark.unit
def test_identity_shortcut_behavior() -> None:
    """Verify IdentityShortcut passes inputs and gradients unchanged."""
    sc = IdentityShortcut()
    x = [[[[1.0, -1.0], [2.0, -2.0]]]]
    assert sc.forward(x) == x
    assert sc.backward(x) == x
    assert sc.get_parameters() == {}
    assert sc.get_gradients() == {}
    assert sc.get_state() == {}


@pytest.mark.unit
def test_projection_shortcut_forward_and_params() -> None:
    """Verify ProjectionShortcut projects channels and spatial dimensions."""
    # Project 2 channels -> 4 channels with stride 2
    sc = ProjectionShortcut(
        in_channels=2,
        out_channels=4,
        stride=2,
        normalization="batch_norm",
        seed=42,
    )
    sc.train()

    # Input: 1 sample, 2 channels, 4x4 image
    x = [
        [
            [[1.0] * 4 for _ in range(4)],
            [[2.0] * 4 for _ in range(4)],
        ]
    ]
    out = sc.forward(x)

    # Output: 1 sample, 4 channels, 2x2 image
    assert len(out) == 1
    assert len(out[0]) == 4
    assert len(out[0][0]) == 2
    assert len(out[0][0][0]) == 2

    # Verify parameter discovery
    params = sc.get_parameters()
    assert "proj_conv_weights" in params
    assert "proj_conv_bias" in params
    assert "proj_norm_gamma" in params
    assert "proj_norm_beta" in params

    # Backward pass
    d_out = [[[[0.1] * 2 for _ in range(2)] for _ in range(4)]]
    d_x = sc.backward(d_out)
    assert len(d_x) == 1
    assert len(d_x[0]) == 2
    assert len(d_x[0][0]) == 4
    assert len(d_x[0][0][0]) == 4

    grads = sc.get_gradients()
    assert "grad_proj_conv_weights" in grads


@pytest.mark.unit
def test_residual_block_identity_forward_and_backward() -> None:
    """Verify ResidualBlock with identity shortcut passes gradients."""
    block = ResidualBlock(
        in_channels=4,
        out_channels=4,
        stride=1,
        normalization="batch_norm",
        activation="relu",
        seed=42,
    )
    block.train()

    x = [[[[0.5] * 4 for _ in range(4)] for _ in range(4)]]
    out = block.forward(x)

    assert len(out) == 1
    assert len(out[0]) == 4
    assert len(out[0][0]) == 4
    assert len(out[0][0][0]) == 4
    assert block.has_projection is False

    d_out = [[[[0.2] * 4 for _ in range(4)] for _ in range(4)]]
    d_x = block.backward(d_out)

    assert len(d_x) == 1
    assert len(d_x[0]) == 4
    assert len(d_x[0][0]) == 4

    grads = block.get_gradients()
    assert "grad_conv1_weights" in grads
    assert "grad_conv2_weights" in grads
    assert "grad_norm1_gamma" in grads
    assert "grad_norm2_gamma" in grads


@pytest.mark.unit
def test_residual_block_projection_forward_and_backward() -> None:
    """Verify ResidualBlock with projection shortcut downsamples."""
    block = ResidualBlock(
        in_channels=2,
        out_channels=4,
        stride=2,
        normalization="batch_norm",
        activation="relu",
        seed=42,
    )
    block.train()

    x = [[[[1.0] * 4 for _ in range(4)] for _ in range(2)]]
    out = block.forward(x)

    assert len(out) == 1
    assert len(out[0]) == 4
    assert len(out[0][0]) == 2
    assert len(out[0][0][0]) == 2
    assert block.has_projection is True

    d_out = [[[[0.5] * 2 for _ in range(2)] for _ in range(4)]]
    d_x = block.backward(d_out)

    assert len(d_x) == 1
    assert len(d_x[0]) == 2
    assert len(d_x[0][0]) == 4

    grads = block.get_gradients()
    assert "grad_proj_conv_weights" in grads
    assert "grad_conv1_weights" in grads
    assert "grad_conv2_weights" in grads


@pytest.mark.unit
def test_residual_block_train_eval_modes() -> None:
    """Verify BatchNorm running stats update only in train mode."""
    block = ResidualBlock(
        in_channels=2,
        out_channels=2,
        stride=1,
        normalization="batch_norm",
        seed=42,
    )
    block.train()

    x = [[[[1.0, 2.0], [3.0, 4.0]], [[0.5, 1.5], [2.5, 3.5]]]]
    _ = block.forward(x)

    state_after_train = block.get_state()
    assert state_after_train["norm1_num_batches_tracked"] == 1
    assert state_after_train["norm2_num_batches_tracked"] == 1

    # Eval mode: stats freeze
    block.eval()
    _ = block.forward(x)
    state_after_eval = block.get_state()
    assert state_after_eval == state_after_train
