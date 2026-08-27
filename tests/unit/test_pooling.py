"""Unit tests for spatial pooling layers (MaxPool2D and AvgPool2D)."""

import pytest

from prism.core.errors import ValidationError
from prism.models.pooling import AvgPool2D, MaxPool2D
from prism.models.spatial import compute_pool2d_output_shape


@pytest.mark.unit
def test_compute_pool2d_output_shape() -> None:
    """Verify pooling output shape computation."""
    # 8x8 input, 2x2 pool, stride 2 -> 4x4
    assert compute_pool2d_output_shape(8, 8, kernel_size=2, stride=2) == (4, 4)
    # 7x7 input, 2x2 pool, stride 2 -> 3x3
    assert compute_pool2d_output_shape(7, 7, kernel_size=2, stride=2) == (3, 3)

    with pytest.raises(ValidationError):
        compute_pool2d_output_shape(2, 2, kernel_size=4, stride=1)


@pytest.mark.unit
def test_maxpool2d_forward_and_backward_routing() -> None:
    """Verify MaxPool2D selects exact maximum and routes gradient to argmax."""
    pool = MaxPool2D(kernel_size=2, stride=2, padding=0)

    # 1 sample, 1 channel, 4x4 image
    x = [
        [
            [
                [1.0, 3.0, 2.0, 4.0],
                [5.0, 2.0, 8.0, 1.0],
                [0.0, 7.0, 3.0, 6.0],
                [4.0, 9.0, 5.0, 2.0],
            ]
        ]
    ]

    out = pool.forward(x)
    # Expected 2x2 output:
    # top-left window: max(1, 3, 5, 2) = 5.0 (at (1, 0))
    # top-right window: max(2, 4, 8, 1) = 8.0 (at (1, 2))
    # bottom-left window: max(0, 7, 4, 9) = 9.0 (at (3, 1))
    # bottom-right window: max(3, 6, 5, 2) = 6.0 (at (2, 3))
    assert out == [[[[5.0, 8.0], [9.0, 6.0]]]]

    # Backward Pass: upstream gradient [[1.0, 2.0], [3.0, 4.0]]
    d_out = [[[[1.0, 2.0], [3.0, 4.0]]]]
    dx = pool.backward(d_out)

    expected_dx = [
        [
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 2.0, 0.0],  # 1.0 routed to (1, 0), 2.0 routed to (1, 2)
                [0.0, 0.0, 0.0, 4.0],  # 4.0 routed to (2, 3)
                [0.0, 3.0, 0.0, 0.0],  # 3.0 routed to (3, 1)
            ]
        ]
    ]
    assert dx == expected_dx


@pytest.mark.unit
def test_maxpool2d_overlapping_windows() -> None:
    """Verify overlapping MaxPool2D accumulates gradients on shared argmax."""
    pool = MaxPool2D(kernel_size=3, stride=1, padding=0)
    x = [
        [
            [
                [1.0, 2.0, 1.0, 0.0],
                [2.0, 10.0, 2.0, 1.0],  # (1, 1) is 10.0 (shared max)
                [1.0, 2.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        ]
    ]
    out = pool.forward(x)
    assert len(out[0][0]) == 2
    assert len(out[0][0][0]) == 2
    assert out == [[[[10.0, 10.0], [10.0, 10.0]]]]

    d_out = [[[[1.0, 1.0], [1.0, 1.0]]]]
    dx = pool.backward(d_out)

    # All 4 windows picked (1, 1), so accumulated gradient at (1, 1) should be 4.0
    assert dx[0][0][1][1] == pytest.approx(4.0)


@pytest.mark.unit
def test_avgpool2d_forward_and_backward() -> None:
    """Verify AvgPool2D computes window means and distributes gradients uniformly."""
    pool = AvgPool2D(kernel_size=2, stride=2, padding=0)
    x = [
        [
            [
                [2.0, 4.0],
                [6.0, 8.0],
            ]
        ]
    ]
    out = pool.forward(x)
    assert out == [[[[5.0]]]]  # (2+4+6+8)/4 = 5.0

    d_out = [[[[4.0]]]]
    dx = pool.backward(d_out)
    assert dx == [[[[1.0, 1.0], [1.0, 1.0]]]]
