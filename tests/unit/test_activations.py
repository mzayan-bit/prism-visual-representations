"""Unit tests for ReLU and GELU activation functions and analytical derivatives."""

import pytest

from prism.core.errors import ConfigurationError, ValidationError
from prism.models.activations import GELUActivation, ReLUActivation, get_activation


@pytest.mark.unit
def test_relu_forward_positive_and_negative() -> None:
    """Verify ReLU maps negative values to 0.0 and preserves positive values."""
    relu = ReLUActivation()
    x = [[-2.0, -0.5, 0.0, 1.5, 3.0]]
    out = relu.forward(x)
    assert out == [[0.0, 0.0, 0.0, 1.5, 3.0]]


@pytest.mark.unit
def test_relu_backward_derivatives() -> None:
    """Verify ReLU backward propagates gradient only where input > 0."""
    relu = ReLUActivation()
    x = [[-2.0, 0.0, 1.5, 3.0]]
    d_out = [[1.0, 1.0, 1.0, 1.0]]
    d_in = relu.backward(x, d_out)
    assert d_in == [[0.0, 0.0, 1.0, 1.0]]


@pytest.mark.unit
def test_relu_backward_dimension_mismatch() -> None:
    """Verify ReLU backward raises ValidationError on shape mismatch."""
    relu = ReLUActivation()
    with pytest.raises(ValidationError, match="Shape mismatch in ReLU backward"):
        relu.backward([[1.0]], [[1.0], [2.0]])


@pytest.mark.unit
def test_gelu_forward_smoothness() -> None:
    """Verify GELU forward produces expected smooth Gaussian-gated values."""
    gelu = GELUActivation()
    x = [[0.0, 1.0, -1.0, 2.0]]
    out = gelu.forward(x)

    # At x=0, GELU(0) = 0
    assert out[0][0] == pytest.approx(0.0, abs=1e-5)
    # At x=1, GELU(1) ≈ 0.8413
    assert out[0][1] == pytest.approx(0.8413, abs=1e-3)
    # At x=-1, GELU(-1) ≈ -0.1587
    assert out[0][2] == pytest.approx(-0.1587, abs=1e-3)
    # At x=2, GELU(2) ≈ 1.9545
    assert out[0][3] == pytest.approx(1.9545, abs=1e-3)


@pytest.mark.unit
def test_gelu_backward_derivatives() -> None:
    """Verify GELU backward computes finite, consistent gradient tensors."""
    gelu = GELUActivation()
    x = [[0.0, 1.0, -1.0]]
    d_out = [[1.0, 1.0, 1.0]]
    d_in = gelu.backward(x, d_out)

    assert len(d_in) == 1
    assert len(d_in[0]) == 3
    # At x=0, derivative is 0.5
    assert d_in[0][0] == pytest.approx(0.5, abs=1e-3)
    assert d_in[0][1] > 0.0


@pytest.mark.unit
def test_get_activation_factory() -> None:
    """Verify get_activation returns correct instance and fails on unsupported."""
    assert isinstance(get_activation("relu"), ReLUActivation)
    assert isinstance(get_activation("gelu"), GELUActivation)

    with pytest.raises(ConfigurationError, match="Unsupported activation"):
        get_activation("sigmoid")
