"""Activation functions and their analytical derivatives."""

import math
from abc import ABC, abstractmethod
from typing import Any

from prism.core.errors import ConfigurationError, ValidationError


def _relu_forward(x: Any) -> Any:
    if isinstance(x, list):
        return [_relu_forward(item) for item in x]
    return max(0.0, float(x))


def _relu_backward(x: Any, d_out: Any) -> Any:
    if isinstance(x, list):
        if not isinstance(d_out, list) or len(x) != len(d_out):
            raise ValidationError("Shape mismatch in ReLU backward.")
        return [_relu_backward(x[i], d_out[i]) for i in range(len(x))]
    return float(d_out) if float(x) > 0.0 else 0.0


def _gelu_forward_val(val: float, sqrt_2_over_pi: float, coeff: float) -> float:
    cube = val * val * val
    inner = sqrt_2_over_pi * (val + coeff * cube)
    clamped_inner = max(-50.0, min(50.0, inner))
    return 0.5 * val * (1.0 + math.tanh(clamped_inner))


def _gelu_forward(x: Any, sqrt_2_over_pi: float, coeff: float) -> Any:
    if isinstance(x, list):
        return [_gelu_forward(item, sqrt_2_over_pi, coeff) for item in x]
    return _gelu_forward_val(float(x), sqrt_2_over_pi, coeff)


def _gelu_backward_val(
    val: float, dout_val: float, sqrt_2_over_pi: float, coeff: float
) -> float:
    cube = val * val * val
    inner = sqrt_2_over_pi * (val + coeff * cube)
    clamped_inner = max(-50.0, min(50.0, inner))
    t = math.tanh(clamped_inner)
    dt_dx = (1.0 - t * t) * sqrt_2_over_pi * (1.0 + 3.0 * coeff * val * val)
    df_dx = 0.5 * (1.0 + t) + 0.5 * val * dt_dx
    return dout_val * df_dx


def _gelu_backward(
    x: Any, d_out: Any, sqrt_2_over_pi: float, coeff: float
) -> Any:
    if isinstance(x, list):
        if not isinstance(d_out, list) or len(x) != len(d_out):
            raise ValidationError("Shape mismatch in GELU backward.")
        return [
            _gelu_backward(x[i], d_out[i], sqrt_2_over_pi, coeff)
            for i in range(len(x))
        ]
    return _gelu_backward_val(float(x), float(d_out), sqrt_2_over_pi, coeff)


class BaseActivation(ABC):
    """Abstract base class for activation functions."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the activation function."""
        ...

    @abstractmethod
    def forward(self, x: Any) -> Any:
        """Apply activation elementwise to input tensor (2D or 4D)."""
        ...

    @abstractmethod
    def backward(self, x: Any, d_out: Any) -> Any:
        """Compute upstream derivative given pre-activation x and upstream d_out."""
        ...


class ReLUActivation(BaseActivation):
    """Rectified Linear Unit: f(x) = max(0, x)."""

    @property
    def name(self) -> str:
        return "relu"

    def forward(self, x: Any) -> Any:
        return _relu_forward(x)

    def backward(self, x: Any, d_out: Any) -> Any:
        return _relu_backward(x, d_out)


class GELUActivation(BaseActivation):
    """Gaussian Error Linear Unit (approximated via tanh formula)."""

    SQRT_2_OVER_PI: float = math.sqrt(2.0 / math.pi)
    COEFF: float = 0.044715

    @property
    def name(self) -> str:
        return "gelu"

    def forward(self, x: Any) -> Any:
        return _gelu_forward(x, self.SQRT_2_OVER_PI, self.COEFF)

    def backward(self, x: Any, d_out: Any) -> Any:
        return _gelu_backward(x, d_out, self.SQRT_2_OVER_PI, self.COEFF)


def get_activation(name: str = "relu") -> BaseActivation:
    """Factory function returning the corresponding activation instance."""
    norm = name.strip().lower()
    if norm == "relu":
        return ReLUActivation()
    if norm == "gelu":
        return GELUActivation()
    raise ConfigurationError(
        f"Unsupported activation function '{name}'. Supported: 'relu', 'gelu'."
    )
