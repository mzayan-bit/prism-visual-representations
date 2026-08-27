"""Activation functions and their analytical derivatives."""

import math
from abc import ABC, abstractmethod

from prism.core.errors import ConfigurationError, ValidationError


class BaseActivation(ABC):
    """Abstract base class for activation functions."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the activation function."""
        ...

    @abstractmethod
    def forward(self, x: list[list[float]]) -> list[list[float]]:
        """Apply activation elementwise to a 2D tensor [B, D]."""
        ...

    @abstractmethod
    def backward(
        self, x: list[list[float]], d_out: list[list[float]]
    ) -> list[list[float]]:
        """Compute upstream derivative given pre-activation x and upstream d_out."""
        ...


class ReLUActivation(BaseActivation):
    """Rectified Linear Unit: f(x) = max(0, x)."""

    @property
    def name(self) -> str:
        return "relu"

    def forward(self, x: list[list[float]]) -> list[list[float]]:
        return [[max(0.0, val) for val in row] for row in x]

    def backward(
        self, x: list[list[float]], d_out: list[list[float]]
    ) -> list[list[float]]:
        if len(x) != len(d_out):
            raise ValidationError(
                f"Shape mismatch in ReLU backward: {len(x)} vs {len(d_out)}."
            )
        d_in: list[list[float]] = []
        for i in range(len(x)):
            x_row = x[i]
            d_row = d_out[i]
            if len(x_row) != len(d_row):
                raise ValidationError(
                    f"Row dimension mismatch in ReLU backward at index {i}."
                )
            row_grad = [
                d_row[j] if x_row[j] > 0.0 else 0.0 for j in range(len(x_row))
            ]
            d_in.append(row_grad)
        return d_in


class GELUActivation(BaseActivation):
    """Gaussian Error Linear Unit (approximated via tanh formula)."""

    SQRT_2_OVER_PI: float = math.sqrt(2.0 / math.pi)
    COEFF: float = 0.044715

    @property
    def name(self) -> str:
        return "gelu"

    def forward(self, x: list[list[float]]) -> list[list[float]]:
        out: list[list[float]] = []
        for row in x:
            out_row: list[float] = []
            for val in row:
                cube = val * val * val
                inner = self.SQRT_2_OVER_PI * (val + self.COEFF * cube)
                # Clamp inner to prevent overflow in math.tanh
                clamped_inner = max(-50.0, min(50.0, inner))
                res = 0.5 * val * (1.0 + math.tanh(clamped_inner))
                out_row.append(res)
            out.append(out_row)
        return out

    def backward(
        self, x: list[list[float]], d_out: list[list[float]]
    ) -> list[list[float]]:
        if len(x) != len(d_out):
            raise ValidationError(
                f"Shape mismatch in GELU backward: {len(x)} vs {len(d_out)}."
            )
        d_in: list[list[float]] = []
        for i in range(len(x)):
            x_row = x[i]
            d_row = d_out[i]
            if len(x_row) != len(d_row):
                raise ValidationError(
                    f"Row dimension mismatch in GELU backward at index {i}."
                )
            in_row: list[float] = []
            for j in range(len(x_row)):
                val = x_row[j]
                cube = val * val * val
                inner = self.SQRT_2_OVER_PI * (val + self.COEFF * cube)
                clamped_inner = max(-50.0, min(50.0, inner))
                t = math.tanh(clamped_inner)
                dt_dx = (
                    (1.0 - t * t)
                    * self.SQRT_2_OVER_PI
                    * (1.0 + 3.0 * self.COEFF * val * val)
                )
                df_dx = 0.5 * (1.0 + t) + 0.5 * val * dt_dx
                in_row.append(d_row[j] * df_dx)
            d_in.append(in_row)
        return d_in


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
