"""Optimizer implementations for PRISM trainable models."""

from abc import ABC, abstractmethod
from typing import Any

from prism.core.errors import ConfigurationError, ValidationError
from prism.models.base import BaseVisionModel
from prism.training.configuration import OptimizerSpecification


class BaseOptimizer(ABC):
    """Abstract base class for model optimizers."""

    @property
    @abstractmethod
    def lr(self) -> float:
        """Current effective learning rate."""
        ...

    @lr.setter
    @abstractmethod
    def lr(self, value: float) -> None:
        """Set current effective learning rate."""
        ...

    @abstractmethod
    def step(self) -> None:
        """Perform a single optimization step updating model parameters."""
        ...

    @abstractmethod
    def zero_grad(self) -> None:
        """Clear all parameter gradients in the associated model."""
        ...


class SGDOptimizer(BaseOptimizer):
    """Stochastic Gradient Descent optimizer with optional momentum and weight decay."""

    def __init__(
        self,
        model: BaseVisionModel,
        lr: float = 1e-2,
        momentum: float | None = None,
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0.0:
            raise ValidationError(f"Learning rate must be positive, got {lr}.")
        if weight_decay < 0.0:
            raise ValidationError(
                f"Weight decay must be non-negative, got {weight_decay}."
            )
        if momentum is not None and (momentum < 0.0 or momentum > 1.0):
            raise ValidationError(f"Momentum must be in [0.0, 1.0], got {momentum}.")

        self.model = model
        self._lr = lr
        self.momentum = momentum or 0.0
        self.weight_decay = weight_decay

        # Initialize velocity buffers for arbitrary parameter dictionaries
        params = model.get_parameters()
        self.velocity: dict[str, Any] = {}
        for k, v in params.items():
            if isinstance(v, list):
                if v and isinstance(v[0], list):
                    # 2D weight matrix
                    self.velocity[k] = [
                        [0.0 for _ in range(len(v[0]))] for _ in range(len(v))
                    ]
                else:
                    # 1D bias vector
                    self.velocity[k] = [0.0 for _ in range(len(v))]

    @property
    def lr(self) -> float:
        return self._lr

    @lr.setter
    def lr(self, value: float) -> None:
        if value < 0.0:
            raise ValidationError(f"Learning rate must be non-negative, got {value}.")
        self._lr = value

    def step(self) -> None:
        """Update model parameters using computed gradients."""
        params = self.model.get_parameters()
        grads = self.model.get_gradients()

        for k, param_val in params.items():
            grad_key = f"grad_{k}"
            if grad_key not in grads:
                continue

            grad_val = grads[grad_key]
            v_val = self.velocity.get(k)

            # Check if parameter is a 2D weight matrix
            if (
                isinstance(param_val, list)
                and param_val
                and isinstance(param_val[0], list)
            ):
                is_weight = "weight" in k.lower()
                rows = len(param_val)
                cols = len(param_val[0])

                for r in range(rows):
                    for c in range(cols):
                        g = grad_val[r][c]
                        if is_weight and self.weight_decay > 0.0:
                            g += self.weight_decay * param_val[r][c]

                        if self.momentum > 0.0 and v_val is not None:
                            v_val[r][c] = self.momentum * v_val[r][c] + g
                            delta = v_val[r][c]
                        else:
                            delta = g

                        param_val[r][c] -= self._lr * delta

            # Check if parameter is a 1D vector (bias)
            elif isinstance(param_val, list):
                is_weight = "weight" in k.lower()
                cols = len(param_val)

                for c in range(cols):
                    g = grad_val[c]
                    if is_weight and self.weight_decay > 0.0:
                        g += self.weight_decay * param_val[c]

                    if self.momentum > 0.0 and v_val is not None:
                        v_val[c] = self.momentum * v_val[c] + g
                        delta = v_val[c]
                    else:
                        delta = g

                    param_val[c] -= self._lr * delta

        self.model.set_parameters(params)

    def zero_grad(self) -> None:
        """Clear model gradients."""
        self.model.zero_grad()


def create_optimizer(
    config: OptimizerSpecification,
    model: BaseVisionModel,
) -> BaseOptimizer:
    """Factory function creating an optimizer from an OptimizerSpecification."""
    opt_type = config.type.lower()
    if opt_type == "sgd":
        return SGDOptimizer(
            model=model,
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    raise ConfigurationError(
        f"Unsupported optimizer type '{config.type}'. PRISM supports 'sgd'."
    )
