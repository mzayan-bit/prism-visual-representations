"""Optimizer implementations for PRISM trainable models."""

from abc import ABC, abstractmethod
from typing import Any

from prism.core.errors import ConfigurationError, ValidationError
from prism.models.base import BaseVisionModel
from prism.training.configuration import OptimizerSpecification


class BaseOptimizer(ABC):
    """Abstract base class for model optimizers."""

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
        self.lr = lr
        self.momentum = momentum or 0.0
        self.weight_decay = weight_decay

        # Initialize velocity buffers for parameters
        params = model.get_parameters()
        self.velocity: dict[str, Any] = {}
        if "weights" in params:
            weights = params["weights"]
            self.velocity["weights"] = [
                [0.0 for _ in range(len(weights[0]))] for _ in range(len(weights))
            ]
        if "bias" in params:
            bias = params["bias"]
            self.velocity["bias"] = [0.0 for _ in range(len(bias))]

    def step(self) -> None:
        """Update model parameters using computed gradients."""
        params = self.model.get_parameters()
        grads = self.model.get_gradients()

        if "weights" in params and "grad_weights" in grads:
            weights = params["weights"]
            grad_weights = grads["grad_weights"]
            v_weights = self.velocity.get("weights")

            d_rows = len(weights)
            c_cols = len(weights[0])

            for d in range(d_rows):
                for c in range(c_cols):
                    grad = grad_weights[d][c]
                    if self.weight_decay > 0.0:
                        grad += self.weight_decay * weights[d][c]

                    if self.momentum > 0.0 and v_weights is not None:
                        v_weights[d][c] = self.momentum * v_weights[d][c] + grad
                        delta = v_weights[d][c]
                    else:
                        delta = grad

                    weights[d][c] -= self.lr * delta

        if "bias" in params and "grad_bias" in grads:
            bias = params["bias"]
            grad_bias = grads["grad_bias"]
            v_bias = self.velocity.get("bias")

            c_cols = len(bias)
            for c in range(c_cols):
                grad_b = grad_bias[c]
                if self.momentum > 0.0 and v_bias is not None:
                    v_bias[c] = self.momentum * v_bias[c] + grad_b
                    delta_b = v_bias[c]
                else:
                    delta_b = grad_b

                bias[c] -= self.lr * delta_b

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
        f"Unsupported optimizer type '{config.type}'. Phase 6 baseline supports 'sgd'."
    )
