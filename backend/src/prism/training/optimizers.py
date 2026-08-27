"""Optimizer implementations for PRISM trainable models."""

from abc import ABC, abstractmethod
from typing import Any

from prism.core.errors import ConfigurationError, ValidationError
from prism.models.base import BaseVisionModel
from prism.training.configuration import OptimizerSpecification


def _zero_like(val: Any) -> Any:
    """Create a matching nested structure of zeros for velocity buffers."""
    if isinstance(val, list):
        return [_zero_like(item) for item in val]
    return 0.0


def _apply_sgd_update(
    param: Any,
    grad: Any,
    vel: Any,
    lr: float,
    momentum: float,
    weight_decay: float,
    is_weight: bool,
) -> None:
    """Recursively apply SGD in-place update with momentum and weight decay."""
    if isinstance(param, list):
        for i in range(len(param)):
            if isinstance(param[i], list):
                _apply_sgd_update(
                    param=param[i],
                    grad=grad[i],
                    vel=vel[i] if vel is not None else None,
                    lr=lr,
                    momentum=momentum,
                    weight_decay=weight_decay,
                    is_weight=is_weight,
                )
            else:
                g = float(grad[i])
                if is_weight and weight_decay > 0.0:
                    g += weight_decay * float(param[i])

                if momentum > 0.0 and vel is not None:
                    vel[i] = momentum * float(vel[i]) + g
                    delta = vel[i]
                else:
                    delta = g

                param[i] -= lr * delta


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
    """Stochastic Gradient Descent optimizer supporting arbitrary tensor dimensions."""

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

        # Initialize velocity buffers matching parameter shapes
        params = model.get_parameters()
        self.velocity: dict[str, Any] = {k: _zero_like(v) for k, v in params.items()}

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
            is_weight = "weight" in k.lower()

            _apply_sgd_update(
                param=param_val,
                grad=grad_val,
                vel=v_val,
                lr=self._lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
                is_weight=is_weight,
            )

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
