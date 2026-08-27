"""Learning rate schedulers for deterministic training schedules."""

import math
from abc import ABC, abstractmethod

from prism.core.errors import ConfigurationError, ValidationError
from prism.training.configuration import SchedulerSpecification


class BaseLRScheduler(ABC):
    """Abstract base contract for learning rate schedulers."""

    def __init__(self, base_lr: float, total_epochs: int) -> None:
        if base_lr <= 0.0:
            raise ValidationError(f"base_lr must be positive, got {base_lr}.")
        if total_epochs <= 0:
            raise ValidationError(f"total_epochs must be positive, got {total_epochs}.")
        self.base_lr = base_lr
        self.total_epochs = total_epochs
        self._history: list[float] = []

    @abstractmethod
    def get_lr(self, epoch: int) -> float:
        """Compute and return learning rate for a specific 0-indexed epoch."""
        ...

    def step(self, epoch: int) -> float:
        """Step scheduler for given epoch, record into history, and return LR."""
        lr = self.get_lr(epoch)
        self._history.append(lr)
        return lr

    @property
    def history(self) -> list[float]:
        """Return full history of evaluated learning rates."""
        return list(self._history)


class ConstantLRScheduler(BaseLRScheduler):
    """Constant learning rate schedule: lr(epoch) = base_lr."""

    def get_lr(self, epoch: int) -> float:
        if epoch < 0:
            raise ValidationError(f"Epoch must be non-negative, got {epoch}.")
        return self.base_lr


class StepLRScheduler(BaseLRScheduler):
    """Step decay schedule: lr(epoch) = base_lr * (gamma ** (epoch // step_size))."""

    def __init__(
        self,
        base_lr: float,
        total_epochs: int,
        step_size: int = 30,
        gamma: float = 0.1,
        warmup_epochs: int = 0,
        min_lr: float = 0.0,
    ) -> None:
        super().__init__(base_lr, total_epochs)
        if step_size <= 0:
            raise ValidationError(f"step_size must be positive, got {step_size}.")
        if gamma <= 0.0 or gamma > 1.0:
            raise ValidationError(f"gamma must be in (0.0, 1.0], got {gamma}.")
        if warmup_epochs < 0:
            raise ValidationError(
                f"warmup_epochs must be non-negative, got {warmup_epochs}."
            )
        if min_lr < 0.0:
            raise ValidationError(f"min_lr must be non-negative, got {min_lr}.")

        self.step_size = step_size
        self.gamma = gamma
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr

    def get_lr(self, epoch: int) -> float:
        if epoch < 0:
            raise ValidationError(f"Epoch must be non-negative, got {epoch}.")

        # 1. Warmup phase
        if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
            warmup_factor = float(epoch + 1) / float(self.warmup_epochs)
            return max(self.min_lr, self.base_lr * warmup_factor)

        # 2. Step decay phase
        effective_epoch = epoch - self.warmup_epochs
        num_steps = effective_epoch // self.step_size
        decayed_lr = self.base_lr * (self.gamma**num_steps)
        return max(self.min_lr, decayed_lr)


class CosineAnnealingLRScheduler(BaseLRScheduler):
    """Cosine annealing schedule over total training epochs."""

    def __init__(
        self,
        base_lr: float,
        total_epochs: int,
        min_lr: float = 0.0,
        warmup_epochs: int = 0,
    ) -> None:
        super().__init__(base_lr, total_epochs)
        if min_lr < 0.0:
            raise ValidationError(f"min_lr must be non-negative, got {min_lr}.")
        if warmup_epochs < 0:
            raise ValidationError(
                f"warmup_epochs must be non-negative, got {warmup_epochs}."
            )
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs

    def get_lr(self, epoch: int) -> float:
        if epoch < 0:
            raise ValidationError(f"Epoch must be non-negative, got {epoch}.")

        # 1. Warmup phase
        if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
            warmup_factor = float(epoch + 1) / float(self.warmup_epochs)
            return max(self.min_lr, self.base_lr * warmup_factor)

        # 2. Cosine annealing phase
        effective_epoch = epoch - self.warmup_epochs
        horizon = max(1, self.total_epochs - self.warmup_epochs)
        clamped_epoch = min(effective_epoch, horizon)

        cos_factor = 0.5 * (
            1.0 + math.cos(math.pi * float(clamped_epoch) / float(horizon))
        )
        lr = self.min_lr + (self.base_lr - self.min_lr) * cos_factor
        return max(self.min_lr, lr)


def create_scheduler(
    spec: SchedulerSpecification | None,
    base_lr: float,
    total_epochs: int,
) -> BaseLRScheduler:
    """Factory function creating a learning rate scheduler from specification."""
    if spec is None or spec.type.lower() in ("none", "constant", ""):
        return ConstantLRScheduler(base_lr=base_lr, total_epochs=total_epochs)

    sched_type = spec.type.lower()
    if sched_type == "step":
        step_size = spec.step_size or 30
        gamma = spec.gamma if spec.gamma is not None else 0.1
        return StepLRScheduler(
            base_lr=base_lr,
            total_epochs=total_epochs,
            step_size=step_size,
            gamma=gamma,
            warmup_epochs=spec.warmup_epochs,
            min_lr=spec.min_lr,
        )

    if sched_type in ("cosine", "cosine_annealing"):
        return CosineAnnealingLRScheduler(
            base_lr=base_lr,
            total_epochs=total_epochs,
            min_lr=spec.min_lr,
            warmup_epochs=spec.warmup_epochs,
        )

    raise ConfigurationError(
        f"Unsupported scheduler type '{spec.type}'. "
        f"Supported: 'constant', 'step', 'cosine'."
    )
