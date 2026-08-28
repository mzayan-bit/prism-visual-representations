"""Learning rate schedulers for reproducible and deterministic optimization control."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

from prism.core.errors import ConfigurationError, ValidationError
from prism.training.configuration import SchedulerSpecification
from prism.training.scheduler_state import SchedulerState


class BaseLRScheduler(ABC):
    """Abstract base contract for reproducible learning rate schedulers."""

    def __init__(
        self,
        initial_lr: float,
        total_epochs: int | None = None,
        total_steps: int | None = None,
        min_lr: float = 0.0,
        step_unit: str = "epoch",
    ) -> None:
        if math.isnan(initial_lr) or math.isinf(initial_lr) or initial_lr <= 0.0:
            raise ValidationError(f"base_lr must be positive, got {initial_lr}.")
        if total_epochs is not None and total_epochs <= 0:
            raise ValidationError(f"total_epochs must be positive, got {total_epochs}.")
        if total_steps is not None and total_steps <= 0:
            raise ValidationError(f"total_steps must be positive, got {total_steps}.")
        if math.isnan(min_lr) or math.isinf(min_lr) or min_lr < 0.0:
            raise ValidationError(
                f"min_lr must be a non-negative finite float, got {min_lr}."
            )
        if min_lr > initial_lr:
            raise ValidationError(
                f"min_lr ({min_lr}) cannot exceed initial_lr ({initial_lr})."
            )
        if step_unit not in ("epoch", "step"):
            raise ValidationError(
                f"step_unit must be 'epoch' or 'step', got '{step_unit}'."
            )

        self.initial_lr = initial_lr
        self.total_epochs = total_epochs
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.step_unit = step_unit

        self.current_step: int = 0
        self.current_epoch: int = 0
        self._history: list[float] = []

    @property
    def base_lr(self) -> float:
        """Alias for initial_lr for backward compatibility."""
        return self.initial_lr

    @property
    def lr(self) -> float:
        """Current effective learning rate."""
        if self._history:
            return self._history[-1]
        return self.get_lr_at(self.current_step, epoch=self.current_epoch)

    @property
    def history(self) -> list[float]:
        """Full historical progression of emitted learning rates."""
        return list(self._history)

    @abstractmethod
    def get_lr_at(self, step: int, epoch: int | None = None) -> float:
        """Compute learning rate at a given step/epoch without state mutation."""
        ...

    def get_lr(self, epoch: int) -> float:
        """Legacy helper computing learning rate for a given epoch index."""
        return self.get_lr_at(epoch, epoch=epoch)

    def step(self, epoch: int | None = None) -> float:
        """Advance scheduler progress by one step and return current LR."""
        if epoch is not None:
            if epoch < 0:
                raise ValidationError(f"Epoch must be non-negative, got {epoch}.")
            self.current_epoch = epoch

        lr = self.get_lr_at(self.current_step, epoch=self.current_epoch)
        if math.isnan(lr) or math.isinf(lr) or lr < 0.0:
            raise ValidationError(f"Scheduler emitted invalid learning rate: {lr}.")

        self._history.append(lr)
        self.current_step += 1
        return lr

    def get_state(self) -> SchedulerState:
        """Capture full reproducible snapshot of scheduler state."""
        return SchedulerState(
            schedule_type=self._get_schedule_type_name(),
            initial_lr=self.initial_lr,
            current_lr=self.lr,
            current_step=self.current_step,
            current_epoch=self.current_epoch,
            total_steps=self.total_steps,
            total_epochs=self.total_epochs,
            step_unit=self.step_unit,
            min_lr=self.min_lr,
            hyperparameters=self._get_hyperparameters(),
            warmup_progress=self._get_warmup_progress(),
            is_warmup_completed=self._get_is_warmup_completed(),
            history=list(self._history),
        )

    def set_state(self, state: SchedulerState | dict[str, Any]) -> None:
        """Restore scheduler state from a snapshot."""
        if isinstance(state, dict):
            state_obj = SchedulerState.from_dict(state)
        else:
            state_obj = state

        expected_type = self._get_schedule_type_name()
        if state_obj.schedule_type != expected_type:
            raise ValidationError(
                f"State schedule_type mismatch: expected '{expected_type}', "
                f"got '{state_obj.schedule_type}'."
            )

        self.initial_lr = state_obj.initial_lr
        self.current_step = state_obj.current_step
        self.current_epoch = state_obj.current_epoch
        self.total_steps = state_obj.total_steps
        self.total_epochs = state_obj.total_epochs
        self.step_unit = state_obj.step_unit
        self.min_lr = state_obj.min_lr
        self._history = list(state_obj.history)
        self._restore_hyperparameters(state_obj.hyperparameters)

    def _get_schedule_type_name(self) -> str:
        return "base"

    def _get_hyperparameters(self) -> dict[str, Any]:
        return {}

    def _restore_hyperparameters(self, hp: dict[str, Any]) -> None:
        _ = hp

    def _get_warmup_progress(self) -> float | None:
        return None

    def _get_is_warmup_completed(self) -> bool | None:
        return None


class ConstantLRScheduler(BaseLRScheduler):
    """Constant learning rate schedule: lr(t) = initial_lr."""

    def __init__(
        self,
        base_lr: float,
        total_epochs: int | None = None,
        total_steps: int | None = None,
        step_unit: str = "epoch",
    ) -> None:
        super().__init__(
            initial_lr=base_lr,
            total_epochs=total_epochs,
            total_steps=total_steps,
            min_lr=0.0,
            step_unit=step_unit,
        )

    def get_lr_at(self, step: int, epoch: int | None = None) -> float:
        if step < 0:
            raise ValidationError(f"Step must be non-negative, got {step}.")
        return self.initial_lr

    def _get_schedule_type_name(self) -> str:
        return "constant"


class StepLRScheduler(BaseLRScheduler):
    """Step decay schedule: lr(t) = max(min_lr, initial_lr * (gamma ** k))."""

    def __init__(
        self,
        base_lr: float,
        total_epochs: int | None = None,
        step_size: int = 30,
        gamma: float = 0.1,
        warmup_epochs: int = 0,
        min_lr: float = 0.0,
        total_steps: int | None = None,
        step_unit: str = "epoch",
    ) -> None:
        super().__init__(
            initial_lr=base_lr,
            total_epochs=total_epochs,
            total_steps=total_steps,
            min_lr=min_lr,
            step_unit=step_unit,
        )
        if step_size <= 0:
            raise ValidationError(f"step_size must be positive, got {step_size}.")
        if math.isnan(gamma) or math.isinf(gamma) or gamma <= 0.0 or gamma > 1.0:
            raise ValidationError(f"gamma must be in (0.0, 1.0], got {gamma}.")
        if warmup_epochs < 0:
            raise ValidationError(
                f"warmup_epochs must be non-negative, got {warmup_epochs}."
            )

        self.step_size = step_size
        self.gamma = gamma
        self.warmup_epochs = warmup_epochs

    def get_lr_at(self, step: int, epoch: int | None = None) -> float:
        if step < 0:
            raise ValidationError(f"Step must be non-negative, got {step}.")

        # Handle legacy epoch warmup if specified directly on class
        idx = epoch if (self.step_unit == "epoch" and epoch is not None) else step
        if self.warmup_epochs > 0 and idx < self.warmup_epochs:
            warmup_factor = float(idx + 1) / float(self.warmup_epochs)
            return max(self.min_lr, self.initial_lr * warmup_factor)

        effective_step = idx - self.warmup_epochs if self.warmup_epochs > 0 else idx
        k = effective_step // self.step_size
        decayed_lr = float(self.initial_lr * (self.gamma**k))
        return max(self.min_lr, decayed_lr)

    def _get_schedule_type_name(self) -> str:
        return "step"

    def _get_hyperparameters(self) -> dict[str, Any]:
        return {
            "step_size": self.step_size,
            "gamma": self.gamma,
            "warmup_epochs": self.warmup_epochs,
        }

    def _restore_hyperparameters(self, hp: dict[str, Any]) -> None:
        self.step_size = hp.get("step_size", self.step_size)
        self.gamma = hp.get("gamma", self.gamma)
        self.warmup_epochs = hp.get("warmup_epochs", self.warmup_epochs)


class ExponentialLRScheduler(BaseLRScheduler):
    """Exponential decay schedule: lr(t) = max(min_lr, initial_lr * (gamma ** f(t)))."""

    def __init__(
        self,
        base_lr: float,
        gamma: float = 0.95,
        decay_steps: int = 1,
        total_epochs: int | None = None,
        total_steps: int | None = None,
        min_lr: float = 0.0,
        step_unit: str = "epoch",
    ) -> None:
        super().__init__(
            initial_lr=base_lr,
            total_epochs=total_epochs,
            total_steps=total_steps,
            min_lr=min_lr,
            step_unit=step_unit,
        )
        if math.isnan(gamma) or math.isinf(gamma) or gamma <= 0.0 or gamma > 1.0:
            raise ValidationError(f"gamma must be in (0.0, 1.0], got {gamma}.")
        if decay_steps <= 0:
            raise ValidationError(f"decay_steps must be positive, got {decay_steps}.")

        self.gamma = gamma
        self.decay_steps = decay_steps

    def get_lr_at(self, step: int, epoch: int | None = None) -> float:
        if step < 0:
            raise ValidationError(f"Step must be non-negative, got {step}.")

        idx = epoch if (self.step_unit == "epoch" and epoch is not None) else step
        decay_ratio = float(idx) / float(self.decay_steps)
        decayed_lr = float(self.initial_lr * (self.gamma**decay_ratio))
        return max(self.min_lr, decayed_lr)

    def _get_schedule_type_name(self) -> str:
        return "exponential"

    def _get_hyperparameters(self) -> dict[str, Any]:
        return {
            "gamma": self.gamma,
            "decay_steps": self.decay_steps,
        }

    def _restore_hyperparameters(self, hp: dict[str, Any]) -> None:
        self.gamma = hp.get("gamma", self.gamma)
        self.decay_steps = hp.get("decay_steps", self.decay_steps)


class CosineAnnealingLRScheduler(BaseLRScheduler):
    """Cosine annealing decay schedule over a planned training horizon."""

    def __init__(
        self,
        base_lr: float,
        total_epochs: int | None = None,
        total_steps: int | None = None,
        min_lr: float = 0.0,
        warmup_epochs: int = 0,
        step_unit: str = "epoch",
    ) -> None:
        super().__init__(
            initial_lr=base_lr,
            total_epochs=total_epochs,
            total_steps=total_steps,
            min_lr=min_lr,
            step_unit=step_unit,
        )
        if warmup_epochs < 0:
            raise ValidationError(
                f"warmup_epochs must be non-negative, got {warmup_epochs}."
            )

        horizon = total_steps if step_unit == "step" else total_epochs
        if horizon is None or horizon <= 0:
            raise ValidationError(
                f"CosineAnnealing requires positive horizon for {step_unit}s, "
                f"got {horizon}."
            )

        self.warmup_epochs = warmup_epochs

    def get_lr_at(self, step: int, epoch: int | None = None) -> float:
        if step < 0:
            raise ValidationError(f"Step must be non-negative, got {step}.")

        idx = epoch if (self.step_unit == "epoch" and epoch is not None) else step

        # Handle legacy epoch warmup if set directly
        if self.warmup_epochs > 0 and idx < self.warmup_epochs:
            warmup_factor = float(idx + 1) / float(self.warmup_epochs)
            return max(self.min_lr, self.initial_lr * warmup_factor)

        effective_step = idx - self.warmup_epochs if self.warmup_epochs > 0 else idx
        total_h = (
            self.total_steps if self.step_unit == "step" else self.total_epochs
        ) or 1
        horizon = max(1, total_h - self.warmup_epochs)
        clamped_step = min(effective_step, horizon)

        progress = float(clamped_step) / float(horizon)
        cos_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = self.min_lr + (self.initial_lr - self.min_lr) * cos_factor
        return max(self.min_lr, lr)

    def _get_schedule_type_name(self) -> str:
        return "cosine"

    def _get_hyperparameters(self) -> dict[str, Any]:
        return {
            "warmup_epochs": self.warmup_epochs,
        }

    def _restore_hyperparameters(self, hp: dict[str, Any]) -> None:
        self.warmup_epochs = hp.get("warmup_epochs", self.warmup_epochs)


class LinearWarmupScheduler(BaseLRScheduler):
    """Linear warmup schedule interpolating from warmup_start_lr to target_lr."""

    def __init__(
        self,
        target_lr: float,
        warmup_steps: int,
        warmup_start_lr: float = 0.0,
        total_epochs: int | None = None,
        total_steps: int | None = None,
        step_unit: str = "step",
    ) -> None:
        if warmup_steps <= 0:
            raise ValidationError(f"warmup_steps must be positive, got {warmup_steps}.")
        if (
            math.isnan(warmup_start_lr)
            or math.isinf(warmup_start_lr)
            or warmup_start_lr < 0.0
        ):
            raise ValidationError(
                f"warmup_start_lr must be non-negative finite float, "
                f"got {warmup_start_lr}."
            )
        if warmup_start_lr > target_lr:
            raise ValidationError(
                f"warmup_start_lr ({warmup_start_lr}) cannot exceed "
                f"target_lr ({target_lr})."
            )

        super().__init__(
            initial_lr=target_lr,
            total_epochs=total_epochs,
            total_steps=total_steps,
            min_lr=warmup_start_lr,
            step_unit=step_unit,
        )
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr
        self.target_lr = target_lr

    def get_lr_at(self, step: int, epoch: int | None = None) -> float:
        if step < 0:
            raise ValidationError(f"Step must be non-negative, got {step}.")

        idx = epoch if (self.step_unit == "epoch" and epoch is not None) else step
        clamped_step = min(idx, self.warmup_steps)
        progress = float(clamped_step) / float(self.warmup_steps)
        return self.warmup_start_lr + (self.target_lr - self.warmup_start_lr) * progress

    def _get_schedule_type_name(self) -> str:
        return "linear_warmup"

    def _get_hyperparameters(self) -> dict[str, Any]:
        return {
            "warmup_steps": self.warmup_steps,
            "warmup_start_lr": self.warmup_start_lr,
            "target_lr": self.target_lr,
        }

    def _restore_hyperparameters(self, hp: dict[str, Any]) -> None:
        self.warmup_steps = hp.get("warmup_steps", self.warmup_steps)
        self.warmup_start_lr = hp.get("warmup_start_lr", self.warmup_start_lr)
        self.target_lr = hp.get("target_lr", self.target_lr)

    def _get_warmup_progress(self) -> float | None:
        return min(1.0, float(self.current_step) / float(self.warmup_steps))

    def _get_is_warmup_completed(self) -> bool | None:
        return self.current_step >= self.warmup_steps


class WarmupScheduler(BaseLRScheduler):
    """Composed schedule combining Linear Warmup with an after_scheduler."""

    def __init__(
        self,
        after_scheduler: BaseLRScheduler,
        warmup_steps: int,
        warmup_start_lr: float = 0.0,
        step_unit: str | None = None,
    ) -> None:
        if after_scheduler is None:
            raise ValidationError("after_scheduler cannot be None.")
        if warmup_steps <= 0:
            raise ValidationError(f"warmup_steps must be positive, got {warmup_steps}.")
        if (
            math.isnan(warmup_start_lr)
            or math.isinf(warmup_start_lr)
            or warmup_start_lr < 0.0
        ):
            raise ValidationError(
                f"warmup_start_lr must be non-negative finite float, "
                f"got {warmup_start_lr}."
            )

        unit = step_unit or after_scheduler.step_unit
        super().__init__(
            initial_lr=after_scheduler.initial_lr,
            total_epochs=after_scheduler.total_epochs,
            total_steps=after_scheduler.total_steps,
            min_lr=min(warmup_start_lr, after_scheduler.min_lr),
            step_unit=unit,
        )
        self.after_scheduler = after_scheduler
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr

    def get_lr_at(self, step: int, epoch: int | None = None) -> float:
        if step < 0:
            raise ValidationError(f"Step must be non-negative, got {step}.")

        idx = epoch if (self.step_unit == "epoch" and epoch is not None) else step

        if idx < self.warmup_steps:
            progress = float(idx) / float(self.warmup_steps)
            return (
                self.warmup_start_lr
                + (self.after_scheduler.initial_lr - self.warmup_start_lr) * progress
            )

        # Post-warmup: evaluate downstream scheduler at shifted progress offset
        downstream_idx = idx - self.warmup_steps
        return self.after_scheduler.get_lr_at(step=downstream_idx, epoch=downstream_idx)

    def _get_schedule_type_name(self) -> str:
        return "composed_warmup"

    def _get_hyperparameters(self) -> dict[str, Any]:
        return {
            "warmup_steps": self.warmup_steps,
            "warmup_start_lr": self.warmup_start_lr,
        }

    def _restore_hyperparameters(self, hp: dict[str, Any]) -> None:
        self.warmup_steps = hp.get("warmup_steps", self.warmup_steps)
        self.warmup_start_lr = hp.get("warmup_start_lr", self.warmup_start_lr)

    def _get_warmup_progress(self) -> float | None:
        return min(1.0, float(self.current_step) / float(self.warmup_steps))

    def _get_is_warmup_completed(self) -> bool | None:
        return self.current_step >= self.warmup_steps

    def get_state(self) -> SchedulerState:
        base_state = super().get_state()
        inner_state = self.after_scheduler.get_state().to_dict()
        return SchedulerState(
            **base_state.model_dump(exclude={"composed_inner_state"}),
            composed_inner_state=inner_state,
        )

    def set_state(self, state: SchedulerState | dict[str, Any]) -> None:
        if isinstance(state, dict):
            state_obj = SchedulerState.from_dict(state)
        else:
            state_obj = state

        super().set_state(state_obj)
        if state_obj.composed_inner_state is not None:
            self.after_scheduler.set_state(state_obj.composed_inner_state)


def create_scheduler(
    spec: SchedulerSpecification | None,
    base_lr: float,
    total_epochs: int,
    total_steps: int | None = None,
) -> BaseLRScheduler:
    """Factory function creating a learning rate scheduler from specification."""
    if spec is None or spec.type.lower() in ("none", "constant", ""):
        return ConstantLRScheduler(
            base_lr=base_lr,
            total_epochs=total_epochs,
            total_steps=total_steps,
            step_unit=spec.step_unit if spec else "epoch",
        )

    sched_type = spec.type.lower()
    step_unit = spec.step_unit
    min_lr = spec.min_lr
    w_start = spec.warmup_start_lr if spec.warmup_start_lr > 0.0 else min_lr

    # Determine warmup duration and horizon matching step_unit
    if step_unit == "step":
        w_duration = spec.warmup_steps or spec.warmup_epochs
        horizon = total_steps or (total_epochs * 100)
    else:
        w_duration = spec.warmup_epochs or spec.warmup_steps
        horizon = total_epochs

    # 1. Linear Warmup Standalone
    if sched_type in ("linear", "linear_warmup", "warmup"):
        w_steps = w_duration or (spec.step_size or 5)
        return LinearWarmupScheduler(
            target_lr=base_lr,
            warmup_steps=w_steps,
            warmup_start_lr=w_start,
            total_epochs=total_epochs,
            total_steps=total_steps,
            step_unit=step_unit,
        )

    # 2. Step Decay
    if sched_type == "step":
        step_size = spec.step_size or 30
        gamma = spec.gamma if spec.gamma is not None else 0.1
        if w_duration > 0:
            inner_horizon = max(1, horizon - w_duration)
            step_sched = StepLRScheduler(
                base_lr=base_lr,
                total_epochs=inner_horizon if step_unit == "epoch" else total_epochs,
                total_steps=inner_horizon if step_unit == "step" else total_steps,
                step_size=step_size,
                gamma=gamma,
                min_lr=min_lr,
                step_unit=step_unit,
            )
            return WarmupScheduler(
                after_scheduler=step_sched,
                warmup_steps=w_duration,
                warmup_start_lr=w_start,
                step_unit=step_unit,
            )
        return StepLRScheduler(
            base_lr=base_lr,
            total_epochs=total_epochs,
            total_steps=total_steps,
            step_size=step_size,
            gamma=gamma,
            warmup_epochs=0,
            min_lr=min_lr,
            step_unit=step_unit,
        )

    # 3. Exponential Decay
    if sched_type in ("exponential", "exp"):
        gamma = spec.gamma if spec.gamma is not None else 0.95
        decay_steps = spec.decay_steps or spec.step_size or 1
        if w_duration > 0:
            inner_horizon = max(1, horizon - w_duration)
            exp_sched = ExponentialLRScheduler(
                base_lr=base_lr,
                gamma=gamma,
                decay_steps=decay_steps,
                total_epochs=inner_horizon if step_unit == "epoch" else total_epochs,
                total_steps=inner_horizon if step_unit == "step" else total_steps,
                min_lr=min_lr,
                step_unit=step_unit,
            )
            return WarmupScheduler(
                after_scheduler=exp_sched,
                warmup_steps=w_duration,
                warmup_start_lr=w_start,
                step_unit=step_unit,
            )
        return ExponentialLRScheduler(
            base_lr=base_lr,
            gamma=gamma,
            decay_steps=decay_steps,
            total_epochs=total_epochs,
            total_steps=total_steps,
            min_lr=min_lr,
            step_unit=step_unit,
        )

    # 4. Cosine Decay
    if sched_type in ("cosine", "cosine_annealing"):
        if w_duration > 0:
            inner_horizon = max(1, horizon - w_duration)
            cos_sched = CosineAnnealingLRScheduler(
                base_lr=base_lr,
                total_epochs=inner_horizon if step_unit == "epoch" else total_epochs,
                total_steps=inner_horizon if step_unit == "step" else total_steps,
                min_lr=min_lr,
                warmup_epochs=0,
                step_unit=step_unit,
            )
            return WarmupScheduler(
                after_scheduler=cos_sched,
                warmup_steps=w_duration,
                warmup_start_lr=w_start,
                step_unit=step_unit,
            )
        return CosineAnnealingLRScheduler(
            base_lr=base_lr,
            total_epochs=total_epochs,
            total_steps=total_steps,
            min_lr=min_lr,
            warmup_epochs=0,
            step_unit=step_unit,
        )

    raise ConfigurationError(
        f"Unsupported scheduler type '{spec.type}'. "
        f"Supported types: 'constant', 'step', 'exponential', 'cosine', 'linear'."
    )
