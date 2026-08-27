"""Unit tests for learning rate schedulers."""

import pytest

from prism.core.errors import ConfigurationError, ValidationError
from prism.training.configuration import SchedulerSpecification
from prism.training.schedulers import (
    ConstantLRScheduler,
    CosineAnnealingLRScheduler,
    StepLRScheduler,
    create_scheduler,
)


@pytest.mark.unit
def test_constant_scheduler() -> None:
    """Verify ConstantLRScheduler returns base_lr across all epochs."""
    sched = ConstantLRScheduler(base_lr=0.05, total_epochs=10)
    for epoch in range(10):
        lr = sched.step(epoch)
        assert lr == 0.05
    assert len(sched.history) == 10
    assert all(r == 0.05 for r in sched.history)


@pytest.mark.unit
def test_step_scheduler_decay() -> None:
    """Verify StepLRScheduler decays at expected step_size boundaries."""
    sched = StepLRScheduler(
        base_lr=0.1,
        total_epochs=90,
        step_size=30,
        gamma=0.1,
        warmup_epochs=0,
    )
    # Epochs 0-29: 0.1
    assert sched.get_lr(0) == pytest.approx(0.1)
    assert sched.get_lr(29) == pytest.approx(0.1)

    # Epochs 30-59: 0.01
    assert sched.get_lr(30) == pytest.approx(0.01)
    assert sched.get_lr(59) == pytest.approx(0.01)

    # Epochs 60+: 0.001
    assert sched.get_lr(60) == pytest.approx(0.001)


@pytest.mark.unit
def test_step_scheduler_with_warmup() -> None:
    """Verify StepLRScheduler ramps linearly during warmup."""
    sched = StepLRScheduler(
        base_lr=0.1,
        total_epochs=50,
        step_size=20,
        gamma=0.5,
        warmup_epochs=5,
    )
    # Warmup epochs 0 to 4: linear ramp
    assert sched.get_lr(0) == pytest.approx(0.02)  # (1/5) * 0.1
    assert sched.get_lr(4) == pytest.approx(0.10)  # (5/5) * 0.1

    # Post-warmup effective epoch 0 (epoch 5): 0.1
    assert sched.get_lr(5) == pytest.approx(0.10)


@pytest.mark.unit
def test_cosine_annealing_scheduler() -> None:
    """Verify CosineAnnealingLRScheduler smoothly decays to min_lr."""
    sched = CosineAnnealingLRScheduler(
        base_lr=0.1,
        total_epochs=100,
        min_lr=0.001,
        warmup_epochs=0,
    )
    # Epoch 0: base_lr
    assert sched.get_lr(0) == pytest.approx(0.1)

    # Epoch 50 (midway cosine decay): 0.0505
    assert sched.get_lr(50) == pytest.approx(0.0505, abs=1e-3)

    # Epoch 100 (final): min_lr
    assert sched.get_lr(100) == pytest.approx(0.001, abs=1e-4)


@pytest.mark.unit
def test_create_scheduler_factory() -> None:
    """Verify create_scheduler builds correct instances from specifications."""
    none_sched = create_scheduler(None, base_lr=0.01, total_epochs=10)
    assert isinstance(none_sched, ConstantLRScheduler)

    step_spec = SchedulerSpecification(type="step", step_size=10, gamma=0.5)
    step_sched = create_scheduler(step_spec, base_lr=0.01, total_epochs=30)
    assert isinstance(step_sched, StepLRScheduler)

    cos_spec = SchedulerSpecification(type="cosine", min_lr=1e-4)
    cos_sched = create_scheduler(cos_spec, base_lr=0.01, total_epochs=50)
    assert isinstance(cos_sched, CosineAnnealingLRScheduler)

    bad_spec = SchedulerSpecification(type="unknown_schedule")
    with pytest.raises(ConfigurationError, match="Unsupported scheduler type"):
        create_scheduler(bad_spec, base_lr=0.01, total_epochs=10)


@pytest.mark.unit
def test_scheduler_validation_errors() -> None:
    """Verify schedulers reject invalid parameters."""
    with pytest.raises(ValidationError, match="base_lr must be positive"):
        ConstantLRScheduler(base_lr=0.0, total_epochs=10)

    with pytest.raises(ValidationError, match="total_epochs must be positive"):
        ConstantLRScheduler(base_lr=0.01, total_epochs=0)

    with pytest.raises(ValidationError, match="step_size must be positive"):
        StepLRScheduler(base_lr=0.01, total_epochs=10, step_size=0)
