"""Unit tests for reproducible learning rate schedulers."""

import math

import pytest

from prism.core.errors import ValidationError
from prism.training.schedulers import (
    ConstantLRScheduler,
    CosineAnnealingLRScheduler,
    ExponentialLRScheduler,
    LinearWarmupScheduler,
    StepLRScheduler,
)


@pytest.mark.unit
def test_constant_lr_scheduler_lifecycle() -> None:
    """Verify ConstantLRScheduler emits invariant LR and tracks progress."""
    sched = ConstantLRScheduler(base_lr=0.05, total_epochs=5)
    assert sched.lr == 0.05
    assert sched.current_step == 0

    lrs = [sched.step(epoch=e) for e in range(5)]
    assert lrs == [0.05, 0.05, 0.05, 0.05, 0.05]
    assert sched.history == [0.05, 0.05, 0.05, 0.05, 0.05]
    assert sched.current_step == 5
    assert sched.current_epoch == 4


@pytest.mark.unit
def test_step_lr_scheduler_exact_boundaries() -> None:
    """Verify StepLRScheduler decay at exact step/epoch boundaries."""
    # base_lr=1.0, step_size=10, gamma=0.5
    sched = StepLRScheduler(
        base_lr=1.0,
        total_epochs=30,
        step_size=10,
        gamma=0.5,
        min_lr=0.01,
    )

    # Steps 0-9: lr = 1.0
    for i in range(10):
        lr = sched.step(epoch=i)
        assert pytest.approx(lr, abs=1e-7) == 1.0

    # Steps 10-19: lr = 0.5
    for i in range(10, 20):
        lr = sched.step(epoch=i)
        assert pytest.approx(lr, abs=1e-7) == 0.5

    # Steps 20-29: lr = 0.25
    for i in range(20, 30):
        lr = sched.step(epoch=i)
        assert pytest.approx(lr, abs=1e-7) == 0.25

    assert len(sched.history) == 30


@pytest.mark.unit
def test_step_lr_scheduler_min_lr_floor() -> None:
    """Verify StepLRScheduler respects min_lr floor."""
    sched = StepLRScheduler(
        base_lr=0.1,
        total_epochs=10,
        step_size=2,
        gamma=0.1,
        min_lr=0.005,
    )
    # step 0, 1: 0.1
    # step 2, 3: 0.01
    # step 4, 5: 0.001 -> clamped to 0.005
    assert pytest.approx(sched.get_lr_at(0), abs=1e-7) == 0.1
    assert pytest.approx(sched.get_lr_at(2), abs=1e-7) == 0.01
    assert pytest.approx(sched.get_lr_at(4), abs=1e-7) == 0.005
    assert pytest.approx(sched.get_lr_at(6), abs=1e-7) == 0.005


@pytest.mark.unit
def test_exponential_lr_scheduler_decay() -> None:
    """Verify ExponentialLRScheduler mathematical progression."""
    # gamma = 0.5, decay_steps = 2
    sched = ExponentialLRScheduler(
        base_lr=1.0,
        gamma=0.5,
        decay_steps=2,
        total_epochs=10,
    )
    # t = 0: 1.0 * (0.5 ** 0) = 1.0
    # t = 1: 1.0 * (0.5 ** 0.5) = sqrt(0.5) ~= 0.70710678
    # t = 2: 1.0 * (0.5 ** 1) = 0.5
    # t = 4: 1.0 * (0.5 ** 2) = 0.25
    assert pytest.approx(sched.get_lr_at(0), abs=1e-6) == 1.0
    assert pytest.approx(sched.get_lr_at(1), abs=1e-6) == math.sqrt(0.5)
    assert pytest.approx(sched.get_lr_at(2), abs=1e-6) == 0.5
    assert pytest.approx(sched.get_lr_at(4), abs=1e-6) == 0.25


@pytest.mark.unit
def test_exponential_lr_scheduler_gamma_one() -> None:
    """Verify ExponentialLRScheduler with gamma=1.0 remains constant."""
    sched = ExponentialLRScheduler(
        base_lr=0.03,
        gamma=1.0,
        decay_steps=1,
        total_epochs=5,
    )
    for _ in range(5):
        assert pytest.approx(sched.step(), abs=1e-7) == 0.03


@pytest.mark.unit
def test_cosine_annealing_lr_scheduler_progression() -> None:
    """Verify CosineAnnealingLRScheduler endpoint and midpoint mathematical behavior."""
    sched = CosineAnnealingLRScheduler(
        base_lr=0.1,
        total_epochs=10,
        min_lr=0.01,
        step_unit="epoch",
    )

    # t = 0: initial_lr = 0.1
    assert pytest.approx(sched.get_lr_at(0), abs=1e-7) == 0.1

    # t = 5 (midpoint): 0.01 + 0.5 * 0.09 * (1 + cos(pi/2)) = 0.01 + 0.045 = 0.055
    assert pytest.approx(sched.get_lr_at(5), abs=1e-7) == 0.055

    # t = 10 (horizon): 0.01 + 0.5 * 0.09 * (1 + cos(pi)) = 0.01 + 0 = 0.01
    assert pytest.approx(sched.get_lr_at(10), abs=1e-7) == 0.01

    # t = 12 (beyond horizon): clamped to min_lr = 0.01
    assert pytest.approx(sched.get_lr_at(12), abs=1e-7) == 0.01


@pytest.mark.unit
def test_linear_warmup_scheduler_progression() -> None:
    """Verify LinearWarmupScheduler starts at warmup_start_lr and reaches target_lr."""
    sched = LinearWarmupScheduler(
        target_lr=0.1,
        warmup_steps=10,
        warmup_start_lr=0.0,
    )

    # t = 0: 0.0
    assert pytest.approx(sched.get_lr_at(0), abs=1e-7) == 0.0
    # t = 5: 0.05
    assert pytest.approx(sched.get_lr_at(5), abs=1e-7) == 0.05
    # t = 10: 0.1
    assert pytest.approx(sched.get_lr_at(10), abs=1e-7) == 0.1
    # t = 15: clamped to 0.1
    assert pytest.approx(sched.get_lr_at(15), abs=1e-7) == 0.1


@pytest.mark.unit
def test_scheduler_validation_rejections() -> None:
    """Verify schedulers reject invalid configurations."""
    with pytest.raises(ValidationError, match="base_lr must be positive"):
        ConstantLRScheduler(base_lr=-0.1)

    with pytest.raises(ValidationError, match="base_lr must be positive"):
        ConstantLRScheduler(base_lr=float("nan"))

    with pytest.raises(ValidationError, match="min_lr"):
        ConstantLRScheduler(base_lr=0.1, total_epochs=5)
        CosineAnnealingLRScheduler(base_lr=0.1, total_epochs=5, min_lr=-0.01)

    with pytest.raises(ValidationError, match=r"min_lr .* cannot exceed initial_lr"):
        CosineAnnealingLRScheduler(base_lr=0.01, total_epochs=5, min_lr=0.05)

    with pytest.raises(ValidationError, match="step_size must be positive"):
        StepLRScheduler(base_lr=0.1, step_size=0)

    with pytest.raises(ValidationError, match="gamma must be in"):
        StepLRScheduler(base_lr=0.1, gamma=0.0)

    with pytest.raises(ValidationError, match="gamma must be in"):
        StepLRScheduler(base_lr=0.1, gamma=1.5)

    with pytest.raises(ValidationError, match="decay_steps must be positive"):
        ExponentialLRScheduler(base_lr=0.1, decay_steps=0)

    with pytest.raises(ValidationError, match="warmup_steps must be positive"):
        LinearWarmupScheduler(target_lr=0.1, warmup_steps=0)
