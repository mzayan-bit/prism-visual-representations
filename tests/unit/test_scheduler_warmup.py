"""Unit tests for warmup scheduler compositions and continuity."""

import pytest

from prism.training.schedulers import (
    ConstantLRScheduler,
    CosineAnnealingLRScheduler,
    ExponentialLRScheduler,
    StepLRScheduler,
    WarmupScheduler,
)


@pytest.mark.unit
def test_warmup_cosine_continuity() -> None:
    """Verify WarmupScheduler + CosineAnnealing transition is smooth without jump."""
    base_lr = 0.1
    warmup_steps = 4
    total_epochs = 16

    cos = CosineAnnealingLRScheduler(
        base_lr=base_lr,
        total_epochs=total_epochs - warmup_steps,
        min_lr=0.01,
    )
    sched = WarmupScheduler(
        after_scheduler=cos,
        warmup_steps=warmup_steps,
        warmup_start_lr=0.0,
    )

    # Warmup phase: step 0 to 3
    assert pytest.approx(sched.get_lr_at(0), abs=1e-7) == 0.0
    assert pytest.approx(sched.get_lr_at(1), abs=1e-7) == 0.025
    assert pytest.approx(sched.get_lr_at(2), abs=1e-7) == 0.050
    assert pytest.approx(sched.get_lr_at(3), abs=1e-7) == 0.075

    # Exactly at warmup boundary (step 4 = warmup_steps), reaches full base_lr = 0.1
    assert pytest.approx(sched.get_lr_at(4), abs=1e-7) == base_lr

    # Post-warmup begins cosine decay
    assert sched.get_lr_at(5) < base_lr

    # At horizon (step 16): reaches min_lr = 0.01
    assert pytest.approx(sched.get_lr_at(16), abs=1e-7) == 0.01


@pytest.mark.unit
def test_warmup_step_composition() -> None:
    """Verify WarmupScheduler + StepLRScheduler composition."""
    step_sched = StepLRScheduler(
        base_lr=1.0,
        total_epochs=20,
        step_size=5,
        gamma=0.5,
    )
    sched = WarmupScheduler(
        after_scheduler=step_sched,
        warmup_steps=5,
        warmup_start_lr=0.2,
    )

    # During warmup
    assert pytest.approx(sched.get_lr_at(0), abs=1e-7) == 0.2
    assert pytest.approx(sched.get_lr_at(2), abs=1e-7) == 0.2 + 0.8 * (2 / 5)

    # At warmup end (step 5)
    assert pytest.approx(sched.get_lr_at(5), abs=1e-7) == 1.0

    # Steps 5-9 (effective 0-4 of step_sched): lr = 1.0
    for t in range(5, 10):
        assert pytest.approx(sched.get_lr_at(t), abs=1e-7) == 1.0

    # Steps 10-14 (effective 5-9 of step_sched): lr = 0.5
    for t in range(10, 15):
        assert pytest.approx(sched.get_lr_at(t), abs=1e-7) == 0.5


@pytest.mark.unit
def test_warmup_exponential_composition() -> None:
    """Verify WarmupScheduler + ExponentialLRScheduler composition."""
    exp_sched = ExponentialLRScheduler(
        base_lr=0.5,
        gamma=0.9,
        decay_steps=2,
    )
    sched = WarmupScheduler(
        after_scheduler=exp_sched,
        warmup_steps=3,
        warmup_start_lr=0.0,
    )

    assert pytest.approx(sched.get_lr_at(0), abs=1e-7) == 0.0
    assert pytest.approx(sched.get_lr_at(3), abs=1e-7) == 0.5
    # Step 5 is effective step 2 for exp_sched -> 0.5 * (0.9 ** 1) = 0.45
    assert pytest.approx(sched.get_lr_at(5), abs=1e-7) == 0.45


@pytest.mark.unit
def test_warmup_constant_composition() -> None:
    """Verify WarmupScheduler + ConstantLRScheduler composition."""
    const_sched = ConstantLRScheduler(base_lr=0.08)
    sched = WarmupScheduler(
        after_scheduler=const_sched,
        warmup_steps=4,
        warmup_start_lr=0.02,
    )

    assert pytest.approx(sched.get_lr_at(0), abs=1e-7) == 0.02
    assert pytest.approx(sched.get_lr_at(4), abs=1e-7) == 0.08
    assert pytest.approx(sched.get_lr_at(8), abs=1e-7) == 0.08


@pytest.mark.unit
def test_warmup_progress_tracking() -> None:
    """Verify warmup progress indicators."""
    const_sched = ConstantLRScheduler(base_lr=0.1)
    sched = WarmupScheduler(
        after_scheduler=const_sched,
        warmup_steps=4,
    )

    assert sched.get_state().warmup_progress == 0.0
    assert sched.get_state().is_warmup_completed is False

    sched.step()
    sched.step()
    assert sched.get_state().warmup_progress == 0.5
    assert sched.get_state().is_warmup_completed is False

    sched.step()
    sched.step()
    assert sched.get_state().warmup_progress == 1.0
    assert sched.get_state().is_warmup_completed is True
