"""Unit tests for scheduler serialization and continuation."""

import pytest

from prism.training.schedulers import (
    ConstantLRScheduler,
    CosineAnnealingLRScheduler,
    ExponentialLRScheduler,
    StepLRScheduler,
    WarmupScheduler,
)


@pytest.mark.unit
def test_constant_scheduler_serialization_roundtrip() -> None:
    """Verify ConstantLRScheduler state serialization and continuation."""
    sched1 = ConstantLRScheduler(base_lr=0.04, total_epochs=10)
    for e in range(5):
        sched1.step(epoch=e)

    state = sched1.get_state()
    json_str = state.to_json()

    sched2 = ConstantLRScheduler(base_lr=0.01, total_epochs=10)
    sched2.set_state(state.from_json(json_str))

    assert sched2.current_step == 5
    assert sched2.current_epoch == 4
    assert sched2.history == sched1.history

    # Continuation produces identical sequence
    seq1 = [sched1.step(epoch=e) for e in range(5, 10)]
    seq2 = [sched2.step(epoch=e) for e in range(5, 10)]
    assert seq1 == seq2


@pytest.mark.unit
def test_step_scheduler_serialization_and_continuation() -> None:
    """Verify StepLRScheduler state restoration produces identical trajectory."""
    uninterrupted = StepLRScheduler(
        base_lr=1.0,
        total_epochs=20,
        step_size=5,
        gamma=0.5,
    )
    full_trajectory = [uninterrupted.step(epoch=e) for e in range(20)]

    interrupted = StepLRScheduler(
        base_lr=1.0,
        total_epochs=20,
        step_size=5,
        gamma=0.5,
    )
    first_half = [interrupted.step(epoch=e) for e in range(7)]

    state_json = interrupted.get_state().to_json()

    restored = StepLRScheduler(
        base_lr=0.01,
        total_epochs=1,
        step_size=1,
        gamma=0.9,
    )
    restored.set_state(interrupted.get_state().from_json(state_json))

    second_half = [restored.step(epoch=e) for e in range(7, 20)]

    assert first_half + second_half == full_trajectory


@pytest.mark.unit
def test_exponential_scheduler_serialization() -> None:
    """Verify ExponentialLRScheduler state restoration preserves decay progression."""
    uninterrupted = ExponentialLRScheduler(
        base_lr=0.5,
        gamma=0.8,
        decay_steps=2,
        total_epochs=12,
    )
    full_trajectory = [uninterrupted.step(epoch=e) for e in range(12)]

    interrupted = ExponentialLRScheduler(
        base_lr=0.5,
        gamma=0.8,
        decay_steps=2,
        total_epochs=12,
    )
    part1 = [interrupted.step(epoch=e) for e in range(6)]

    state_dict = interrupted.get_state().to_dict()

    restored = ExponentialLRScheduler(
        base_lr=0.1,
        gamma=0.5,
        decay_steps=1,
        total_epochs=5,
    )
    restored.set_state(state_dict)

    part2 = [restored.step(epoch=e) for e in range(6, 12)]

    assert [pytest.approx(v, abs=1e-7) for v in part1 + part2] == [
        pytest.approx(v, abs=1e-7) for v in full_trajectory
    ]


@pytest.mark.unit
def test_cosine_scheduler_serialization() -> None:
    """Verify CosineAnnealing restoration preserves cosine progression."""
    uninterrupted = CosineAnnealingLRScheduler(
        base_lr=0.1,
        total_epochs=20,
        min_lr=0.001,
    )
    full_trajectory = [uninterrupted.step(epoch=e) for e in range(20)]

    interrupted = CosineAnnealingLRScheduler(
        base_lr=0.1,
        total_epochs=20,
        min_lr=0.001,
    )
    part1 = [interrupted.step(epoch=e) for e in range(8)]

    state_json = interrupted.get_state().to_json()

    restored = CosineAnnealingLRScheduler(
        base_lr=0.5,
        total_epochs=5,
        min_lr=0.0,
    )
    restored.set_state(restored.get_state().from_json(state_json))

    part2 = [restored.step(epoch=e) for e in range(8, 20)]

    assert [pytest.approx(v, abs=1e-7) for v in part1 + part2] == [
        pytest.approx(v, abs=1e-7) for v in full_trajectory
    ]


@pytest.mark.unit
def test_composed_warmup_scheduler_serialization() -> None:
    """Verify WarmupScheduler with inner Cosine scheduler serializes composite state."""
    cos = CosineAnnealingLRScheduler(base_lr=0.1, total_epochs=15, min_lr=0.01)
    uninterrupted = WarmupScheduler(
        after_scheduler=cos,
        warmup_steps=5,
        warmup_start_lr=0.0,
    )
    full_trajectory = [uninterrupted.step(epoch=e) for e in range(20)]

    # Interrupt during post-warmup phase (at step 8)
    cos_interrupted = CosineAnnealingLRScheduler(
        base_lr=0.1, total_epochs=15, min_lr=0.01
    )
    interrupted = WarmupScheduler(
        after_scheduler=cos_interrupted,
        warmup_steps=5,
        warmup_start_lr=0.0,
    )
    part1 = [interrupted.step(epoch=e) for e in range(8)]

    state_dict = interrupted.get_state().to_dict()

    cos_restored = CosineAnnealingLRScheduler(base_lr=0.01, total_epochs=2, min_lr=0.0)
    restored = WarmupScheduler(
        after_scheduler=cos_restored,
        warmup_steps=1,
        warmup_start_lr=0.0,
    )
    restored.set_state(state_dict)

    part2 = [restored.step(epoch=e) for e in range(8, 20)]

    assert [pytest.approx(v, abs=1e-7) for v in part1 + part2] == [
        pytest.approx(v, abs=1e-7) for v in full_trajectory
    ]
