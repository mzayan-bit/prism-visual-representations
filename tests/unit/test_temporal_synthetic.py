"""Unit tests for synthetic video generator and motion trajectory contracts."""

from prism.core.enums import SplitName
from prism.temporal.synthetic import SyntheticVideoGenerator


def test_synthetic_video_generator_determinism() -> None:
    gen1 = SyntheticVideoGenerator(num_frames=4, height=12, width=12, seed=42)
    gen2 = SyntheticVideoGenerator(num_frames=4, height=12, width=12, seed=42)

    ds1 = gen1.generate_dataset(num_samples=4, split=SplitName.TRAIN)
    ds2 = gen2.generate_dataset(num_samples=4, split=SplitName.TRAIN)

    assert len(ds1) == len(ds2) == 4
    for s1, s2 in zip(ds1, ds2, strict=True):
        assert s1.video_id == s2.video_id
        assert s1.label == s2.label
        assert s1.dataset_fingerprint == s2.dataset_fingerprint
        assert s1.frame_tensors == s2.frame_tensors


def test_synthetic_motion_trajectories() -> None:
    gen = SyntheticVideoGenerator(num_frames=4, height=12, width=12, seed=42)
    ds = gen.generate_dataset(num_samples=4, split=SplitName.TRAIN)

    # Class 0: left_to_right -> x strictly increases
    s0 = ds[0]
    assert s0.motion_trajectory is not None
    assert s0.motion_trajectory.direction == "left_to_right"
    xs = [pos[0] for pos in s0.motion_trajectory.per_frame_positions]
    assert xs[0] < xs[1] < xs[2] < xs[3]
    assert not s0.motion_trajectory.is_stationary

    # Class 1: right_to_left -> x strictly decreases
    s1 = ds[1]
    assert s1.motion_trajectory is not None
    assert s1.motion_trajectory.direction == "right_to_left"
    xs_rev = [pos[0] for pos in s1.motion_trajectory.per_frame_positions]
    assert xs_rev[0] > xs_rev[1] > xs_rev[2] > xs_rev[3]

    # Class 3: stationary -> position constant
    s3 = ds[3]
    assert s3.motion_trajectory is not None
    assert s3.motion_trajectory.direction == "stationary"
    assert s3.motion_trajectory.is_stationary
    assert s3.motion_trajectory.velocity_magnitude == 0.0
    for pos in s3.motion_trajectory.per_frame_positions:
        assert pos == s3.motion_trajectory.start_pos


def test_static_sequence_control_invariant() -> None:
    gen = SyntheticVideoGenerator(num_frames=4, height=12, width=12, seed=42)
    ds = gen.generate_dataset(num_samples=1, split=SplitName.TRAIN)
    base_sample = ds[0]

    static_sample = gen.generate_static_sequence(base_sample)
    assert static_sample.frame_count == base_sample.frame_count
    assert static_sample.metadata.get("is_static_control") is True

    # All frames must be bitwise identical to frame 0
    frame_0 = static_sample.frame_tensors[0]
    for t_i in range(1, static_sample.frame_count):
        assert static_sample.frame_tensors[t_i] == frame_0

    assert static_sample.motion_trajectory is not None
    assert static_sample.motion_trajectory.is_stationary
    assert static_sample.motion_trajectory.velocity_magnitude == 0.0
