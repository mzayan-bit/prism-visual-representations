"""Unit tests for temporal data models, sequence invariants, and serialization."""

import pytest

from prism.core.enums import SplitName
from prism.core.errors import ValidationError
from prism.temporal.contracts import (
    FrameMetadata,
    MotionTrajectory,
    RNNDynamicsSummary,
    TemporalConsistencySummary,
    TemporalWeightSummary,
    VideoBatch,
    VideoSample,
)


def test_motion_trajectory_serialization() -> None:
    traj = MotionTrajectory(
        start_pos=(0.2, 0.5),
        end_pos=(0.8, 0.5),
        per_frame_positions=[(0.2, 0.5), (0.4, 0.5), (0.6, 0.5), (0.8, 0.5)],
        direction="left_to_right",
        velocity_magnitude=0.2,
        is_stationary=False,
    )
    d = traj.to_dict()
    restored = MotionTrajectory.from_dict(d)
    assert restored.direction == "left_to_right"
    assert restored.start_pos == (0.2, 0.5)
    assert len(restored.per_frame_positions) == 4
    assert not restored.is_stationary


def test_frame_metadata_serialization() -> None:
    meta = FrameMetadata(
        video_id="vid_001",
        frame_index=2,
        frame_id="vid_001_f2",
        timestamp=0.133,
    )
    d = meta.to_dict()
    restored = FrameMetadata.from_dict(d)
    assert restored.video_id == "vid_001"
    assert restored.frame_index == 2
    assert restored.frame_id == "vid_001_f2"
    assert abs(restored.timestamp - 0.133) < 1e-6


def test_valid_video_sample_and_serialization() -> None:
    # 2 frames, 1 channel, 2x2 pixels
    frame0 = [[[0.1, 0.2], [0.3, 0.4]]]
    frame1 = [[[0.5, 0.6], [0.7, 0.8]]]
    sample = VideoSample(
        video_id="vid_test_01",
        frame_tensors=[frame0, frame1],
        frame_ids=["vid_test_01_f0", "vid_test_01_f1"],
        frame_indices=[0, 1],
        label=2,
        frame_count=2,
        frame_shape=(1, 2, 2),
        split=SplitName.TRAIN,
    )
    d = sample.to_dict()
    restored = VideoSample.from_dict(d)
    assert restored.video_id == "vid_test_01"
    assert restored.frame_count == 2
    assert restored.label == 2
    assert restored.frame_shape == (1, 2, 2)
    assert restored.split == SplitName.TRAIN


def test_video_sample_rejections() -> None:
    frame0 = [[[0.1, 0.2], [0.3, 0.4]]]

    # Empty video id
    with pytest.raises(ValidationError, match="non-empty video_id"):
        VideoSample(
            video_id="",
            frame_tensors=[frame0],
            frame_ids=["f0"],
            frame_indices=[0],
            label=0,
            frame_count=1,
            frame_shape=(1, 2, 2),
        )

    # Frame count <= 0
    with pytest.raises(ValidationError, match="frame_count must be > 0"):
        VideoSample(
            video_id="v1",
            frame_tensors=[],
            frame_ids=[],
            frame_indices=[],
            label=0,
            frame_count=0,
            frame_shape=(1, 2, 2),
        )

    # Duplicate frame IDs
    with pytest.raises(ValidationError, match="Duplicate frame_ids"):
        VideoSample(
            video_id="v1",
            frame_tensors=[frame0, frame0],
            frame_ids=["dup_id", "dup_id"],
            frame_indices=[0, 1],
            label=0,
            frame_count=2,
            frame_shape=(1, 2, 2),
        )

    # Non-finite pixel
    frame_inf = [[[0.1, float("inf")], [0.3, 0.4]]]
    with pytest.raises(ValidationError, match="Non-finite pixel"):
        VideoSample(
            video_id="v1",
            frame_tensors=[frame_inf],
            frame_ids=["f0"],
            frame_indices=[0],
            label=0,
            frame_count=1,
            frame_shape=(1, 2, 2),
        )


def test_video_batch_properties() -> None:
    batch = VideoBatch(
        video_ids=["v1", "v2"],
        videos=[
            [[[[0.1]]]],  # T=1, C=1, H=1, W=1
            [[[[0.2]]]],
        ],
        labels=[0, 1],
        frame_ids=[["v1_f0"], ["v2_f0"]],
    )
    assert batch.batch_size == 2
    assert batch.num_frames == 1


def test_summary_serialization_roundtrips() -> None:
    c_sum = TemporalConsistencySummary(
        mean_adjacent_distance=0.15,
        median_adjacent_distance=0.14,
        std_adjacent_distance=0.02,
        mean_adjacent_cosine_similarity=0.98,
        max_temporal_jump=0.18,
        timestep_of_max_jump=2,
        adjacent_distances=[0.14, 0.18, 0.13],
        adjacent_cosine_similarities=[0.99, 0.97, 0.98],
    )
    c_restored = TemporalConsistencySummary.from_dict(c_sum.to_dict())
    assert abs(c_restored.mean_adjacent_distance - 0.15) < 1e-6
    assert c_restored.timestep_of_max_jump == 2

    w_sum = TemporalWeightSummary(
        weights=[0.1, 0.2, 0.3, 0.4],
        entropy=1.28,
        max_weight_timestep=3,
        max_weight=0.4,
    )
    w_restored = TemporalWeightSummary.from_dict(w_sum.to_dict())
    assert len(w_restored.weights) == 4
    assert w_restored.max_weight_timestep == 3

    r_sum = RNNDynamicsSummary(
        hidden_norms=[0.5, 0.8, 1.1],
        mean_norm=0.8,
        max_norm=1.1,
        final_norm=1.1,
    )
    r_restored = RNNDynamicsSummary.from_dict(r_sum.to_dict())
    assert len(r_restored.hidden_norms) == 3
    assert abs(r_restored.max_norm - 1.1) < 1e-6
