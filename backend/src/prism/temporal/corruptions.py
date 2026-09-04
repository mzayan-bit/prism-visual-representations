"""Deterministic temporal corruptions and perturbation pipelines for video sequences."""

from __future__ import annotations

import random
from typing import Any

from prism.core.errors import ValidationError
from prism.temporal.contracts import MotionTrajectory, VideoSample
from prism.temporal.enums import TemporalCorruptionType


def apply_frame_drop(
    sample: VideoSample,
    drop_fraction: float = 0.5,
    seed: int = 42,
) -> tuple[VideoSample, list[str]]:
    """Deterministically drop frames preserving relative temporal order."""
    t = sample.frame_count
    if t <= 1:
        return sample, []

    rng = random.Random(seed + hash(sample.video_id) % 10000)
    num_to_drop = max(1, min(t - 1, round(t * drop_fraction)))

    indices_to_drop = set(rng.sample(range(t), num_to_drop))
    dropped_frame_ids = [sample.frame_ids[i] for i in sorted(indices_to_drop)]

    kept_frames: list[list[list[list[float]]]] = []
    kept_ids: list[str] = []
    kept_indices: list[int] = []
    kept_positions: list[tuple[float, float]] = []

    for i in range(t):
        if i not in indices_to_drop:
            kept_frames.append(sample.frame_tensors[i])
            kept_ids.append(sample.frame_ids[i])
            kept_indices.append(len(kept_frames) - 1)
            if sample.motion_trajectory:
                kept_positions.append(sample.motion_trajectory.per_frame_positions[i])

    trajectory = (
        MotionTrajectory(
            start_pos=kept_positions[0],
            end_pos=kept_positions[-1],
            per_frame_positions=kept_positions,
            direction=sample.motion_trajectory.direction,
            velocity_magnitude=sample.motion_trajectory.velocity_magnitude,
            is_stationary=sample.motion_trajectory.is_stationary,
        )
        if sample.motion_trajectory and kept_positions
        else None
    )

    corrupted = VideoSample(
        video_id=f"{sample.video_id}_drop_{num_to_drop}",
        frame_tensors=kept_frames,
        frame_ids=kept_ids,
        frame_indices=kept_indices,
        label=sample.label,
        frame_count=len(kept_frames),
        frame_shape=sample.frame_shape,
        motion_trajectory=trajectory,
        dataset_fingerprint=sample.dataset_fingerprint,
        split=sample.split,
        metadata={
            "corruption": "frame_drop",
            "dropped_count": num_to_drop,
            "dropped_ids": dropped_frame_ids,
            "source_video_id": sample.video_id,
        },
    )
    return corrupted, dropped_frame_ids


def apply_frame_duplication(
    sample: VideoSample,
    dup_index: int = 0,
    dup_count: int = 1,
) -> VideoSample:
    """Duplicate selected frame to test sensitivity to temporal stutter."""
    t = sample.frame_count
    target_idx = max(0, min(t - 1, dup_index))

    new_frames: list[list[list[list[float]]]] = []
    new_ids: list[str] = []
    new_indices: list[int] = []
    new_positions: list[tuple[float, float]] = []

    for i in range(t):
        new_frames.append(sample.frame_tensors[i])
        new_ids.append(sample.frame_ids[i])
        new_indices.append(len(new_frames) - 1)
        if sample.motion_trajectory:
            new_positions.append(sample.motion_trajectory.per_frame_positions[i])

        if i == target_idx:
            for d in range(dup_count):
                new_frames.append(
                    [[list(row) for row in ch] for ch in sample.frame_tensors[i]]
                )
                new_ids.append(f"{sample.frame_ids[i]}_dup{d + 1}")
                new_indices.append(len(new_frames) - 1)
                if sample.motion_trajectory:
                    new_positions.append(
                        sample.motion_trajectory.per_frame_positions[i]
                    )

    trajectory = (
        MotionTrajectory(
            start_pos=new_positions[0],
            end_pos=new_positions[-1],
            per_frame_positions=new_positions,
            direction=sample.motion_trajectory.direction,
            velocity_magnitude=sample.motion_trajectory.velocity_magnitude,
            is_stationary=sample.motion_trajectory.is_stationary,
        )
        if sample.motion_trajectory and new_positions
        else None
    )

    return VideoSample(
        video_id=f"{sample.video_id}_dup_{target_idx}",
        frame_tensors=new_frames,
        frame_ids=new_ids,
        frame_indices=new_indices,
        label=sample.label,
        frame_count=len(new_frames),
        frame_shape=sample.frame_shape,
        motion_trajectory=trajectory,
        dataset_fingerprint=sample.dataset_fingerprint,
        split=sample.split,
        metadata={
            "corruption": "frame_duplication",
            "dup_target_index": target_idx,
            "dup_count": dup_count,
            "source_video_id": sample.video_id,
        },
    )


def apply_frame_shuffle(
    sample: VideoSample,
    seed: int = 42,
) -> tuple[VideoSample, list[int]]:
    """Deterministically permute frame order for order sensitivity tests."""
    t = sample.frame_count
    if t <= 1:
        return sample, list(range(t))

    rng = random.Random(seed + hash(sample.video_id) % 10000)
    perm = list(range(t))
    rng.shuffle(perm)

    shuffled_frames = [sample.frame_tensors[p] for p in perm]
    shuffled_ids = [sample.frame_ids[p] for p in perm]
    shuffled_indices = list(range(t))
    shuffled_positions = (
        [sample.motion_trajectory.per_frame_positions[p] for p in perm]
        if sample.motion_trajectory
        else []
    )

    trajectory = (
        MotionTrajectory(
            start_pos=shuffled_positions[0],
            end_pos=shuffled_positions[-1],
            per_frame_positions=shuffled_positions,
            direction="shuffled",
            velocity_magnitude=sample.motion_trajectory.velocity_magnitude,
            is_stationary=False,
        )
        if sample.motion_trajectory and shuffled_positions
        else None
    )

    corrupted = VideoSample(
        video_id=f"{sample.video_id}_shuf",
        frame_tensors=shuffled_frames,
        frame_ids=shuffled_ids,
        frame_indices=shuffled_indices,
        label=sample.label,
        frame_count=t,
        frame_shape=sample.frame_shape,
        motion_trajectory=trajectory,
        dataset_fingerprint=sample.dataset_fingerprint,
        split=sample.split,
        metadata={
            "corruption": "frame_shuffle",
            "permutation": perm,
            "source_video_id": sample.video_id,
        },
    )
    return corrupted, perm


def apply_temporal_subsampling(
    sample: VideoSample,
    stride: int = 2,
) -> VideoSample:
    """Subsample sequence with fixed stride to simulate reduced frame rate."""
    if stride < 1:
        raise ValidationError(f"Stride must be >= 1, got {stride}.")

    t = sample.frame_count
    indices = list(range(0, t, stride))

    sub_frames = [sample.frame_tensors[i] for i in indices]
    sub_ids = [sample.frame_ids[i] for i in indices]
    sub_indices = list(range(len(indices)))
    sub_positions = (
        [sample.motion_trajectory.per_frame_positions[i] for i in indices]
        if sample.motion_trajectory
        else []
    )

    trajectory = (
        MotionTrajectory(
            start_pos=sub_positions[0],
            end_pos=sub_positions[-1],
            per_frame_positions=sub_positions,
            direction=sample.motion_trajectory.direction,
            velocity_magnitude=(sample.motion_trajectory.velocity_magnitude * stride),
            is_stationary=sample.motion_trajectory.is_stationary,
        )
        if sample.motion_trajectory and sub_positions
        else None
    )

    return VideoSample(
        video_id=f"{sample.video_id}_stride_{stride}",
        frame_tensors=sub_frames,
        frame_ids=sub_ids,
        frame_indices=sub_indices,
        label=sample.label,
        frame_count=len(sub_frames),
        frame_shape=sample.frame_shape,
        motion_trajectory=trajectory,
        dataset_fingerprint=sample.dataset_fingerprint,
        split=sample.split,
        metadata={
            "corruption": "temporal_subsampling",
            "stride": stride,
            "source_video_id": sample.video_id,
        },
    )


def apply_spatial_composite(
    sample: VideoSample,
    noise_level: float = 0.15,
    seed: int = 42,
) -> VideoSample:
    """Apply uniform Gaussian-like spatial noise across all sequence frames."""
    rng = random.Random(seed + hash(sample.video_id) % 10000)
    c, h, w = sample.frame_shape
    t = sample.frame_count

    corrupted_frames: list[list[list[list[float]]]] = []
    for t_i in range(t):
        frame_i = sample.frame_tensors[t_i]
        c_frame: list[list[list[float]]] = []
        for ch in range(c):
            row_list: list[list[float]] = []
            for y in range(h):
                val_list: list[float] = []
                for x in range(w):
                    noise = rng.gauss(0.0, noise_level)
                    val = max(0.0, min(1.0, frame_i[ch][y][x] + noise))
                    val_list.append(val)
                row_list.append(val_list)
            c_frame.append(row_list)
        corrupted_frames.append(c_frame)

    return VideoSample(
        video_id=f"{sample.video_id}_spat_noise",
        frame_tensors=corrupted_frames,
        frame_ids=[f"{fid}_noisy" for fid in sample.frame_ids],
        frame_indices=list(sample.frame_indices),
        label=sample.label,
        frame_count=t,
        frame_shape=sample.frame_shape,
        motion_trajectory=sample.motion_trajectory,
        dataset_fingerprint=sample.dataset_fingerprint,
        split=sample.split,
        metadata={
            "corruption": "spatial_composite",
            "noise_level": noise_level,
            "source_video_id": sample.video_id,
        },
    )


def apply_temporal_corruption(
    sample: VideoSample,
    corruption_type: TemporalCorruptionType,
    **kwargs: Any,
) -> tuple[VideoSample, dict[str, Any]]:
    """Dispatcher applying requested temporal perturbation with lineage metadata."""
    if corruption_type == TemporalCorruptionType.FRAME_DROP:
        frac = float(kwargs.get("drop_fraction", 0.5))
        seed = int(kwargs.get("seed", 42))
        corrupted, dropped = apply_frame_drop(sample, drop_fraction=frac, seed=seed)
        return corrupted, {"dropped_frame_ids": dropped, "drop_fraction": frac}

    if corruption_type == TemporalCorruptionType.FRAME_DUPLICATION:
        idx = int(kwargs.get("dup_index", 0))
        count = int(kwargs.get("dup_count", 1))
        corrupted = apply_frame_duplication(sample, dup_index=idx, dup_count=count)
        return corrupted, {"dup_target_index": idx, "dup_count": count}

    if corruption_type == TemporalCorruptionType.FRAME_SHUFFLE:
        seed = int(kwargs.get("seed", 42))
        corrupted, perm = apply_frame_shuffle(sample, seed=seed)
        return corrupted, {"permutation": perm}

    if corruption_type == TemporalCorruptionType.TEMPORAL_SUBSAMPLING:
        stride = int(kwargs.get("stride", 2))
        corrupted = apply_temporal_subsampling(sample, stride=stride)
        return corrupted, {"stride": stride}

    if corruption_type == TemporalCorruptionType.SPATIAL_COMPOSITE:
        noise = float(kwargs.get("noise_level", 0.15))
        seed = int(kwargs.get("seed", 42))
        corrupted = apply_spatial_composite(sample, noise_level=noise, seed=seed)
        return corrupted, {"noise_level": noise}

    raise ValidationError(f"Unsupported temporal corruption type: {corruption_type}.")
