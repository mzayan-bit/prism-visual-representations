"""Data models and sequence contracts for PRISM Video & Temporal Learning."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from prism.core.enums import SplitName
from prism.core.errors import ValidationError


@dataclass(frozen=True)
class MotionTrajectory:
    """Ground-truth motion metadata for synthetic and controlled video sequences."""

    start_pos: tuple[float, float]
    end_pos: tuple[float, float]
    per_frame_positions: list[tuple[float, float]]
    direction: str
    velocity_magnitude: float
    is_stationary: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize trajectory metadata to dictionary."""
        return {
            "start_pos": list(self.start_pos),
            "end_pos": list(self.end_pos),
            "per_frame_positions": [list(p) for p in self.per_frame_positions],
            "direction": self.direction,
            "velocity_magnitude": self.velocity_magnitude,
            "is_stationary": self.is_stationary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MotionTrajectory:
        """Instantiate MotionTrajectory from dictionary."""
        return cls(
            start_pos=(float(data["start_pos"][0]), float(data["start_pos"][1])),
            end_pos=(float(data["end_pos"][0]), float(data["end_pos"][1])),
            per_frame_positions=[
                (float(p[0]), float(p[1])) for p in data["per_frame_positions"]
            ],
            direction=str(data["direction"]),
            velocity_magnitude=float(data["velocity_magnitude"]),
            is_stationary=bool(data.get("is_stationary", False)),
        )


@dataclass(frozen=True)
class FrameMetadata:
    """Identity and timing information for an individual video frame."""

    video_id: str
    frame_index: int
    frame_id: str
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize frame metadata."""
        return {
            "video_id": self.video_id,
            "frame_index": self.frame_index,
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FrameMetadata:
        """Instantiate FrameMetadata from dictionary."""
        return cls(
            video_id=str(data["video_id"]),
            frame_index=int(data["frame_index"]),
            frame_id=str(data["frame_id"]),
            timestamp=float(data.get("timestamp", 0.0)),
        )


@dataclass
class VideoSample:
    """Canonical typed video sample container representing [T, C, H, W] frames."""

    video_id: str
    frame_tensors: list[list[list[list[float]]]]  # T x C x H x W
    frame_ids: list[str]
    frame_indices: list[int]
    label: int
    frame_count: int
    frame_shape: tuple[int, int, int]  # C, H, W
    motion_trajectory: MotionTrajectory | None = None
    dataset_fingerprint: str = ""
    split: SplitName = SplitName.TRAIN
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate temporal sequence integrity and invariants."""
        if not self.video_id:
            raise ValidationError("VideoSample requires a non-empty video_id.")

        if self.frame_count <= 0:
            raise ValidationError(
                f"VideoSample frame_count must be > 0, got {self.frame_count}."
            )

        if len(self.frame_tensors) != self.frame_count:
            raise ValidationError(
                f"Mismatch: declared frame_count {self.frame_count} != "
                f"len(frame_tensors) {len(self.frame_tensors)}."
            )

        if len(self.frame_ids) != self.frame_count:
            raise ValidationError(
                f"Mismatch: declared frame_count {self.frame_count} != "
                f"len(frame_ids) {len(self.frame_ids)}."
            )

        if len(self.frame_indices) != self.frame_count:
            raise ValidationError(
                f"Mismatch: declared frame_count {self.frame_count} != "
                f"len(frame_indices) {len(self.frame_indices)}."
            )

        if len(set(self.frame_ids)) != self.frame_count:
            raise ValidationError(
                f"Duplicate frame_ids detected in video '{self.video_id}'."
            )

        c_exp, h_exp, w_exp = self.frame_shape
        for t_idx, frame in enumerate(self.frame_tensors):
            if len(frame) != c_exp:
                raise ValidationError(
                    f"Frame {t_idx} channel dim {len(frame)} != expected {c_exp}."
                )
            for c_idx, row_list in enumerate(frame):
                if len(row_list) != h_exp:
                    raise ValidationError(
                        f"Frame {t_idx} ch {c_idx} height {len(row_list)} != {h_exp}."
                    )
                for r_idx, val_list in enumerate(row_list):
                    if len(val_list) != w_exp:
                        raise ValidationError(
                            f"Frame {t_idx} ch {c_idx} row {r_idx} len "
                            f"{len(val_list)} != {w_exp}."
                        )
                    for px in val_list:
                        if not math.isfinite(px):
                            raise ValidationError(
                                f"Non-finite pixel {px} found in video {self.video_id}."
                            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize VideoSample into a JSON-compatible dictionary."""
        split_val = (
            self.split.value if isinstance(self.split, SplitName) else str(self.split)
        )
        return {
            "video_id": self.video_id,
            "frame_tensors": self.frame_tensors,
            "frame_ids": self.frame_ids,
            "frame_indices": self.frame_indices,
            "label": self.label,
            "frame_count": self.frame_count,
            "frame_shape": list(self.frame_shape),
            "motion_trajectory": (
                self.motion_trajectory.to_dict() if self.motion_trajectory else None
            ),
            "dataset_fingerprint": self.dataset_fingerprint,
            "split": split_val,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VideoSample:
        """Construct VideoSample from dictionary representation."""
        raw_shape = data["frame_shape"]
        frame_shape = (int(raw_shape[0]), int(raw_shape[1]), int(raw_shape[2]))
        split_val = data.get("split", SplitName.TRAIN.value)
        try:
            split_enum = SplitName(split_val)
        except ValueError:
            split_enum = SplitName.TRAIN

        trajectory = (
            MotionTrajectory.from_dict(data["motion_trajectory"])
            if data.get("motion_trajectory") is not None
            else None
        )

        return cls(
            video_id=str(data["video_id"]),
            frame_tensors=data["frame_tensors"],
            frame_ids=[str(fid) for fid in data["frame_ids"]],
            frame_indices=[int(idx) for idx in data["frame_indices"]],
            label=int(data["label"]),
            frame_count=int(data["frame_count"]),
            frame_shape=frame_shape,
            motion_trajectory=trajectory,
            dataset_fingerprint=str(data.get("dataset_fingerprint", "")),
            split=split_enum,
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class VideoBatch:
    """Batched video sequence container representing [N, T, C, H, W] tensors."""

    video_ids: list[str]
    videos: list[list[list[list[list[float]]]]]  # N x T x C x H x W
    labels: list[int]
    frame_ids: list[list[str]]
    mask: list[list[float]] | None = None  # N x T (1.0 for valid, 0.0 for padded)

    @property
    def batch_size(self) -> int:
        """Number of video sequences in batch."""
        return len(self.videos)

    @property
    def num_frames(self) -> int:
        """Temporal sequence length T (assuming uniform length in batch)."""
        return len(self.videos[0]) if self.videos else 0


@dataclass(frozen=True)
class TemporalConsistencySummary:
    """Summary of adjacent-frame representation dynamics and consistency."""

    mean_adjacent_distance: float
    median_adjacent_distance: float
    std_adjacent_distance: float
    mean_adjacent_cosine_similarity: float
    max_temporal_jump: float
    timestep_of_max_jump: int
    adjacent_distances: list[float] = field(default_factory=list)
    adjacent_cosine_similarities: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize temporal consistency summary."""
        return {
            "mean_adjacent_distance": self.mean_adjacent_distance,
            "median_adjacent_distance": self.median_adjacent_distance,
            "std_adjacent_distance": self.std_adjacent_distance,
            "mean_adjacent_cosine_similarity": self.mean_adjacent_cosine_similarity,
            "max_temporal_jump": self.max_temporal_jump,
            "timestep_of_max_jump": self.timestep_of_max_jump,
            "adjacent_distances": self.adjacent_distances,
            "adjacent_cosine_similarities": self.adjacent_cosine_similarities,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TemporalConsistencySummary:
        """Deserialize temporal consistency summary."""
        return cls(
            mean_adjacent_distance=float(data["mean_adjacent_distance"]),
            median_adjacent_distance=float(data["median_adjacent_distance"]),
            std_adjacent_distance=float(data["std_adjacent_distance"]),
            mean_adjacent_cosine_similarity=float(
                data["mean_adjacent_cosine_similarity"]
            ),
            max_temporal_jump=float(data["max_temporal_jump"]),
            timestep_of_max_jump=int(data["timestep_of_max_jump"]),
            adjacent_distances=[float(x) for x in data.get("adjacent_distances", [])],
            adjacent_cosine_similarities=[
                float(x) for x in data.get("adjacent_cosine_similarities", [])
            ],
        )


@dataclass(frozen=True)
class TemporalWeightSummary:
    """Summary of learned temporal aggregation weights across sequence frames."""

    weights: list[float]
    entropy: float
    max_weight_timestep: int
    max_weight: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize temporal weight summary."""
        return {
            "weights": self.weights,
            "entropy": self.entropy,
            "max_weight_timestep": self.max_weight_timestep,
            "max_weight": self.max_weight,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TemporalWeightSummary:
        """Deserialize temporal weight summary."""
        return cls(
            weights=[float(w) for w in data["weights"]],
            entropy=float(data["entropy"]),
            max_weight_timestep=int(data["max_weight_timestep"]),
            max_weight=float(data["max_weight"]),
        )


@dataclass(frozen=True)
class RNNDynamicsSummary:
    """Summary of SimpleRNN recurrent hidden state dynamics."""

    hidden_norms: list[float]
    mean_norm: float
    max_norm: float
    final_norm: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize RNN dynamics summary."""
        return {
            "hidden_norms": self.hidden_norms,
            "mean_norm": self.mean_norm,
            "max_norm": self.max_norm,
            "final_norm": self.final_norm,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RNNDynamicsSummary:
        """Deserialize RNN dynamics summary."""
        return cls(
            hidden_norms=[float(x) for x in data["hidden_norms"]],
            mean_norm=float(data["mean_norm"]),
            max_norm=float(data["max_norm"]),
            final_norm=float(data["final_norm"]),
        )
