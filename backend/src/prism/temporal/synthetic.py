"""Synthetic video generator for PRISM Video & Temporal Representation Learning."""

from __future__ import annotations

import hashlib
import random
from typing import ClassVar

from prism.core.enums import SplitName
from prism.temporal.contracts import MotionTrajectory, VideoSample


class SyntheticVideoGenerator:
    """Deterministic synthetic video generator for temporal representation probing."""

    DIRECTIONS: ClassVar[list[str]] = [
        "left_to_right",  # Class 0: horizontal rightward motion
        "right_to_left",  # Class 1: horizontal leftward motion
        "vertical_down",  # Class 2: vertical downward motion
        "stationary",  # Class 3: stationary object control
    ]

    def __init__(
        self,
        num_frames: int = 4,
        channels: int = 3,
        height: int = 16,
        width: int = 16,
        seed: int = 42,
    ) -> None:
        self.num_frames = num_frames
        self.channels = channels
        self.height = height
        self.width = width
        self.seed = seed

    def _render_frame(
        self,
        center_x: float,
        center_y: float,
        radius_x: float,
        radius_y: float,
        color: tuple[float, float, float],
        shape_type: str,
        bg_noise: float = 0.05,
        rng: random.Random | None = None,
    ) -> list[list[list[float]]]:
        """Render a single 2D synthetic frame with geometric shape and background."""
        frame: list[list[list[float]]] = [
            [[0.0 for _ in range(self.width)] for _ in range(self.height)]
            for _ in range(self.channels)
        ]

        # Base subtle background gradient
        for y in range(self.height):
            y_norm = y / max(1, self.height - 1)
            for x in range(self.width):
                x_norm = x / max(1, self.width - 1)
                base_bg = 0.08 + 0.04 * (x_norm + y_norm)
                noise = rng.uniform(-bg_noise, bg_noise) if rng else 0.0
                for c in range(self.channels):
                    frame[c][y][x] = max(0.0, min(1.0, base_bg + noise))

        # Render foreground shape
        px_cx = center_x * self.width
        px_cy = center_y * self.height
        px_rx = max(1.0, radius_x * self.width)
        px_ry = max(1.0, radius_y * self.height)

        for y in range(self.height):
            for x in range(self.width):
                dx = (x - px_cx) / px_rx
                dy = (y - px_cy) / px_ry

                inside = False
                if shape_type == "circle":
                    inside = (dx * dx + dy * dy) <= 1.0
                elif shape_type == "rectangle":
                    inside = (abs(dx) <= 1.0) and (abs(dy) <= 1.0)
                elif shape_type == "triangle":
                    inside = (
                        (dy <= 1.0)
                        and (dy >= -1.0)
                        and (abs(dx) <= (1.0 - (dy + 1.0) / 2.0))
                    )

                if inside:
                    for c in range(self.channels):
                        frame[c][y][x] = max(0.0, min(1.0, color[c]))

        return frame

    def generate_dataset(
        self,
        num_samples: int = 24,
        split: SplitName = SplitName.TRAIN,
    ) -> list[VideoSample]:
        """Generate a deterministic synthetic video dataset with motion ground truth."""
        rng = random.Random(self.seed + (hash(split.value) % 10000))
        samples: list[VideoSample] = []

        palette = [
            (0.85, 0.25, 0.25),  # Crimson
            (0.25, 0.80, 0.40),  # Emerald
            (0.25, 0.50, 0.95),  # Cobalt
            (0.95, 0.75, 0.20),  # Amber
            (0.75, 0.35, 0.90),  # Purple
        ]
        shapes = ["rectangle", "circle", "triangle"]

        for i in range(num_samples):
            video_id = f"vid_{split.value}_{i:04d}"
            label = i % len(self.DIRECTIONS)
            direction = self.DIRECTIONS[label]

            shape_type = shapes[i % len(shapes)]
            color = palette[i % len(palette)]
            rad_x = 0.14 + 0.04 * rng.random()
            rad_y = 0.14 + 0.04 * rng.random()

            # Define trajectory
            is_stationary = direction == "stationary"
            per_frame_positions: list[tuple[float, float]] = []

            if direction == "left_to_right":
                start_x, end_x = 0.20, 0.80
                fixed_y = 0.30 + 0.40 * rng.random()
                start_pos = (start_x, fixed_y)
                end_pos = (end_x, fixed_y)
                for t in range(self.num_frames):
                    alpha = t / max(1, self.num_frames - 1)
                    cur_x = start_x + alpha * (end_x - start_x)
                    per_frame_positions.append((cur_x, fixed_y))
                velocity = (end_x - start_x) / max(1, self.num_frames - 1)

            elif direction == "right_to_left":
                start_x, end_x = 0.80, 0.20
                fixed_y = 0.30 + 0.40 * rng.random()
                start_pos = (start_x, fixed_y)
                end_pos = (end_x, fixed_y)
                for t in range(self.num_frames):
                    alpha = t / max(1, self.num_frames - 1)
                    cur_x = start_x + alpha * (end_x - start_x)
                    per_frame_positions.append((cur_x, fixed_y))
                velocity = abs(end_x - start_x) / max(1, self.num_frames - 1)

            elif direction == "vertical_down":
                start_y, end_y = 0.20, 0.80
                fixed_x = 0.30 + 0.40 * rng.random()
                start_pos = (fixed_x, start_y)
                end_pos = (fixed_x, end_y)
                for t in range(self.num_frames):
                    alpha = t / max(1, self.num_frames - 1)
                    cur_y = start_y + alpha * (end_y - start_y)
                    per_frame_positions.append((fixed_x, cur_y))
                velocity = (end_y - start_y) / max(1, self.num_frames - 1)

            else:  # stationary
                stat_x = 0.30 + 0.40 * rng.random()
                stat_y = 0.30 + 0.40 * rng.random()
                start_pos = (stat_x, stat_y)
                end_pos = (stat_x, stat_y)
                for _ in range(self.num_frames):
                    per_frame_positions.append((stat_x, stat_y))
                velocity = 0.0

            trajectory = MotionTrajectory(
                start_pos=start_pos,
                end_pos=end_pos,
                per_frame_positions=per_frame_positions,
                direction=direction,
                velocity_magnitude=velocity,
                is_stationary=is_stationary,
            )

            # Render frames
            frame_tensors: list[list[list[list[float]]]] = []
            frame_ids: list[str] = []
            frame_indices: list[int] = []

            for t_idx, (px, py) in enumerate(per_frame_positions):
                frame_data = self._render_frame(
                    center_x=px,
                    center_y=py,
                    radius_x=rad_x,
                    radius_y=rad_y,
                    color=color,
                    shape_type=shape_type,
                    bg_noise=0.03,
                    rng=rng,
                )
                frame_tensors.append(frame_data)
                frame_ids.append(f"{video_id}_f{t_idx}")
                frame_indices.append(t_idx)

            sample = VideoSample(
                video_id=video_id,
                frame_tensors=frame_tensors,
                frame_ids=frame_ids,
                frame_indices=frame_indices,
                label=label,
                frame_count=self.num_frames,
                frame_shape=(self.channels, self.height, self.width),
                motion_trajectory=trajectory,
                dataset_fingerprint="",
                split=split,
                metadata={
                    "shape_type": shape_type,
                    "direction": direction,
                    "color": list(color),
                },
            )
            samples.append(sample)

        # Compute dataset fingerprint
        hasher = hashlib.sha256()
        for s in samples:
            hasher.update(s.video_id.encode("utf-8"))
            hasher.update(str(s.label).encode("utf-8"))
        fingerprint = hasher.hexdigest()[:16]

        for s in samples:
            s.dataset_fingerprint = fingerprint

        return samples

    def generate_static_sequence(self, base_sample: VideoSample) -> VideoSample:
        """Create static control sequence by repeating first frame across time."""
        frame_0 = base_sample.frame_tensors[0]
        t = base_sample.frame_count
        static_frames = [
            [[list(row) for row in channel] for channel in frame_0] for _ in range(t)
        ]
        static_ids = [f"{base_sample.video_id}_static_f{i}" for i in range(t)]
        static_indices = list(range(t))

        pos_0 = (
            base_sample.motion_trajectory.per_frame_positions[0]
            if base_sample.motion_trajectory
            else (0.5, 0.5)
        )
        trajectory = MotionTrajectory(
            start_pos=pos_0,
            end_pos=pos_0,
            per_frame_positions=[pos_0 for _ in range(t)],
            direction="stationary_control",
            velocity_magnitude=0.0,
            is_stationary=True,
        )

        return VideoSample(
            video_id=f"{base_sample.video_id}_static",
            frame_tensors=static_frames,
            frame_ids=static_ids,
            frame_indices=static_indices,
            label=base_sample.label,
            frame_count=t,
            frame_shape=base_sample.frame_shape,
            motion_trajectory=trajectory,
            dataset_fingerprint=base_sample.dataset_fingerprint,
            split=base_sample.split,
            metadata={
                "is_static_control": True,
                "source_video_id": base_sample.video_id,
            },
        )
