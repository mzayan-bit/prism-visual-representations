"""Deterministic synthetic Out-of-Distribution (OOD) dataset generation."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import Any

from prism.core.errors import ValidationError
from prism.uncertainty.contracts import OODSample
from prism.uncertainty.enums import OODCategory


@dataclass(frozen=True)
class SyntheticOODSpec:
    """Specification for synthetic OOD dataset generation."""

    dataset_name: str = "synthetic-ood-v1"
    num_samples: int = 50
    image_shape: tuple[int, int, int] = (3, 32, 32)
    categories: list[OODCategory] = field(
        default_factory=lambda: [
            OODCategory.OUT_OF_DISTRIBUTION,
            OODCategory.NEAR_OOD,
            OODCategory.CORRUPTED_IN_DISTRIBUTION,
        ]
    )
    ood_patterns: list[str] = field(
        default_factory=lambda: [
            "cross_star",
            "diamond_ring",
            "checkerboard_stripe",
            "procedural_sine_noise",
            "inverted_palette",
        ]
    )
    id_dataset_fingerprint: str = "id-synthetic-shapes-v1"
    seed: int = 42


def _compute_fingerprint(data_bytes: bytes) -> str:
    """Compute sha256 hex digest for dataset fingerprinting."""
    return hashlib.sha256(data_bytes).hexdigest()[:16]


def _create_blank_tensor(shape: tuple[int, int, int]) -> list[list[list[float]]]:
    """Create a zero-initialized (C, H, W) tensor as nested lists."""
    c, h, w = shape
    return [[[0.0 for _ in range(w)] for _ in range(h)] for _ in range(c)]


def _render_cross_star(
    tensor: list[list[list[float]]],
    center_y: int,
    center_x: int,
    size: int,
    color: tuple[float, float, float],
) -> None:
    """Render a cross/star shape (OOD semantic structure)."""
    c, h, w = len(tensor), len(tensor[0]), len(tensor[0][0])
    for y in range(h):
        for x in range(w):
            dy = abs(y - center_y)
            dx = abs(x - center_x)
            is_cross = (dy <= size and dx <= 1) or (dx <= size and dy <= 1)
            is_diag = dx == dy and dx <= size // 2
            if is_cross or is_diag:
                for ch in range(c):
                    tensor[ch][y][x] = color[ch % len(color)]


def _render_diamond_ring(
    tensor: list[list[list[float]]],
    center_y: int,
    center_x: int,
    radius: int,
    color: tuple[float, float, float],
) -> None:
    """Render a diamond ring outline (OOD geometric structure)."""
    c, h, w = len(tensor), len(tensor[0]), len(tensor[0][0])
    for y in range(h):
        for x in range(w):
            dist = abs(y - center_y) + abs(x - center_x)
            if abs(dist - radius) <= 1:
                for ch in range(c):
                    tensor[ch][y][x] = color[ch % len(color)]


def _render_checkerboard(
    tensor: list[list[list[float]]],
    grid_size: int,
    color1: tuple[float, float, float],
    color2: tuple[float, float, float],
) -> None:
    """Render high-frequency checkerboard pattern."""
    c, h, w = len(tensor), len(tensor[0]), len(tensor[0][0])
    for y in range(h):
        for x in range(w):
            cell = ((y // grid_size) + (x // grid_size)) % 2
            chosen = color1 if cell == 0 else color2
            for ch in range(c):
                tensor[ch][y][x] = chosen[ch % len(chosen)]


def _render_sine_noise(
    tensor: list[list[list[float]]],
    freq_x: float,
    freq_y: float,
    phase: float,
) -> None:
    """Render 2D sinusoidal procedural wave pattern."""
    c, h, w = len(tensor), len(tensor[0]), len(tensor[0][0])
    for y in range(h):
        for x in range(w):
            val1 = 0.5 + 0.5 * math.sin(x * freq_x + phase)
            val2 = 0.5 + 0.5 * math.cos(y * freq_y + phase * 1.5)
            val = (val1 + val2) / 2.0
            for ch in range(c):
                shift = ch * 0.3
                tensor[ch][y][x] = 0.5 + 0.5 * math.sin(val * math.pi * 2.0 + shift)


def _render_near_ood_distorted_polygon(
    tensor: list[list[list[float]]],
    center_y: int,
    center_x: int,
    radius: int,
    color: tuple[float, float, float],
) -> None:
    """Render an ambiguous morphed polygon (Near-OOD structure)."""
    c, h, w = len(tensor), len(tensor[0]), len(tensor[0][0])
    for y in range(h):
        for x in range(w):
            dy = y - center_y
            dx = x - center_x
            angle = math.atan2(dy, dx)
            mod_r = radius * (1.0 + 0.35 * math.sin(5.0 * angle))
            if math.sqrt(dy * dy + dx * dx) <= mod_r:
                for ch in range(c):
                    tensor[ch][y][x] = color[ch % len(color)]


def generate_synthetic_ood_dataset(
    spec: SyntheticOODSpec | None = None,
) -> tuple[list[OODSample], dict[str, Any]]:
    """Generate deterministic synthetic OOD dataset.

    Parameters
    ----------
    spec : SyntheticOODSpec | None
        Specification configuration for sample count, shapes, seed, etc.

    Returns
    -------
    tuple[list[OODSample], dict[str, Any]]
        List of generated OODSample instances and generation metadata.
    """
    if spec is None:
        spec = SyntheticOODSpec()

    if spec.num_samples <= 0:
        raise ValidationError("num_samples must be positive.")
    if len(spec.image_shape) != 3 or any(dim <= 0 for dim in spec.image_shape):
        raise ValidationError(f"Invalid image_shape: {spec.image_shape}.")

    rng = random.Random(spec.seed)
    c_dim, h_dim, w_dim = spec.image_shape

    samples: list[OODSample] = []
    fingerprint_accumulator = bytearray()

    colors = [
        (0.9, 0.2, 0.2),  # Reddish
        (0.2, 0.8, 0.3),  # Greenish
        (0.2, 0.4, 0.9),  # Bluish
        (0.9, 0.7, 0.1),  # Amber
        (0.8, 0.2, 0.8),  # Magenta
        (0.1, 0.8, 0.8),  # Cyan
    ]

    for idx in range(spec.num_samples):
        sample_id = f"{spec.dataset_name}_{idx:04d}"
        cat = spec.categories[idx % len(spec.categories)]
        pattern = spec.ood_patterns[idx % len(spec.ood_patterns)]

        tensor = _create_blank_tensor(spec.image_shape)
        color = colors[idx % len(colors)]
        color_alt = colors[(idx + 3) % len(colors)]

        cy = h_dim // 2 + rng.randint(-2, 2)
        cx = w_dim // 2 + rng.randint(-2, 2)

        if cat == OODCategory.OUT_OF_DISTRIBUTION:
            if pattern == "cross_star":
                _render_cross_star(tensor, cy, cx, size=h_dim // 3, color=color)
            elif pattern == "diamond_ring":
                _render_diamond_ring(tensor, cy, cx, radius=h_dim // 3, color=color)
            elif pattern == "checkerboard_stripe":
                _render_checkerboard(
                    tensor, grid_size=4, color1=color, color2=color_alt
                )
            elif pattern == "procedural_sine_noise":
                _render_sine_noise(
                    tensor,
                    freq_x=0.4 + 0.1 * (idx % 3),
                    freq_y=0.4 + 0.1 * ((idx + 1) % 3),
                    phase=idx * 0.5,
                )
            else:
                inv_color = (
                    1.0 - color[0],
                    1.0 - color[1],
                    1.0 - color[2],
                )
                _render_cross_star(tensor, cy, cx, size=h_dim // 4, color=inv_color)
            semantic_class = None

        elif cat == OODCategory.NEAR_OOD:
            _render_near_ood_distorted_polygon(
                tensor, cy, cx, radius=h_dim // 3, color=color
            )
            semantic_class = None

        elif cat == OODCategory.CORRUPTED_IN_DISTRIBUTION:
            sz = h_dim // 4
            for y in range(max(0, cy - sz), min(h_dim, cy + sz)):
                for x in range(max(0, cx - sz), min(w_dim, cx + sz)):
                    for ch in range(c_dim):
                        tensor[ch][y][x] = color[ch]
            for y in range(h_dim):
                for x in range(w_dim):
                    if rng.random() < 0.2:
                        noise_val = rng.uniform(0.0, 1.0)
                        for ch in range(c_dim):
                            tensor[ch][y][x] = noise_val
            semantic_class = f"class_{idx % 3}"

        else:
            _render_sine_noise(tensor, freq_x=0.5, freq_y=0.5, phase=float(idx))
            semantic_class = None

        # Hash sample contents for dataset fingerprinting
        for ch in range(c_dim):
            for row in tensor[ch]:
                for px in row:
                    fingerprint_accumulator.extend(
                        int(px * 1000).to_bytes(4, "big", signed=True)
                    )

        sample = OODSample(
            sample_id=sample_id,
            image=tensor,
            source_dataset_identity=spec.dataset_name,
            category=cat,
            semantic_class=semantic_class,
            metadata={
                "pattern": pattern,
                "seed": spec.seed,
                "center": (cy, cx),
            },
        )
        samples.append(sample)

    ood_fingerprint = _compute_fingerprint(bytes(fingerprint_accumulator))

    metadata = {
        "dataset_name": spec.dataset_name,
        "num_samples": len(samples),
        "image_shape": spec.image_shape,
        "seed": spec.seed,
        "id_dataset_fingerprint": spec.id_dataset_fingerprint,
        "ood_dataset_fingerprint": ood_fingerprint,
        "category_counts": {
            cat.value: sum(1 for s in samples if s.category == cat)
            for cat in spec.categories
        },
    }

    return samples, metadata
