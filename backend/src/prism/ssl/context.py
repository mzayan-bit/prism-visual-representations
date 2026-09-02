"""Deterministic augmentation context and RNG seed derivation."""

from __future__ import annotations

import hashlib
import struct

from pydantic import BaseModel, ConfigDict, Field


class AugmentationContext(BaseModel):
    """Immutable deterministic context governing stochastic augmentation decisions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    global_seed: int = Field(..., description="Master experiment reproducibility seed")
    sample_id: str = Field(..., description="Unique sample identifier")
    epoch: int = Field(ge=0, description="Current training epoch")
    view_index: int = Field(
        ge=0, le=1, description="View identifier (0 for view A, 1 for view B)"
    )
    transform_index: int = Field(
        default=0, ge=0, description="Index of transform in augmentation sequence"
    )

    def derive_seed(self, extra: str = "") -> int:
        """Derive a deterministic 32-bit unsigned integer seed from context fields."""
        key = (
            f"{self.global_seed}::{self.sample_id}::epoch_{self.epoch}::"
            f"view_{self.view_index}::tf_{self.transform_index}::{extra}"
        )
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        seed_32: int = int(struct.unpack(">I", digest[:4])[0])
        return seed_32

    def next_transform(self) -> AugmentationContext:
        """Return a copy with incremented transform_index."""
        return AugmentationContext(
            global_seed=self.global_seed,
            sample_id=self.sample_id,
            epoch=self.epoch,
            view_index=self.view_index,
            transform_index=self.transform_index + 1,
        )


class DeterministicFloatRNG:
    """Deterministic pseudo-random float generator derived from an integer seed."""

    def __init__(self, seed: int) -> None:
        self.state = seed & 0xFFFFFFFF

    def next_float(self) -> float:
        """Generate a deterministic float in [0.0, 1.0)."""
        # Linear congruential generator parameters (Numerical Recipes)
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return float(self.state) / 4294967296.0

    def uniform(self, low: float, high: float) -> float:
        """Generate float in [low, high)."""
        return low + (high - low) * self.next_float()

    def randint(self, low: int, high: int) -> int:
        """Generate integer in [low, high] inclusive."""
        if high <= low:
            return low
        span = high - low + 1
        return low + int(self.next_float() * span)
