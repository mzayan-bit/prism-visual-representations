"""Deterministic masking contexts and pseudo-random state derivation."""

from __future__ import annotations

import hashlib

from pydantic import BaseModel, ConfigDict, Field, field_validator

from prism.core.errors import ValidationError


class MaskingContext(BaseModel):
    """Immutable context capturing all factors for deterministic mask decisions.

    Guarantees that mask generation is fully reproducible across platforms,
    operating systems, and process runs without relying on global random state.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    global_seed: int = Field(..., description="Root experiment seed")
    sample_id: str = Field(..., description="Unique sample identifier")
    epoch: int = Field(default=0, ge=0, description="Training epoch index")
    mask_instance_index: int = Field(
        default=0, ge=0, description="Index of mask instance for this sample"
    )
    masking_policy: str = Field(
        default="random_uniform_patches", description="Masking strategy name"
    )
    mask_ratio: float = Field(
        ...,
        gt=0.0,
        lt=1.0,
        description="Fraction of total patches to mask (strictly in (0, 1))",
    )

    @field_validator("mask_ratio")
    @classmethod
    def validate_mask_ratio(cls, v: float) -> float:
        """Validate mask ratio is strictly in open interval (0, 1)."""
        if v <= 0.0 or v >= 1.0:
            raise ValidationError(
                f"mask_ratio must be strictly between 0.0 and 1.0, got {v}."
            )
        return float(v)

    def derive_seed_int(self, stream_tag: str = "mask") -> int:
        """Derive a 64-bit integer seed via cryptographic hashing."""
        seed_str = (
            f"{self.global_seed}:{self.sample_id}:{self.epoch}:"
            f"{self.mask_instance_index}:{self.masking_policy}:{self.mask_ratio:.6f}:"
            f"{stream_tag}"
        )
        digest = hashlib.sha256(seed_str.encode("utf-8")).hexdigest()
        return int(digest[:16], 16)


class DeterministicMaskingRNG:
    """Lightweight, self-contained pseudo-random number generator for masking."""

    def __init__(self, seed: int) -> None:
        self.state: int = seed & 0xFFFFFFFFFFFFFFFF

    def next_uint32(self) -> int:
        """Generate 32-bit unsigned integer using 64-bit LCG step."""
        # 64-bit linear congruential generator (Knuth MMIX)
        mult = 6364136223846793005
        inc = 1442695040888963407
        self.state = (self.state * mult + inc) & 0xFFFFFFFFFFFFFFFF
        return (self.state >> 32) & 0xFFFFFFFF

    def next_float(self) -> float:
        """Generate uniform float in [0.0, 1.0)."""
        return float(self.next_uint32()) / 4294967296.0

    def shuffle_indices(self, indices: list[int]) -> list[int]:
        """Deterministic in-place Fisher-Yates shuffle."""
        shuffled = list(indices)
        for i in range(len(shuffled) - 1, 0, -1):
            j = self.next_uint32() % (i + 1)
            shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
        return shuffled
