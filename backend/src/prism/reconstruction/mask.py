"""Patch mask structures, deterministic partition generators, and serialization."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from prism.core.errors import SerializationError, ValidationError
from prism.reconstruction.context import DeterministicMaskingRNG, MaskingContext


class PatchMask(BaseModel):
    """Immutable record of deterministic partition into masked and visible sets."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    total_patches: int = Field(
        gt=0, description="Total number of patches in sequence (T)"
    )
    masked_indices: list[int] = Field(
        ..., description="Sorted indices of masked patches"
    )
    visible_indices: list[int] = Field(
        ..., description="Sorted indices of visible patches"
    )
    mask_ratio: float = Field(
        ..., gt=0.0, lt=1.0, description="Target mask ratio fraction"
    )
    sample_id: str = Field(
        ..., description="Sample identifier this mask was created for"
    )
    seed_identity: str = Field(..., description="Cryptographic seed stream identifier")

    @property
    def num_masked(self) -> int:
        """Number of masked patches."""
        return len(self.masked_indices)

    @property
    def num_visible(self) -> int:
        """Number of visible patches."""
        return len(self.visible_indices)

    @model_validator(mode="after")
    def validate_partition_integrity(self) -> PatchMask:
        """Verify partition validity, disjointness, and full sequence coverage."""
        t = self.total_patches
        m_set = set(self.masked_indices)
        v_set = set(self.visible_indices)

        if len(m_set) != len(self.masked_indices):
            raise ValidationError("masked_indices contains duplicate patch indices.")
        if len(v_set) != len(self.visible_indices):
            raise ValidationError("visible_indices contains duplicate patch indices.")

        if m_set & v_set:
            raise ValidationError(f"Masked and visible sets overlap: {m_set & v_set}.")

        if len(m_set) + len(v_set) != t:
            raise ValidationError(
                f"Combined index count ({len(m_set) + len(v_set)}) "
                f"does not equal total_patches ({t})."
            )

        for idx in self.masked_indices:
            if idx < 0 or idx >= t:
                raise ValidationError(f"Masked index {idx} out of bounds [0, {t - 1}].")
        for idx in self.visible_indices:
            if idx < 0 or idx >= t:
                raise ValidationError(
                    f"Visible index {idx} out of bounds [0, {t - 1}]."
                )

        return self

    def to_dict(self) -> dict[str, Any]:
        """Serialize mask to dictionary."""
        return self.model_dump()

    def to_json(self) -> str:
        """Serialize mask to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> PatchMask:
        """Deserialize mask from JSON string."""
        try:
            return cls.model_validate(json.loads(json_str))
        except Exception as e:
            raise SerializationError(f"Failed to deserialize PatchMask: {e}") from e


def generate_patch_mask(context: MaskingContext, total_patches: int) -> PatchMask:
    """Deterministically partition a patch sequence into masked and visible sets.

    Parameters
    ----------
    context : MaskingContext
        Deterministic context capturing seed, sample, epoch, and mask ratio.
    total_patches : int
        Total number of patches (T) in the image grid.

    Returns
    -------
    PatchMask
        Immutable, verified mask structure with canonical sorted indices.
    """
    if total_patches <= 1:
        raise ValidationError(
            f"total_patches must be at least 2 for masking, got {total_patches}."
        )

    # Calculate exact masked count, clamped to [1, total_patches - 1]
    num_masked = round(total_patches * context.mask_ratio)
    num_masked = max(1, min(total_patches - 1, num_masked))

    seed_int = context.derive_seed_int("patch_mask")
    rng = DeterministicMaskingRNG(seed_int)

    all_indices = list(range(total_patches))
    shuffled = rng.shuffle_indices(all_indices)

    masked_set = set(shuffled[:num_masked])
    visible_set = set(shuffled[num_masked:])

    return PatchMask(
        total_patches=total_patches,
        masked_indices=sorted(masked_set),
        visible_indices=sorted(visible_set),
        mask_ratio=context.mask_ratio,
        sample_id=context.sample_id,
        seed_identity=f"mask_{seed_int:016x}",
    )
