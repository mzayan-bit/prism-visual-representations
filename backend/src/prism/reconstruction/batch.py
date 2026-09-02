"""Reconstruction batch structures, batching loaders, and data pipelines."""

from __future__ import annotations

import copy
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from prism.core.errors import ValidationError
from prism.data.materialized import MaterializedSample
from prism.models.patches import ImagePatchExtractor, PatchGeometry
from prism.reconstruction.context import MaskingContext
from prism.reconstruction.enums import ReconstructionMethod
from prism.reconstruction.mask import PatchMask, generate_patch_mask
from prism.robustness.corruptions import (
    CorruptionSpecification,
    apply_corruption,
)


class ReconstructionBatch(BaseModel):
    """Container for a batch of reconstruction samples, masks, and targets."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_ids: list[str] = Field(..., description="Unique sample identifiers in batch")
    inputs: list[Any] = Field(
        ..., description="Model inputs (corrupted images or masked patches)"
    )
    targets: list[Any] = Field(..., description="Reconstruction ground truth targets")
    masks: list[PatchMask] | None = Field(
        default=None, description="Patch masks if masked modeling was applied"
    )
    method: ReconstructionMethod = Field(
        ..., description="Reconstruction method objective"
    )
    labels: list[int | str | None] | None = Field(
        default=None,
        description="Optional ground truth class labels (never contributing to loss)",
    )
    batch_size: int = Field(gt=0, description="Number of samples in batch")


def prepare_masked_patch_batch(
    samples: list[MaterializedSample],
    geometry: PatchGeometry,
    epoch: int = 0,
    mask_ratio: float = 0.5,
    seed: int = 42,
) -> ReconstructionBatch:
    """Extract patches, generate deterministic masks, and prepare a batch.

    Parameters
    ----------
    samples : list[MaterializedSample]
        Input clean samples.
    geometry : PatchGeometry
        Patch configuration for image slicing.
    epoch : int
        Current epoch index for deterministic seed derivation.
    mask_ratio : float
        Fraction of patches to mask.
    seed : int
        Global experiment seed.

    Returns
    -------
    ReconstructionBatch
        Batch containing extracted clean patch targets and per-sample PatchMasks.
    """
    if not samples:
        raise ValidationError("Cannot prepare batch from empty sample list.")

    extractor = ImagePatchExtractor(geometry=geometry)
    raw_images = [s.data for s in samples]
    all_patches = extractor.forward(raw_images)

    sample_ids: list[str] = []
    masks: list[PatchMask] = []
    labels: list[int | str | None] = []

    for i, s in enumerate(samples):
        sample_ids.append(s.sample_id)
        ctx = MaskingContext(
            global_seed=seed,
            sample_id=s.sample_id,
            epoch=epoch,
            mask_instance_index=i,
            masking_policy="random_uniform_patches",
            mask_ratio=mask_ratio,
        )
        mask = generate_patch_mask(ctx, total_patches=geometry.total_patches)
        masks.append(mask)
        labels.append(s.target)

    return ReconstructionBatch(
        sample_ids=sample_ids,
        inputs=copy.deepcopy(all_patches),
        targets=copy.deepcopy(all_patches),
        masks=masks,
        method=ReconstructionMethod.MASKED_PATCH_RECONSTRUCTION,
        labels=labels,
        batch_size=len(samples),
    )


def prepare_denoising_batch(
    samples: list[MaterializedSample],
    corruption_spec: CorruptionSpecification,
    epoch: int = 0,
    seed: int = 42,
) -> ReconstructionBatch:
    """Corrupt clean images using Phase 15 operators and prepare a denoising batch.

    Parameters
    ----------
    samples : list[MaterializedSample]
        Input clean samples.
    corruption_spec : CorruptionSpecification
        Calibrated corruption operator (e.g. Gaussian noise, occlusion, resolution).
    epoch : int
        Current epoch index.
    seed : int
        Experiment seed.

    Returns
    -------
    ReconstructionBatch
        Batch with inputs=corrupted images, targets=clean images.
    """
    if not samples:
        raise ValidationError("Cannot prepare batch from empty sample list.")

    sample_ids: list[str] = []
    corrupted_images: list[Any] = []
    clean_images: list[Any] = []
    labels: list[int | str | None] = []

    for s in samples:
        sample_ids.append(s.sample_id)
        clean_images.append(copy.deepcopy(s.data))
        labels.append(s.target)

        # Apply deterministic corruption
        corrupted = apply_corruption(
            image=s.data,
            spec=corruption_spec,
            sample_id=s.sample_id,
        )
        corrupted_images.append(corrupted)

    return ReconstructionBatch(
        sample_ids=sample_ids,
        inputs=corrupted_images,
        targets=clean_images,
        masks=None,
        method=ReconstructionMethod.DENOISING_AUTOENCODER,
        labels=labels,
        batch_size=len(samples),
    )
