"""Deterministic visual augmentation transformations for contrastive learning."""

from __future__ import annotations

import copy
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from prism.ssl.context import AugmentationContext, DeterministicFloatRNG


class AugmentationTrace(BaseModel):
    """Immutable audit record of a single applied transformation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    transform_name: str = Field(..., description="Name of the transform")
    applied: bool = Field(..., description="Whether transform was applied")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Concrete parameters used during transform"
    )
    derived_seed: int = Field(
        ..., description="Deterministic seed derived for this decision"
    )


class BaseAugmentation:
    """Abstract base class for deterministic 3D image tensor transforms."""

    def __call__(
        self, image: list[list[list[float]]], context: AugmentationContext
    ) -> tuple[list[list[list[float]]], AugmentationTrace]:
        raise NotImplementedError


class RandomHorizontalFlip(BaseAugmentation):
    """Flip 3D image horizontally (across width) with deterministic probability p."""

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(
        self, image: list[list[list[float]]], context: AugmentationContext
    ) -> tuple[list[list[list[float]]], AugmentationTrace]:
        seed = context.derive_seed("RandomHorizontalFlip")
        rng = DeterministicFloatRNG(seed)
        should_flip = rng.next_float() < self.p

        if not should_flip:
            trace = AugmentationTrace(
                transform_name="RandomHorizontalFlip",
                applied=False,
                parameters={"p": self.p},
                derived_seed=seed,
            )
            return copy.deepcopy(image), trace

        c = len(image)
        h = len(image[0])
        w = len(image[0][0])

        out: list[list[list[float]]] = [
            [[image[ch][row][w - 1 - col] for col in range(w)] for row in range(h)]
            for ch in range(c)
        ]

        trace = AugmentationTrace(
            transform_name="RandomHorizontalFlip",
            applied=True,
            parameters={"p": self.p},
            derived_seed=seed,
        )
        return out, trace


class RandomCropWithPadding(BaseAugmentation):
    """Pad spatial dimensions and randomly crop back to original height and width."""

    def __init__(self, padding: int = 2, pad_mode: str = "reflect") -> None:
        self.padding = padding
        self.pad_mode = pad_mode

    def __call__(
        self, image: list[list[list[float]]], context: AugmentationContext
    ) -> tuple[list[list[list[float]]], AugmentationTrace]:
        seed = context.derive_seed("RandomCropWithPadding")
        rng = DeterministicFloatRNG(seed)

        c = len(image)
        h = len(image[0])
        w = len(image[0][0])
        p = self.padding

        if p <= 0:
            trace = AugmentationTrace(
                transform_name="RandomCropWithPadding",
                applied=False,
                parameters={"padding": p},
                derived_seed=seed,
            )
            return copy.deepcopy(image), trace

        pad_h = h + 2 * p
        pad_w = w + 2 * p

        # Build padded image
        padded: list[list[list[float]]] = [
            [[0.0 for _ in range(pad_w)] for _ in range(pad_h)] for _ in range(c)
        ]

        for ch in range(c):
            for r in range(pad_h):
                for col in range(pad_w):
                    orig_r = r - p
                    orig_c = col - p

                    if self.pad_mode == "reflect":
                        if orig_r < 0:
                            orig_r = min(h - 1, -orig_r)
                        elif orig_r >= h:
                            orig_r = max(0, 2 * h - 2 - orig_r)
                        if orig_c < 0:
                            orig_c = min(w - 1, -orig_c)
                        elif orig_c >= w:
                            orig_c = max(0, 2 * w - 2 - orig_c)
                        padded[ch][r][col] = image[ch][orig_r][orig_c]
                    else:  # zero padding
                        if 0 <= orig_r < h and 0 <= orig_c < w:
                            padded[ch][r][col] = image[ch][orig_r][orig_c]
                        else:
                            padded[ch][r][col] = 0.0

        # Choose crop start row and col
        crop_r = rng.randint(0, 2 * p)
        crop_c = rng.randint(0, 2 * p)

        out: list[list[list[float]]] = [
            [
                [padded[ch][crop_r + r][crop_c + col] for col in range(w)]
                for r in range(h)
            ]
            for ch in range(c)
        ]

        trace = AugmentationTrace(
            transform_name="RandomCropWithPadding",
            applied=True,
            parameters={"padding": p, "crop_r": crop_r, "crop_c": crop_c},
            derived_seed=seed,
        )
        return out, trace


class ColorJitter(BaseAugmentation):
    """Adjust brightness and contrast deterministically."""

    def __init__(self, brightness: float = 0.2, contrast: float = 0.2) -> None:
        self.brightness = brightness
        self.contrast = contrast

    def __call__(
        self, image: list[list[list[float]]], context: AugmentationContext
    ) -> tuple[list[list[list[float]]], AugmentationTrace]:
        seed = context.derive_seed("ColorJitter")
        rng = DeterministicFloatRNG(seed)

        b_factor = (
            rng.uniform(1.0 - self.brightness, 1.0 + self.brightness)
            if self.brightness > 0
            else 1.0
        )
        c_factor = (
            rng.uniform(1.0 - self.contrast, 1.0 + self.contrast)
            if self.contrast > 0
            else 1.0
        )

        c = len(image)
        h = len(image[0])
        w = len(image[0][0])

        # Compute global mean for contrast blend
        total = 0.0
        count = c * h * w
        for ch in range(c):
            for r in range(h):
                for col in range(w):
                    total += image[ch][r][col]
        mean_val = total / float(count) if count > 0 else 0.5

        out: list[list[list[float]]] = [
            [[0.0 for _ in range(w)] for _ in range(h)] for _ in range(c)
        ]

        for ch in range(c):
            for r in range(h):
                for col in range(w):
                    val = image[ch][r][col] * b_factor
                    val = mean_val + c_factor * (val - mean_val)
                    # Clip to standard unit interval
                    out[ch][r][col] = min(1.0, max(0.0, val))

        trace = AugmentationTrace(
            transform_name="ColorJitter",
            applied=True,
            parameters={
                "brightness_factor": round(b_factor, 4),
                "contrast_factor": round(c_factor, 4),
            },
            derived_seed=seed,
        )
        return out, trace


class Grayscale(BaseAugmentation):
    """Convert RGB image to single-channel luminance replicated across 3 channels."""

    def __init__(self, p: float = 0.2) -> None:
        self.p = p

    def __call__(
        self, image: list[list[list[float]]], context: AugmentationContext
    ) -> tuple[list[list[list[float]]], AugmentationTrace]:
        seed = context.derive_seed("Grayscale")
        rng = DeterministicFloatRNG(seed)
        should_apply = rng.next_float() < self.p

        if not should_apply or len(image) < 3:
            trace = AugmentationTrace(
                transform_name="Grayscale",
                applied=False,
                parameters={"p": self.p},
                derived_seed=seed,
            )
            return copy.deepcopy(image), trace

        h = len(image[0])
        w = len(image[0][0])

        # Standard ITU-R BT.601 luminance weights
        r_w, g_w, b_w = 0.299, 0.587, 0.114

        gray_plane = [
            [
                r_w * image[0][r][col] + g_w * image[1][r][col] + b_w * image[2][r][col]
                for col in range(w)
            ]
            for r in range(h)
        ]

        out = [copy.deepcopy(gray_plane) for _ in range(len(image))]

        trace = AugmentationTrace(
            transform_name="Grayscale",
            applied=True,
            parameters={"p": self.p},
            derived_seed=seed,
        )
        return out, trace


class AugmentationPolicy:
    """Ordered pipeline of deterministic visual transformations."""

    def __init__(self, transforms: list[BaseAugmentation] | None = None) -> None:
        if transforms is None:
            self.transforms = [
                RandomHorizontalFlip(p=0.5),
                RandomCropWithPadding(padding=2, pad_mode="reflect"),
                ColorJitter(brightness=0.2, contrast=0.2),
            ]
        else:
            self.transforms = list(transforms)

    def apply(
        self, image: list[list[list[float]]], context: AugmentationContext
    ) -> tuple[list[list[list[float]]], list[AugmentationTrace]]:
        """Apply pipeline sequentially, returning augmented image and audit traces."""
        current_img = copy.deepcopy(image)
        traces: list[AugmentationTrace] = []
        cur_ctx = context

        for tf in self.transforms:
            current_img, trace = tf(current_img, cur_ctx)
            traces.append(trace)
            cur_ctx = cur_ctx.next_transform()

        return current_img, traces
