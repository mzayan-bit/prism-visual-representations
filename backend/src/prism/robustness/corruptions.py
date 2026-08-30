"""Controlled image corruption specifications, operators, and dataset views."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from collections.abc import Sequence
from enum import Enum
from typing import Any, overload

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from prism.core.errors import SerializationError, ValidationError
from prism.data.materialized import MaterializedDataset, MaterializedSample


class CorruptionType(str, Enum):
    """Supported visual corruption families."""

    GAUSSIAN_NOISE = "gaussian_noise"
    BLUR = "blur"
    BRIGHTNESS = "brightness"
    CONTRAST = "contrast"
    OCCLUSION = "occlusion"
    RESOLUTION_DEGRADATION = "resolution_degradation"


# Explicit parameter maps for standardized severity levels 1..5
SEVERITY_PARAMETER_MAPS: dict[CorruptionType, dict[int, dict[str, Any]]] = {
    CorruptionType.GAUSSIAN_NOISE: {
        1: {"sigma": 0.05, "clip_min": None, "clip_max": None},
        2: {"sigma": 0.10, "clip_min": None, "clip_max": None},
        3: {"sigma": 0.18, "clip_min": None, "clip_max": None},
        4: {"sigma": 0.28, "clip_min": None, "clip_max": None},
        5: {"sigma": 0.40, "clip_min": None, "clip_max": None},
    },
    CorruptionType.BLUR: {
        1: {"kernel_size": 3, "sigma": 0.6},
        2: {"kernel_size": 3, "sigma": 1.0},
        3: {"kernel_size": 5, "sigma": 1.5},
        4: {"kernel_size": 5, "sigma": 2.2},
        5: {"kernel_size": 7, "sigma": 3.0},
    },
    CorruptionType.BRIGHTNESS: {
        1: {"delta": 0.10, "clip_min": None, "clip_max": None},
        2: {"delta": 0.20, "clip_min": None, "clip_max": None},
        3: {"delta": 0.35, "clip_min": None, "clip_max": None},
        4: {"delta": 0.50, "clip_min": None, "clip_max": None},
        5: {"delta": 0.70, "clip_min": None, "clip_max": None},
    },
    CorruptionType.CONTRAST: {
        1: {"factor": 0.80, "center": "mean"},
        2: {"factor": 0.60, "center": "mean"},
        3: {"factor": 0.40, "center": "mean"},
        4: {"factor": 0.25, "center": "mean"},
        5: {"factor": 0.10, "center": "mean"},
    },
    CorruptionType.OCCLUSION: {
        1: {"area_ratio": 0.05, "fill_value": 0.0, "location": "deterministic"},
        2: {"area_ratio": 0.10, "fill_value": 0.0, "location": "deterministic"},
        3: {"area_ratio": 0.20, "fill_value": 0.0, "location": "deterministic"},
        4: {"area_ratio": 0.35, "fill_value": 0.0, "location": "deterministic"},
        5: {"area_ratio": 0.50, "fill_value": 0.0, "location": "deterministic"},
    },
    CorruptionType.RESOLUTION_DEGRADATION: {
        1: {"downsample_factor": 2},
        2: {"downsample_factor": 3},
        3: {"downsample_factor": 4},
        4: {"downsample_factor": 6},
        5: {"downsample_factor": 8},
    },
}


class CorruptionSpecification(BaseModel):
    """Typed, deterministic specification of an input corruption."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    corruption_type: CorruptionType = Field(description="Corruption family")
    severity: int = Field(ge=1, le=5, description="Calibrated severity level (1..5)")
    seed: int | None = Field(
        default=None, description="Optional seed for stochastic corruption operators"
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Explicit mathematical parameters overriding severity defaults",
    )
    deterministic: bool = Field(
        default=True,
        description="Whether corruption executes deterministically",
    )
    version: str = Field(default="1.0", description="Corruption implementation version")

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v: int) -> int:
        if v < 1 or v > 5:
            raise ValidationError(f"Severity must be in range [1, 5], got {v}.")
        return v

    @model_validator(mode="after")
    def validate_parameters(self) -> CorruptionSpecification:
        # If parameters not provided, ensure valid severity map exists
        if self.corruption_type not in SEVERITY_PARAMETER_MAPS:
            raise ValidationError(
                f"Unsupported corruption type: {self.corruption_type}"
            )
        return self

    def get_effective_parameters(self) -> dict[str, Any]:
        """Return the combined effective parameter dictionary."""
        defaults = SEVERITY_PARAMETER_MAPS[self.corruption_type][self.severity]
        merged = copy.deepcopy(defaults)
        merged.update(self.parameters)
        return merged

    def fingerprint(self) -> str:
        """Compute deterministic SHA-256 fingerprint for this spec."""
        payload = {
            "corruption_type": self.corruption_type.value,
            "severity": self.severity,
            "seed": self.seed,
            "parameters": self.get_effective_parameters(),
            "deterministic": self.deterministic,
            "version": self.version,
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Convert specification to dictionary."""
        return self.model_dump(mode="json")

    def to_json(self, indent: int | None = None) -> str:
        """Convert specification to JSON string."""
        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=True,
            ensure_ascii=True,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CorruptionSpecification:
        """Create specification from dictionary."""
        try:
            return cls.model_validate(data)
        except Exception as exc:
            raise SerializationError(
                f"Failed to deserialize CorruptionSpecification: {exc}"
            ) from exc


# -----------------------------------------------------------------------------
# Pure Python Image Manipulation Utilities
# -----------------------------------------------------------------------------


def _validate_image_shape(image: list[list[list[float]]]) -> tuple[int, int, int]:
    """Validate 3D image tensor [C, H, W] and return dimensions."""
    if not image or not isinstance(image, list):
        raise ValidationError("Image must be a non-empty 3D list [C, H, W].")
    c = len(image)
    if c == 0 or not isinstance(image[0], list):
        raise ValidationError("Image channels must be non-empty.")
    h = len(image[0])
    if h == 0 or not isinstance(image[0][0], list):
        raise ValidationError("Image height must be non-empty.")
    w = len(image[0][0])
    if w == 0:
        raise ValidationError("Image width must be non-empty.")

    # Uniformity check
    for ch_idx, ch in enumerate(image):
        if len(ch) != h:
            raise ValidationError(
                f"Channel {ch_idx} height mismatch: expected {h}, got {len(ch)}."
            )
        for r_idx, row in enumerate(ch):
            if len(row) != w:
                raise ValidationError(
                    f"Channel {ch_idx}, row {r_idx} width mismatch: "
                    f"expected {w}, got {len(row)}."
                )
    return c, h, w


def apply_gaussian_noise(
    image: list[list[list[float]]],
    sigma: float,
    seed: int | None = None,
    clip_min: float | None = 0.0,
    clip_max: float | None = 1.0,
) -> list[list[list[float]]]:
    """Apply additive Gaussian noise to a 3D image tensor [C, H, W]."""
    c, h, w = _validate_image_shape(image)
    rng = random.Random(seed if seed is not None else 42)

    corrupted: list[list[list[float]]] = []
    for ch in range(c):
        ch_out: list[list[float]] = []
        for r in range(h):
            row_out: list[float] = []
            for col in range(w):
                val = image[ch][r][col]
                noise = rng.gauss(0.0, sigma)
                new_val = val + noise
                if clip_min is not None:
                    new_val = max(clip_min, new_val)
                if clip_max is not None:
                    new_val = min(clip_max, new_val)
                row_out.append(new_val)
            ch_out.append(row_out)
        corrupted.append(ch_out)
    return corrupted


def apply_spatial_blur(
    image: list[list[list[float]]],
    kernel_size: int = 3,
    sigma: float = 1.0,
) -> list[list[list[float]]]:
    """Apply deterministic 2D Gaussian blur per channel with replicate padding."""
    c, h, w = _validate_image_shape(image)
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValidationError(
            f"kernel_size must be positive odd integer, got {kernel_size}."
        )

    radius = kernel_size // 2
    # Compute 1D Gaussian kernel
    kernel_1d = [math.exp(-0.5 * (x / sigma) ** 2) for x in range(-radius, radius + 1)]
    kernel_sum = sum(kernel_1d)
    norm_kernel = [k / kernel_sum for k in kernel_1d]

    # Separable 2D convolution: Horizontal then Vertical
    corrupted: list[list[list[float]]] = []
    for ch in range(c):
        # 1. Horizontal blur
        h_blur: list[list[float]] = []
        for r in range(h):
            row_out: list[float] = []
            for col in range(w):
                acc = 0.0
                for k_idx, offset in enumerate(range(-radius, radius + 1)):
                    src_col = min(max(0, col + offset), w - 1)
                    acc += image[ch][r][src_col] * norm_kernel[k_idx]
                row_out.append(acc)
            h_blur.append(row_out)

        # 2. Vertical blur
        v_blur: list[list[float]] = []
        for r in range(h):
            row_out = []
            for col in range(w):
                acc = 0.0
                for k_idx, offset in enumerate(range(-radius, radius + 1)):
                    src_r = min(max(0, r + offset), h - 1)
                    acc += h_blur[src_r][col] * norm_kernel[k_idx]
                row_out.append(acc)
            v_blur.append(row_out)

        corrupted.append(v_blur)
    return corrupted


def apply_brightness_shift(
    image: list[list[list[float]]],
    delta: float,
    clip_min: float | None = None,
    clip_max: float | None = None,
) -> list[list[list[float]]]:
    """Apply additive brightness shift x' = x + delta."""
    c, h, w = _validate_image_shape(image)
    corrupted: list[list[list[float]]] = []
    for ch in range(c):
        ch_out: list[list[float]] = []
        for r in range(h):
            row_out: list[float] = []
            for col in range(w):
                new_val = image[ch][r][col] + delta
                if clip_min is not None:
                    new_val = max(clip_min, new_val)
                if clip_max is not None:
                    new_val = min(clip_max, new_val)
                row_out.append(new_val)
            ch_out.append(row_out)
        corrupted.append(ch_out)
    return corrupted


def apply_contrast_shift(
    image: list[list[list[float]]],
    factor: float,
    center: str | float = "mean",
    clip_min: float | None = None,
    clip_max: float | None = None,
) -> list[list[list[float]]]:
    """Apply contrast adjustment x' = center + factor * (x - center)."""
    c, h, w = _validate_image_shape(image)

    # Compute center
    if isinstance(center, str) and center == "mean":
        total_sum = sum(
            image[ch][r][col] for ch in range(c) for r in range(h) for col in range(w)
        )
        center_val = total_sum / float(c * h * w)
    elif isinstance(center, (int, float)):
        center_val = float(center)
    else:
        center_val = 0.5

    corrupted: list[list[list[float]]] = []
    for ch in range(c):
        ch_out: list[list[float]] = []
        for r in range(h):
            row_out: list[float] = []
            for col in range(w):
                val = image[ch][r][col]
                new_val = center_val + factor * (val - center_val)
                if clip_min is not None:
                    new_val = max(clip_min, new_val)
                if clip_max is not None:
                    new_val = min(clip_max, new_val)
                row_out.append(new_val)
            ch_out.append(row_out)
        corrupted.append(ch_out)
    return corrupted


def apply_rectangular_occlusion(
    image: list[list[list[float]]],
    area_ratio: float,
    fill_value: float = 0.0,
    sample_id: str = "sample",
    seed: int | None = None,
    severity: int = 1,
    location: str = "deterministic",
) -> list[list[list[float]]]:
    """Apply rectangular occlusion mask over a region of the image."""
    c, h, w = _validate_image_shape(image)
    if area_ratio <= 0.0 or area_ratio > 1.0:
        raise ValidationError(f"area_ratio must be in (0, 1], got {area_ratio}.")

    target_area = float(h * w) * area_ratio
    # Aspect ratio between 0.5 and 2.0
    box_h = max(1, min(h, int(math.sqrt(target_area))))
    box_w = max(1, min(w, int(target_area / box_h)))

    if location == "center":
        top = (h - box_h) // 2
        left = (w - box_w) // 2
    else:
        # Deterministic location derived from hash of (sample_id, seed, severity)
        hash_input = f"{sample_id}:{seed}:{severity}".encode()
        hash_val = int(hashlib.md5(hash_input).hexdigest(), 16)
        max_top = max(0, h - box_h)
        max_left = max(0, w - box_w)
        top = (hash_val % (max_top + 1)) if max_top > 0 else 0
        left = ((hash_val // 1000) % (max_left + 1)) if max_left > 0 else 0

    corrupted = copy.deepcopy(image)
    for ch in range(c):
        for r in range(top, min(h, top + box_h)):
            for col in range(left, min(w, left + box_w)):
                corrupted[ch][r][col] = fill_value
    return corrupted


def apply_resolution_degradation(
    image: list[list[list[float]]],
    downsample_factor: int = 2,
) -> list[list[list[float]]]:
    """Apply resolution loss via average pooling and nearest-neighbor upsampling."""
    c, h, w = _validate_image_shape(image)
    f = max(1, downsample_factor)
    if f == 1:
        return copy.deepcopy(image)

    # Downsampled grid dimensions
    down_h = max(1, h // f)
    down_w = max(1, w // f)

    # 1. Downsample (Average pooling)
    downsampled: list[list[list[float]]] = []
    for ch in range(c):
        ch_down: list[list[float]] = []
        for r_d in range(down_h):
            row_down: list[float] = []
            r_start = r_d * f
            r_end = min(h, (r_d + 1) * f)
            for c_d in range(down_w):
                c_start = c_d * f
                c_end = min(w, (c_d + 1) * f)

                patch_sum = 0.0
                patch_count = 0
                for r_i in range(r_start, r_end):
                    for c_i in range(c_start, c_end):
                        patch_sum += image[ch][r_i][c_i]
                        patch_count += 1
                row_down.append(patch_sum / float(max(1, patch_count)))
            ch_down.append(row_down)
        downsampled.append(ch_down)

    # 2. Nearest-Neighbor Upsample back to [C, H, W]
    corrupted: list[list[list[float]]] = []
    for ch in range(c):
        ch_up: list[list[float]] = []
        for r in range(h):
            r_d = min(down_h - 1, r // f)
            row_up: list[float] = []
            for col in range(w):
                c_d = min(down_w - 1, col // f)
                row_up.append(downsampled[ch][r_d][c_d])
            ch_up.append(row_up)
        corrupted.append(ch_up)

    return corrupted


def apply_corruption(
    image: list[list[list[float]]],
    spec: CorruptionSpecification,
    sample_id: str = "sample",
) -> list[list[list[float]]]:
    """Dispatch and apply corruption specification to an image tensor [C, H, W]."""
    params = spec.get_effective_parameters()
    t = spec.corruption_type

    if t == CorruptionType.GAUSSIAN_NOISE:
        return apply_gaussian_noise(
            image=image,
            sigma=float(params["sigma"]),
            seed=spec.seed,
            clip_min=params.get("clip_min"),
            clip_max=params.get("clip_max"),
        )
    if t == CorruptionType.BLUR:
        return apply_spatial_blur(
            image=image,
            kernel_size=int(params["kernel_size"]),
            sigma=float(params["sigma"]),
        )
    if t == CorruptionType.BRIGHTNESS:
        return apply_brightness_shift(
            image=image,
            delta=float(params["delta"]),
            clip_min=params.get("clip_min"),
            clip_max=params.get("clip_max"),
        )
    if t == CorruptionType.CONTRAST:
        return apply_contrast_shift(
            image=image,
            factor=float(params["factor"]),
            center=params.get("center", "mean"),
            clip_min=params.get("clip_min"),
            clip_max=params.get("clip_max"),
        )
    if t == CorruptionType.OCCLUSION:
        return apply_rectangular_occlusion(
            image=image,
            area_ratio=float(params["area_ratio"]),
            fill_value=float(params.get("fill_value", 0.0)),
            sample_id=sample_id,
            seed=spec.seed,
            severity=spec.severity,
            location=str(params.get("location", "deterministic")),
        )
    if t == CorruptionType.RESOLUTION_DEGRADATION:
        return apply_resolution_degradation(
            image=image,
            downsample_factor=int(params["downsample_factor"]),
        )

    raise ValidationError(f"Unknown corruption type: {t}")


# -----------------------------------------------------------------------------
# Corrupted Dataset View
# -----------------------------------------------------------------------------


class CorruptedDatasetView(Sequence[MaterializedSample]):
    """Deterministic, lazy corrupted view wrapping an existing MaterializedDataset."""

    def __init__(
        self,
        base_dataset: MaterializedDataset,
        corruption_spec: CorruptionSpecification,
    ) -> None:
        self.base_dataset = base_dataset
        self.corruption_spec = corruption_spec
        c_val = corruption_spec.corruption_type.value
        s_val = corruption_spec.severity
        self.dataset_id = f"{base_dataset.dataset_id}::corrupted::{c_val}::sev{s_val}"
        self.split_name = base_dataset.split_name

    def __len__(self) -> int:
        return len(self.base_dataset)

    @overload
    def __getitem__(self, index: int) -> MaterializedSample: ...

    @overload
    def __getitem__(self, index: slice) -> CorruptedDatasetView: ...

    def __getitem__(
        self, index: int | slice
    ) -> MaterializedSample | CorruptedDatasetView:
        if isinstance(index, slice):
            sliced_base = self.base_dataset[index]
            assert isinstance(sliced_base, MaterializedDataset)
            return CorruptedDatasetView(
                base_dataset=sliced_base,
                corruption_spec=self.corruption_spec,
            )

        clean_sample = self.base_dataset[index]
        assert isinstance(clean_sample, MaterializedSample)

        # Apply corruption to image payload
        corrupted_data = apply_corruption(
            image=clean_sample.data,
            spec=self.corruption_spec,
            sample_id=clean_sample.sample_id,
        )

        metadata = copy.deepcopy(clean_sample.metadata)
        metadata.update(
            {
                "original_sample_id": clean_sample.sample_id,
                "corruption_type": self.corruption_spec.corruption_type.value,
                "severity": self.corruption_spec.severity,
                "corruption_fingerprint": self.corruption_spec.fingerprint(),
            }
        )

        return MaterializedSample(
            sample_id=clean_sample.sample_id,
            source_split=clean_sample.source_split,
            source_index=clean_sample.source_index,
            data=corrupted_data,
            target=clean_sample.target,
            metadata=metadata,
        )

    def get_sample(self, sample_id: str) -> MaterializedSample:
        """Retrieve a corrupted sample by its clean sample ID."""
        clean_sample = self.base_dataset.get_sample(sample_id)
        corrupted_data = apply_corruption(
            image=clean_sample.data,
            spec=self.corruption_spec,
            sample_id=clean_sample.sample_id,
        )
        metadata = copy.deepcopy(clean_sample.metadata)
        metadata.update(
            {
                "original_sample_id": clean_sample.sample_id,
                "corruption_type": self.corruption_spec.corruption_type.value,
                "severity": self.corruption_spec.severity,
                "corruption_fingerprint": self.corruption_spec.fingerprint(),
            }
        )
        return MaterializedSample(
            sample_id=clean_sample.sample_id,
            source_split=clean_sample.source_split,
            source_index=clean_sample.source_index,
            data=corrupted_data,
            target=clean_sample.target,
            metadata=metadata,
        )

    @property
    def sample_ids(self) -> list[str]:
        """Return the ordered list of sample identifiers."""
        return self.base_dataset.sample_ids

    @property
    def targets(self) -> list[int | str | None]:
        """Return the ordered list of targets."""
        return self.base_dataset.targets
