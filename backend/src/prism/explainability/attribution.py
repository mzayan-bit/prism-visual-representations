"""Core contracts, specifications, statistics, and result models."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from prism.core.errors import SerializationError, ValidationError


class AttributionMethod(str, Enum):
    """Supported attribution and explainability methods."""

    INPUT_GRADIENT = "input_gradient"
    GRADIENT_X_INPUT = "gradient_x_input"
    OCCLUSION_SENSITIVITY = "occlusion_sensitivity"
    GRAD_CAM = "grad_cam"
    VIT_ATTENTION = "vit_attention"


class TargetClassMode(str, Enum):
    """Target class selection mode for class-conditional attribution."""

    PREDICTED_CLASS = "predicted_class"
    TRUE_CLASS = "true_class"
    EXPLICIT_CLASS = "explicit_class"


class ChannelReductionPolicy(str, Enum):
    """Strategy for reducing multi-channel gradients to a 2D spatial heatmap."""

    ABS_MAX = "abs_max"
    ABS_MEAN = "abs_mean"
    L2_CHANNEL_NORM = "l2_channel_norm"


class AttributionNormalizationPolicy(str, Enum):
    """Normalization strategy applied to 2D spatial attribution maps."""

    NONE = "none"
    MIN_MAX_ABSOLUTE = "min_max_absolute"
    ABS_SUM_NORMALIZE = "abs_sum_normalize"
    SIGNED_MIN_MAX = "signed_min_max"


class ViTAttentionHeadPolicy(str, Enum):
    """Multi-head attention aggregation policy for Vision Transformers."""

    MEAN_HEADS = "mean_heads"
    SPECIFIC_HEAD = "specific_head"


class OcclusionFillPolicy(str, Enum):
    """Fill strategy for occluded rectangular spatial windows."""

    ZERO = "zero"
    IMAGE_MEAN = "image_mean"


# -----------------------------------------------------------------------------
# Attribution Specification
# -----------------------------------------------------------------------------


class AttributionSpecification(BaseModel):
    """Declarative specification defining parameters for attribution generation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    method: AttributionMethod = Field(
        description="Visual attribution method to execute"
    )
    target_mode: TargetClassMode = Field(
        default=TargetClassMode.PREDICTED_CLASS,
        description="Semantic target class selection mode",
    )
    explicit_target_class: int | None = Field(
        default=None,
        description="Explicit class index required when target_mode is EXPLICIT_CLASS",
    )
    layer_name: str | None = Field(
        default=None,
        description="Target spatial layer identifier (required for Grad-CAM)",
    )
    channel_reduction: ChannelReductionPolicy = Field(
        default=ChannelReductionPolicy.ABS_MAX,
        description="Multi-channel gradient reduction policy",
    )
    normalization: AttributionNormalizationPolicy = Field(
        default=AttributionNormalizationPolicy.MIN_MAX_ABSOLUTE,
        description="Heatmap normalization policy",
    )
    occlusion_window_size: tuple[int, int] = Field(
        default=(2, 2),
        description="Height and width (h_w, w_w) of occlusion sliding window",
    )
    occlusion_stride: tuple[int, int] = Field(
        default=(1, 1),
        description="Vertical and horizontal stride (s_h, s_w) for occlusion",
    )
    occlusion_fill: OcclusionFillPolicy = Field(
        default=OcclusionFillPolicy.ZERO,
        description="Fill value policy for occluded regions",
    )
    occlusion_max_windows: int = Field(
        default=256,
        ge=1,
        description="Safeguard threshold on maximum allowed occlusion windows",
    )
    vit_head_policy: ViTAttentionHeadPolicy = Field(
        default=ViTAttentionHeadPolicy.MEAN_HEADS,
        description="Attention head aggregation policy for ViT",
    )
    vit_head_index: int | None = Field(
        default=None,
        description="Specific head index when vit_head_policy is SPECIFIC_HEAD",
    )
    vit_layer_index: int = Field(
        default=-1,
        description="Transformer encoder block index (-1 indicates last layer)",
    )
    seed: int | None = Field(
        default=42,
        description="Deterministic seed for stochastic sub-routines where used",
    )
    version: str = Field(
        default="1.0",
        description="Attribution specification schema version",
    )

    @model_validator(mode="after")
    def validate_spec_consistency(self) -> AttributionSpecification:
        """Validate inter-field consistency rules."""
        if (
            self.target_mode == TargetClassMode.EXPLICIT_CLASS
            and self.explicit_target_class is None
        ):
            raise ValidationError(
                "explicit_target_class must be provided when target_mode is "
                "EXPLICIT_CLASS."
            )

        if (
            self.method == AttributionMethod.VIT_ATTENTION
            and self.vit_head_policy == ViTAttentionHeadPolicy.SPECIFIC_HEAD
            and self.vit_head_index is None
        ):
            raise ValidationError(
                "vit_head_index must be specified when vit_head_policy is "
                "SPECIFIC_HEAD."
            )

        if (
            self.occlusion_window_size[0] <= 0
            or self.occlusion_window_size[1] <= 0
            or self.occlusion_stride[0] <= 0
            or self.occlusion_stride[1] <= 0
        ):
            raise ValidationError(
                f"Occlusion window dimensions and strides must be positive, "
                f"got window={self.occlusion_window_size}, "
                f"stride={self.occlusion_stride}."
            )

        return self

    def fingerprint(self) -> str:
        """Compute deterministic cryptographic SHA-256 fingerprint for this spec."""
        payload = {
            "method": self.method.value,
            "target_mode": self.target_mode.value,
            "explicit_target_class": self.explicit_target_class,
            "layer_name": self.layer_name,
            "channel_reduction": self.channel_reduction.value,
            "normalization": self.normalization.value,
            "occlusion_window_size": list(self.occlusion_window_size),
            "occlusion_stride": list(self.occlusion_stride),
            "occlusion_fill": self.occlusion_fill.value,
            "occlusion_max_windows": self.occlusion_max_windows,
            "vit_head_policy": self.vit_head_policy.value,
            "vit_head_index": self.vit_head_index,
            "vit_layer_index": self.vit_layer_index,
            "seed": self.seed,
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
    def from_dict(cls, data: dict[str, Any]) -> AttributionSpecification:
        """Construct specification from dictionary."""
        try:
            return cls.model_validate(data)
        except Exception as exc:
            raise SerializationError(
                f"Failed to deserialize AttributionSpecification: {exc}"
            ) from exc


# -----------------------------------------------------------------------------
# Spatial Map Statistics
# -----------------------------------------------------------------------------


class AttributionStatistics(BaseModel):
    """Descriptive mathematical and spatial statistics for an attribution heatmap."""

    model_config = ConfigDict(extra="forbid")

    min_value: float = Field(description="Minimum value in 2D attribution map")
    max_value: float = Field(description="Maximum value in 2D attribution map")
    mean_value: float = Field(description="Mean value across spatial positions")
    std_value: float = Field(description="Standard deviation across spatial positions")
    total_absolute_mass: float = Field(
        description="Total sum of absolute attribution values across all pixels"
    )
    top_10_percent_mass_fraction: float = Field(
        description="Fraction of total absolute mass concentrated in top 10% pixels"
    )
    top_25_percent_mass_fraction: float = Field(
        description="Fraction of total absolute mass concentrated in top 25% pixels"
    )
    spatial_entropy: float = Field(
        description="Shannon entropy H(p) of normalized non-negative mass distribution"
    )
    concentration_score: float = Field(
        description="Normalized concentration metric in [0, 1] (1 - H(p)/ln(P))"
    )
    center_of_mass_row: float = Field(
        description="Vertical centroid (row) weighted by absolute attribution"
    )
    center_of_mass_col: float = Field(
        description="Horizontal centroid (col) weighted by absolute attribution"
    )
    is_finite: bool = Field(
        default=True,
        description="Whether all values in the heatmap are finite real numbers",
    )


# -----------------------------------------------------------------------------
# Common Attribution Result Contract
# -----------------------------------------------------------------------------


class AttributionResult(BaseModel):
    """Standardized result envelope for an attribution calculation on a sample."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(description="Unique identifier of evaluated sample")
    model_id: str = Field(description="Model identifier that produced attribution")
    architecture: str = Field(description="Model architecture family name")
    method: AttributionMethod = Field(description="Attribution method executed")
    specification: AttributionSpecification = Field(
        description="Specification parameters used to generate this attribution"
    )
    target_class: int = Field(description="Class index target being explained")
    predicted_class: int = Field(
        description="Model's top-1 predicted class index on this sample"
    )
    true_class: int | None = Field(
        default=None, description="Ground truth class index if labeled"
    )
    target_score: float | None = Field(
        default=None, description="Raw model logit score for target class"
    )
    predicted_score: float | None = Field(
        default=None, description="Raw model logit score for top-1 predicted class"
    )
    source_image_shape: list[int] = Field(
        description="Dimensions of input image [C, H, W]"
    )
    attribution_shape: list[int] = Field(
        description="Dimensions of 2D attribution heatmap [H, W]"
    )
    raw_attribution_map: list[list[float]] = Field(
        description="Unnormalized 2D spatial attribution values [H, W]"
    )
    normalized_attribution_map: list[list[float]] = Field(
        description="Normalized 2D spatial attribution values according to spec [H, W]"
    )
    statistics: AttributionStatistics = Field(
        description="Computed spatial and mass statistics"
    )
    positive_mass: float = Field(
        description="Sum of all strictly positive attribution elements"
    )
    negative_mass: float = Field(
        description="Sum of all strictly negative attribution elements"
    )
    absolute_mass: float = Field(
        description="Sum of absolute attribution elements across all pixels"
    )
    method_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Method-specific auxiliary metadata (e.g. layer, heads, windows)",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Descriptive warnings regarding signal strength or sparsity",
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return self.model_dump(mode="json")

    def to_json(self, indent: int | None = None) -> str:
        """Convert result to JSON string."""
        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=True,
            ensure_ascii=True,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AttributionResult:
        """Construct result from dictionary."""
        try:
            return cls.model_validate(data)
        except Exception as exc:
            raise SerializationError(
                f"Failed to deserialize AttributionResult: {exc}"
            ) from exc


# -----------------------------------------------------------------------------
# Channel Reduction & Normalization Mathematical Operations
# -----------------------------------------------------------------------------


def reduce_channels(
    tensor_3d: list[list[list[float]]],
    policy: ChannelReductionPolicy = ChannelReductionPolicy.ABS_MAX,
) -> list[list[float]]:
    """Reduce multi-channel 3D tensor [C, H, W] to 2D spatial map [H, W].

    Args:
        tensor_3d: 3D input tensor with dimensions [C, H, W].
        policy: Channel reduction method (ABS_MAX, ABS_MEAN, L2_CHANNEL_NORM).

    Returns:
        2D spatial float matrix [H, W].
    """
    if not tensor_3d or not tensor_3d[0] or not tensor_3d[0][0]:
        raise ValidationError("Input tensor_3d must be non-empty [C, H, W].")

    c = len(tensor_3d)
    h = len(tensor_3d[0])
    w = len(tensor_3d[0][0])

    out: list[list[float]] = []
    for r in range(h):
        row: list[float] = []
        for col in range(w):
            ch_vals = [tensor_3d[ch_idx][r][col] for ch_idx in range(c)]
            if policy == ChannelReductionPolicy.ABS_MAX:
                val = max(abs(v) for v in ch_vals)
            elif policy == ChannelReductionPolicy.ABS_MEAN:
                val = sum(abs(v) for v in ch_vals) / float(c)
            elif policy == ChannelReductionPolicy.L2_CHANNEL_NORM:
                val = math.sqrt(sum(v * v for v in ch_vals))
            else:
                val = max(abs(v) for v in ch_vals)
            row.append(val)
        out.append(row)
    return out


def normalize_attribution_map(
    map_2d: list[list[float]],
    policy: AttributionNormalizationPolicy = (
        AttributionNormalizationPolicy.MIN_MAX_ABSOLUTE
    ),
    eps: float = 1e-12,
) -> list[list[float]]:
    """Normalize 2D spatial attribution map according to declared policy.

    Args:
        map_2d: 2D spatial matrix [H, W].
        policy: AttributionNormalizationPolicy to apply.
        eps: Small numerical stabilizer to prevent division by zero.

    Returns:
        Normalized 2D spatial matrix [H, W].
    """
    if not map_2d or not map_2d[0]:
        raise ValidationError("Input map_2d must be non-empty [H, W].")

    h = len(map_2d)
    w = len(map_2d[0])

    if policy == AttributionNormalizationPolicy.NONE:
        return copy.deepcopy(map_2d)

    flat = [map_2d[r][c] for r in range(h) for c in range(w)]
    min_val = min(flat)
    max_val = max(flat)
    abs_max = max(abs(min_val), abs(max_val))
    abs_sum = sum(abs(v) for v in flat)

    normalized: list[list[float]] = []

    if policy == AttributionNormalizationPolicy.MIN_MAX_ABSOLUTE:
        denom = abs_max if abs_max > eps else 1.0
        for r in range(h):
            row = [abs(map_2d[r][c]) / denom for c in range(w)]
            normalized.append(row)

    elif policy == AttributionNormalizationPolicy.ABS_SUM_NORMALIZE:
        denom = abs_sum if abs_sum > eps else 1.0
        for r in range(h):
            row = [abs(map_2d[r][c]) / denom for c in range(w)]
            normalized.append(row)

    elif policy == AttributionNormalizationPolicy.SIGNED_MIN_MAX:
        denom = abs_max if abs_max > eps else 1.0
        for r in range(h):
            row = [map_2d[r][c] / denom for c in range(w)]
            normalized.append(row)

    else:
        return copy.deepcopy(map_2d)

    return normalized


def compute_attribution_statistics(
    map_2d: list[list[float]],
    eps: float = 1e-12,
) -> AttributionStatistics:
    """Compute spatial, statistical, and information concentration metrics for a 2D map.

    Args:
        map_2d: 2D spatial attribution matrix [H, W].
        eps: Small stabilizer for entropy and division by zero.

    Returns:
        AttributionStatistics model.
    """
    if not map_2d or not map_2d[0]:
        raise ValidationError("Input map_2d must be non-empty [H, W].")

    h = len(map_2d)
    w = len(map_2d[0])
    total_pixels = float(h * w)

    flat = [map_2d[r][c] for r in range(h) for c in range(w)]
    is_finite = all(math.isfinite(v) for v in flat)

    min_val = float(min(flat))
    max_val = float(max(flat))
    mean_val = float(sum(flat) / total_pixels)
    var_val = sum((v - mean_val) ** 2 for v in flat) / total_pixels
    std_val = float(math.sqrt(max(0.0, var_val)))

    # Absolute mass distribution
    abs_flat = [abs(v) for v in flat]
    total_abs_mass = float(sum(abs_flat))

    # Top-p% mass fractions
    sorted_abs = sorted(abs_flat, reverse=True)
    k_10 = max(1, math.ceil(0.10 * total_pixels))
    k_25 = max(1, math.ceil(0.25 * total_pixels))

    top_10_mass = sum(sorted_abs[:k_10])
    top_25_mass = sum(sorted_abs[:k_25])

    denom_mass = total_abs_mass if total_abs_mass > eps else 1.0
    top_10_frac = float(top_10_mass / denom_mass)
    top_25_frac = float(top_25_mass / denom_mass)

    # Spatial Entropy H(p) = - \sum p_i \ln(p_i + eps)
    # Concentration C = 1 - H(p) / \ln(total_pixels)
    p_dist = [v / denom_mass for v in abs_flat]
    entropy = float(-sum(p * math.log(max(eps, p)) for p in p_dist if p > 0.0))
    max_entropy = math.log(total_pixels) if total_pixels > 1 else 1.0
    concentration = (
        float(max(0.0, min(1.0, 1.0 - (entropy / max_entropy))))
        if max_entropy > eps
        else 1.0
    )

    # Center of mass: (\bar{r}, \bar{c})
    if total_abs_mass > eps:
        com_r = (
            sum(float(r) * abs(map_2d[r][c]) for r in range(h) for c in range(w))
            / total_abs_mass
        )
        com_c = (
            sum(float(c) * abs(map_2d[r][c]) for r in range(h) for c in range(w))
            / total_abs_mass
        )
    else:
        com_r = float(h - 1) / 2.0
        com_c = float(w - 1) / 2.0

    return AttributionStatistics(
        min_value=min_val,
        max_value=max_val,
        mean_value=mean_val,
        std_value=std_val,
        total_absolute_mass=total_abs_mass,
        top_10_percent_mass_fraction=top_10_frac,
        top_25_percent_mass_fraction=top_25_frac,
        spatial_entropy=entropy,
        concentration_score=concentration,
        center_of_mass_row=com_r,
        center_of_mass_col=com_c,
        is_finite=is_finite,
    )
