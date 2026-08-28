"""Feature distribution summaries and representation stability measurement."""

from __future__ import annotations

import json
import math
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from prism.core.errors import SerializationError, ValidationError


def _flatten_and_measure(
    data: Any,
) -> tuple[list[float], tuple[int, ...]]:
    """Recursively extract all numeric scalars and infer tensor shape."""
    if not isinstance(data, list):
        try:
            val = float(data)
            return [val], ()
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                f"Non-numeric element in feature tensor: {data}"
            ) from exc

    if not data:
        return [], (0,)

    shape = [len(data)]
    sub_shape: tuple[int, ...] | None = None
    flat: list[float] = []

    for item in data:
        sub_flat, cur_shape = _flatten_and_measure(item)
        if sub_shape is None:
            sub_shape = cur_shape
        elif sub_shape != cur_shape:
            raise ValidationError(
                f"Ragged nested list structure: {sub_shape} vs {cur_shape}"
            )
        flat.extend(sub_flat)

    full_shape = tuple(shape + (list(sub_shape) if sub_shape else []))
    return flat, full_shape


class FeatureDistributionSummary(BaseModel):
    """Structured statistical summary of a representation or feature tensor."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mean: float = Field(description="Sample mean across all elements")
    variance: float = Field(
        ge=0.0, description="Sample variance across all elements"
    )
    std_dev: float = Field(
        ge=0.0, description="Standard deviation across all elements"
    )
    min_value: float = Field(description="Minimum value in feature tensor")
    max_value: float = Field(description="Maximum value in feature tensor")
    zero_fraction: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of entries approximately or strictly equal to 0.0",
    )
    is_finite: bool = Field(
        description="True if all values are finite (no NaN or Inf)"
    )
    sample_count: int = Field(
        ge=0, description="Total number of scalar entries summarized"
    )
    tensor_shape: tuple[int, ...] = Field(
        description="Shape dimensions of summarized tensor"
    )
    channel_means: list[float] | None = Field(
        default=None,
        description="Per-channel mean for 4D spatial tensors [N, C, H, W]",
    )
    channel_variances: list[float] | None = Field(
        default=None,
        description="Per-channel variance for 4D spatial tensors [N, C, H, W]",
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert distribution summary to dictionary."""
        return self.model_dump(mode="json")

    def to_json(self, indent: int | None = None) -> str:
        """Convert distribution summary to JSON string."""
        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=True,
            ensure_ascii=True,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureDistributionSummary:
        """Create summary from dictionary."""
        try:
            return cls.model_validate(data)
        except Exception as exc:
            raise SerializationError(
                f"Failed to deserialize FeatureDistributionSummary from dict: {exc}"
            ) from exc

    @classmethod
    def from_json(cls, json_str: str) -> FeatureDistributionSummary:
        """Create summary from JSON string."""
        try:
            parsed = json.loads(json_str)
            return cls.from_dict(parsed)
        except Exception as exc:
            raise SerializationError(
                f"Failed to deserialize FeatureDistributionSummary from JSON: {exc}"
            ) from exc


def compute_distribution_summary(
    tensor: Any, zero_threshold: float = 1e-12
) -> FeatureDistributionSummary:
    """Compute exact scalar and channel-wise distribution statistics on a tensor."""
    if tensor is None:
        raise ValidationError("Cannot compute distribution summary on None.")

    flat, shape = _flatten_and_measure(tensor)
    total_count = len(flat)

    if total_count == 0:
        return FeatureDistributionSummary(
            mean=0.0,
            variance=0.0,
            std_dev=0.0,
            min_value=0.0,
            max_value=0.0,
            zero_fraction=0.0,
            is_finite=True,
            sample_count=0,
            tensor_shape=shape,
        )

    # Check finiteness
    is_finite = True
    for val in flat:
        if math.isnan(val) or math.isinf(val):
            is_finite = False
            break

    if not is_finite:
        return FeatureDistributionSummary(
            mean=float("nan"),
            variance=float("nan"),
            std_dev=float("nan"),
            min_value=float("nan"),
            max_value=float("nan"),
            zero_fraction=0.0,
            is_finite=False,
            sample_count=total_count,
            tensor_shape=shape,
        )

    # 1. Global Mean
    sum_val = sum(flat)
    mean_val = sum_val / float(total_count)

    # 2. Global Variance & Std Dev
    var_sum = 0.0
    zero_count = 0
    min_val = flat[0]
    max_val = flat[0]

    for val in flat:
        diff = val - mean_val
        var_sum += diff * diff
        if abs(val) <= zero_threshold:
            zero_count += 1
        if val < min_val:
            min_val = val
        if val > max_val:
            max_val = val

    var_val = var_sum / float(total_count)
    std_val = math.sqrt(max(0.0, var_val))
    zero_frac = float(zero_count) / float(total_count)

    # 3. Optional Channel-Wise Statistics for 4D Spatial Tensors [N, C, H, W]
    channel_means: list[float] | None = None
    channel_vars: list[float] | None = None

    if len(shape) == 4:
        n_samples, c_channels, h_len, w_len = shape
        m = n_samples * h_len * w_len

        ch_means: list[float] = [0.0] * c_channels
        for n in range(n_samples):
            for c in range(c_channels):
                for h in range(h_len):
                    for w in range(w_len):
                        ch_means[c] += tensor[n][c][h][w]
        ch_means = [m_c / float(m) for m_c in ch_means]

        ch_vars: list[float] = [0.0] * c_channels
        for n in range(n_samples):
            for c in range(c_channels):
                for h in range(h_len):
                    for w in range(w_len):
                        d_val = tensor[n][c][h][w] - ch_means[c]
                        ch_vars[c] += d_val * d_val
        ch_vars = [v_c / float(m) for v_c in ch_vars]

        channel_means = ch_means
        channel_vars = ch_vars

    return FeatureDistributionSummary(
        mean=mean_val,
        variance=var_val,
        std_dev=std_val,
        min_value=min_val,
        max_value=max_val,
        zero_fraction=zero_frac,
        is_finite=True,
        sample_count=total_count,
        tensor_shape=shape,
        channel_means=channel_means,
        channel_variances=channel_vars,
    )


def compare_distribution_summaries(
    summary_a: FeatureDistributionSummary,
    summary_b: FeatureDistributionSummary,
) -> dict[str, float]:
    """Compare two representation distribution summaries to measure stability shifts."""
    if not summary_a.is_finite or not summary_b.is_finite:
        return {
            "mean_shift": float("nan"),
            "std_shift": float("nan"),
            "range_delta": float("nan"),
            "zero_fraction_delta": float("nan"),
        }

    range_a = summary_a.max_value - summary_a.min_value
    range_b = summary_b.max_value - summary_b.min_value

    return {
        "mean_shift": abs(summary_a.mean - summary_b.mean),
        "std_shift": abs(summary_a.std_dev - summary_b.std_dev),
        "range_delta": abs(range_a - range_b),
        "zero_fraction_delta": abs(
            summary_a.zero_fraction - summary_b.zero_fraction
        ),
    }
