"""Attention weight representation descriptors and statistical summaries."""

from __future__ import annotations

import json
import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from prism.core.errors import SerializationError
from prism.models.attention import ensure_4d_attention_tensor


class AttentionHeadSummary(BaseModel):
    """Statistical summary of attention weights for a single attention head."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    head_index: int = Field(ge=0, description="Zero-based index of attention head")
    min_value: float = Field(description="Minimum attention probability in head")
    max_value: float = Field(description="Maximum attention probability in head")
    mean_value: float = Field(description="Mean attention probability in head")
    entropy: float = Field(
        description="Average Shannon entropy across rows in this head (in nats)"
    )
    is_row_normalized: bool = Field(
        description="True if all rows sum to 1.0 within tolerance"
    )
    zero_fraction: float = Field(
        description="Fraction of attention weights approximately equal to 0.0"
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert head summary to dictionary."""
        return self.model_dump(mode="json")

    def to_json(self, indent: int | None = None) -> str:
        """Convert head summary to JSON string."""
        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=True,
            ensure_ascii=True,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AttentionHeadSummary:
        """Create summary from dictionary."""
        try:
            return cls.model_validate(data)
        except Exception as exc:
            raise SerializationError(
                f"Failed to deserialize AttentionHeadSummary: {exc}"
            ) from exc

    @classmethod
    def from_json(cls, json_str: str) -> AttentionHeadSummary:
        """Create summary from JSON string."""
        try:
            return cls.from_dict(json.loads(json_str))
        except SerializationError:
            raise
        except Exception as exc:
            raise SerializationError(
                f"Invalid JSON string for AttentionHeadSummary: {exc}"
            ) from exc


class AttentionTensorSummary(BaseModel):
    """Comprehensive summary of an attention weight tensor [N, H, L, L]."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tensor_shape: tuple[int, ...] = Field(
        description="4D shape dimensions (batch_size, num_heads, l_q, l_k)"
    )
    batch_size: int = Field(ge=1, description="Batch size dimension N")
    num_heads: int = Field(ge=1, description="Number of attention heads H")
    seq_len: int = Field(ge=1, description="Query sequence length L_q")
    min_value: float = Field(description="Global minimum attention probability")
    max_value: float = Field(description="Global maximum attention probability")
    mean_value: float = Field(description="Global mean attention probability")
    mean_entropy: float = Field(
        description="Global average Shannon entropy across all heads and rows"
    )
    zero_fraction: float = Field(
        description="Fraction of all entries approximately equal to 0.0"
    )
    is_finite: bool = Field(
        description="True if all attention weights are finite numbers"
    )
    is_row_normalized: bool = Field(
        description="True if all rows across all batch items and heads sum to 1.0"
    )
    head_summaries: list[AttentionHeadSummary] = Field(
        default_factory=list,
        description="Per-head statistical breakdowns",
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert attention tensor summary to dictionary."""
        return self.model_dump(mode="json")

    def to_json(self, indent: int | None = None) -> str:
        """Convert attention tensor summary to JSON string."""
        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=True,
            ensure_ascii=True,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AttentionTensorSummary:
        """Create summary from dictionary."""
        try:
            return cls.model_validate(data)
        except Exception as exc:
            raise SerializationError(
                f"Failed to deserialize AttentionTensorSummary: {exc}"
            ) from exc

    @classmethod
    def from_json(cls, json_str: str) -> AttentionTensorSummary:
        """Create summary from JSON string."""
        try:
            return cls.from_dict(json.loads(json_str))
        except SerializationError:
            raise
        except Exception as exc:
            raise SerializationError(
                f"Invalid JSON string for AttentionTensorSummary: {exc}"
            ) from exc


def summarize_attention_weights(
    weights: Any,
    tolerance: float = 1e-4,
) -> AttentionTensorSummary:
    """Extract non-mutating summary and row-normalization audit from attention tensor.

    Parameters
    ----------
    weights : 4D attention tensor [N, H, L_q, L_k] or 3D [H, L_q, L_k]
    tolerance : Numerical absolute tolerance for row probability sum validation

    Returns
    -------
    AttentionTensorSummary
        Consolidated summary contract containing global and per-head statistics.
    """
    w_4d = ensure_4d_attention_tensor(weights)
    n_samples = len(w_4d)
    num_heads = len(w_4d[0])
    l_q = len(w_4d[0][0])
    l_k = len(w_4d[0][0][0])

    global_min = float("inf")
    global_max = float("-inf")
    global_sum = 0.0
    global_zeros = 0
    total_elements = n_samples * num_heads * l_q * l_k
    all_rows_normalized = True
    all_finite = True
    head_entropies: list[float] = [0.0] * num_heads
    head_summaries: list[AttentionHeadSummary] = []

    for h in range(num_heads):
        h_min = float("inf")
        h_max = float("-inf")
        h_sum = 0.0
        h_zeros = 0
        h_entropy_sum = 0.0
        h_row_normalized = True
        h_elements = n_samples * l_q * l_k

        for n in range(n_samples):
            for i in range(l_q):
                row = w_4d[n][h][i]
                row_sum = 0.0
                row_entropy = 0.0
                for j in range(l_k):
                    val = row[j]
                    if math.isnan(val) or math.isinf(val):
                        all_finite = False
                    val_clamped = max(0.0, val)
                    row_sum += val
                    if val_clamped > 1e-12:
                        row_entropy -= val_clamped * math.log(val_clamped)
                    if abs(val) < 1e-7:
                        h_zeros += 1
                        global_zeros += 1
                    if val < h_min:
                        h_min = val
                    if val > h_max:
                        h_max = val
                    if val < global_min:
                        global_min = val
                    if val > global_max:
                        global_max = val
                    h_sum += val
                    global_sum += val

                h_entropy_sum += row_entropy
                if abs(row_sum - 1.0) > tolerance:
                    h_row_normalized = False
                    all_rows_normalized = False

        h_mean_entropy = h_entropy_sum / float(n_samples * l_q)
        head_entropies[h] = h_mean_entropy

        head_summaries.append(
            AttentionHeadSummary(
                head_index=h,
                min_value=h_min if h_min != float("inf") else 0.0,
                max_value=h_max if h_max != float("-inf") else 0.0,
                mean_value=h_sum / float(h_elements) if h_elements > 0 else 0.0,
                entropy=h_mean_entropy,
                is_row_normalized=h_row_normalized,
                zero_fraction=(
                    float(h_zeros) / float(h_elements) if h_elements > 0 else 0.0
                ),
            )
        )

    mean_entropy = sum(head_entropies) / float(num_heads) if num_heads > 0 else 0.0

    return AttentionTensorSummary(
        tensor_shape=(n_samples, num_heads, l_q, l_k),
        batch_size=n_samples,
        num_heads=num_heads,
        seq_len=l_q,
        min_value=global_min if global_min != float("inf") else 0.0,
        max_value=global_max if global_max != float("-inf") else 0.0,
        mean_value=(global_sum / float(total_elements) if total_elements > 0 else 0.0),
        mean_entropy=mean_entropy,
        zero_fraction=(
            float(global_zeros) / float(total_elements) if total_elements > 0 else 0.0
        ),
        is_finite=all_finite,
        is_row_normalized=all_rows_normalized,
        head_summaries=head_summaries,
    )
