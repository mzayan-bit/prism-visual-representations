"""Attention weight representation descriptors, entropy analysis, and comparison."""

from __future__ import annotations

import json
import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from prism.core.errors import SerializationError, ValidationError
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
    min_entropy: float = Field(
        default=0.0,
        description="Minimum Shannon entropy among rows in this head (in nats)",
    )
    max_entropy: float = Field(
        default=0.0,
        description="Maximum Shannon entropy among rows in this head (in nats)",
    )
    diagonal_mass: float = Field(
        default=0.0,
        description="Average diagonal attention weight (self-token focus)",
    )
    off_diagonal_mass: float = Field(
        default=0.0,
        description="Average off-diagonal attention weight across queries",
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
    """Comprehensive summary of an attention weight tensor [N, H, L_q, L_k]."""

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
    min_entropy: float = Field(
        default=0.0,
        description="Global minimum Shannon entropy across all heads and rows",
    )
    max_entropy: float = Field(
        default=0.0,
        description="Global maximum Shannon entropy across all heads and rows",
    )
    mean_diagonal_mass: float = Field(
        default=0.0,
        description="Global average diagonal attention weight",
    )
    mean_off_diagonal_mass: float = Field(
        default=0.0,
        description="Global average off-diagonal attention weight",
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


# Aliases for pattern contract naming
AttentionPattern = AttentionTensorSummary
AttentionWeightSummary = AttentionTensorSummary


def compute_attention_entropy(
    weights: Any,
    eps: float = 1e-12,
) -> list[list[list[float]]]:
    """Compute token-level Shannon entropy (in nats) for each query row.

    H(q_i) = - sum_j p_ij * ln(p_ij + eps)

    Parameters
    ----------
    weights : 4D attention tensor [N, H, L_q, L_k] or 3D [H, L_q, L_k]
    eps : Small numerical epsilon to prevent log(0)

    Returns
    -------
    list[list[list[float]]]
        3D tensor [N, H, L_q] of Shannon entropies per query token.
    """
    w_4d = ensure_4d_attention_tensor(weights)
    n_samples = len(w_4d)
    num_heads = len(w_4d[0])
    l_q = len(w_4d[0][0])
    l_k = len(w_4d[0][0][0])

    entropies_3d: list[list[list[float]]] = []
    for n in range(n_samples):
        sample_entropies: list[list[float]] = []
        for h in range(num_heads):
            head_entropies: list[float] = []
            for i in range(l_q):
                row = w_4d[n][h][i]
                row_entropy = 0.0
                for j in range(l_k):
                    val = max(0.0, row[j])
                    if val > eps:
                        row_entropy -= val * math.log(val)
                head_entropies.append(row_entropy)
            sample_entropies.append(head_entropies)
        entropies_3d.append(sample_entropies)

    return entropies_3d


def compute_diagonal_attention_mass(weights: Any) -> float:
    """Compute average diagonal attention mass (token self-focus).

    Parameters
    ----------
    weights : 4D attention tensor [N, H, L_q, L_k] or 3D [H, L_q, L_k]

    Returns
    -------
    float
        Average diagonal attention weight across all queries and heads.
    """
    w_4d = ensure_4d_attention_tensor(weights)
    n_samples = len(w_4d)
    num_heads = len(w_4d[0])
    l_q = len(w_4d[0][0])
    l_k = len(w_4d[0][0][0])
    diag_len = min(l_q, l_k)

    if diag_len <= 0:
        return 0.0

    diag_sum = 0.0
    total_diag_elements = n_samples * num_heads * diag_len

    for n in range(n_samples):
        for h in range(num_heads):
            for i in range(diag_len):
                diag_sum += w_4d[n][h][i][i]

    return diag_sum / float(total_diag_elements)


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
    diag_len = min(l_q, l_k)

    global_min = float("inf")
    global_max = float("-inf")
    global_sum = 0.0
    global_zeros = 0
    global_min_entropy = float("inf")
    global_max_entropy = float("-inf")
    global_diag_sum = 0.0
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
        h_min_entropy = float("inf")
        h_max_entropy = float("-inf")
        h_diag_sum = 0.0
        h_row_normalized = True
        h_elements = n_samples * l_q * l_k
        h_queries = n_samples * l_q

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

                    if i == j:
                        h_diag_sum += val
                        global_diag_sum += val

                if row_entropy < h_min_entropy:
                    h_min_entropy = row_entropy
                if row_entropy > h_max_entropy:
                    h_max_entropy = row_entropy
                if row_entropy < global_min_entropy:
                    global_min_entropy = row_entropy
                if row_entropy > global_max_entropy:
                    global_max_entropy = row_entropy

                h_entropy_sum += row_entropy
                if abs(row_sum - 1.0) > tolerance:
                    h_row_normalized = False
                    all_rows_normalized = False

        h_mean_entropy = h_entropy_sum / float(h_queries) if h_queries > 0 else 0.0
        head_entropies[h] = h_mean_entropy

        h_diag_mass = (
            h_diag_sum / float(n_samples * diag_len)
            if (n_samples * diag_len > 0)
            else 0.0
        )
        h_off_diag_mass = max(0.0, 1.0 - h_diag_mass)

        head_summaries.append(
            AttentionHeadSummary(
                head_index=h,
                min_value=h_min if h_min != float("inf") else 0.0,
                max_value=h_max if h_max != float("-inf") else 0.0,
                mean_value=(h_sum / float(h_elements) if h_elements > 0 else 0.0),
                entropy=h_mean_entropy,
                min_entropy=(h_min_entropy if h_min_entropy != float("inf") else 0.0),
                max_entropy=(h_max_entropy if h_max_entropy != float("-inf") else 0.0),
                diagonal_mass=h_diag_mass,
                off_diagonal_mass=h_off_diag_mass,
                is_row_normalized=h_row_normalized,
                zero_fraction=(
                    float(h_zeros) / float(h_elements) if h_elements > 0 else 0.0
                ),
            )
        )

    mean_entropy = sum(head_entropies) / float(num_heads) if num_heads > 0 else 0.0
    total_diag_elements = n_samples * num_heads * diag_len
    mean_diag_mass = (
        global_diag_sum / float(total_diag_elements) if total_diag_elements > 0 else 0.0
    )
    mean_off_diag_mass = max(0.0, 1.0 - mean_diag_mass)

    return AttentionTensorSummary(
        tensor_shape=(n_samples, num_heads, l_q, l_k),
        batch_size=n_samples,
        num_heads=num_heads,
        seq_len=l_q,
        min_value=global_min if global_min != float("inf") else 0.0,
        max_value=global_max if global_max != float("-inf") else 0.0,
        mean_value=(global_sum / float(total_elements) if total_elements > 0 else 0.0),
        mean_entropy=mean_entropy,
        min_entropy=(global_min_entropy if global_min_entropy != float("inf") else 0.0),
        max_entropy=(
            global_max_entropy if global_max_entropy != float("-inf") else 0.0
        ),
        mean_diagonal_mass=mean_diag_mass,
        mean_off_diagonal_mass=mean_off_diag_mass,
        zero_fraction=(
            float(global_zeros) / float(total_elements) if total_elements > 0 else 0.0
        ),
        is_finite=all_finite,
        is_row_normalized=all_rows_normalized,
        head_summaries=head_summaries,
    )


# Alias for function naming
compute_attention_summary = summarize_attention_weights


def compare_attention_summaries(
    summary_a: AttentionTensorSummary,
    summary_b: AttentionTensorSummary,
) -> dict[str, Any]:
    """Compare statistical metrics and entropy shifts between two attention summaries.

    Parameters
    ----------
    summary_a : Baseline attention summary
    summary_b : Candidate attention summary

    Returns
    -------
    dict[str, Any]
        Structured delta dictionary tracking entropy shifts and attention dispersion.
    """
    if summary_a.num_heads != summary_b.num_heads:
        raise ValidationError(
            f"Cannot compare summaries with different head counts: "
            f"{summary_a.num_heads} vs {summary_b.num_heads}."
        )

    head_entropy_deltas = [
        b_head.entropy - a_head.entropy
        for a_head, b_head in zip(
            summary_a.head_summaries, summary_b.head_summaries, strict=True
        )
    ]
    head_diag_deltas = [
        b_head.diagonal_mass - a_head.diagonal_mass
        for a_head, b_head in zip(
            summary_a.head_summaries, summary_b.head_summaries, strict=True
        )
    ]

    return {
        "mean_entropy_delta": summary_b.mean_entropy - summary_a.mean_entropy,
        "min_entropy_delta": summary_b.min_entropy - summary_a.min_entropy,
        "max_entropy_delta": summary_b.max_entropy - summary_a.max_entropy,
        "diagonal_mass_delta": (
            summary_b.mean_diagonal_mass - summary_a.mean_diagonal_mass
        ),
        "off_diagonal_mass_delta": (
            summary_b.mean_off_diagonal_mass - summary_a.mean_off_diagonal_mass
        ),
        "max_value_delta": summary_b.max_value - summary_a.max_value,
        "min_value_delta": summary_b.min_value - summary_a.min_value,
        "zero_fraction_delta": (summary_b.zero_fraction - summary_a.zero_fraction),
        "head_entropy_deltas": head_entropy_deltas,
        "head_diagonal_mass_deltas": head_diag_deltas,
    }


class TransformerAttentionProfile(BaseModel):
    """Multi-layer attention behavior profile across Transformer encoder depth."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str = Field(description="Identifier of evaluated Transformer model")
    depth: int = Field(ge=1, description="Number of encoder layers in profile")
    num_heads: int = Field(ge=1, description="Number of attention heads per layer")
    layer_summaries: list[AttentionTensorSummary] = Field(
        description="Per-layer attention tensor summaries ordered from 0 to L-1"
    )
    layer_mean_entropies: list[float] = Field(
        description="Mean attention entropy at each encoder layer"
    )
    layer_diagonal_masses: list[float] = Field(
        description="Mean diagonal attention mass at each encoder layer"
    )
    entropy_trend: str = Field(
        default="stable",
        description="Qualitative entropy progression across depth",
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert profile to dictionary."""
        return self.model_dump(mode="json")

    def to_json(self, indent: int | None = None) -> str:
        """Convert profile to JSON string."""
        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=True,
            ensure_ascii=True,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TransformerAttentionProfile:
        """Create profile from dictionary."""
        try:
            return cls.model_validate(data)
        except Exception as exc:
            raise SerializationError(
                f"Failed to deserialize TransformerAttentionProfile: {exc}"
            ) from exc

    @classmethod
    def from_json(cls, json_str: str) -> TransformerAttentionProfile:
        """Create profile from JSON string."""
        try:
            return cls.from_dict(json.loads(json_str))
        except SerializationError:
            raise
        except Exception as exc:
            raise SerializationError(
                f"Invalid JSON string for TransformerAttentionProfile: {exc}"
            ) from exc


def compute_transformer_attention_profile(
    attention_weights: list[Any],
    model_id: str = "vit-model",
    tolerance: float = 1e-4,
) -> TransformerAttentionProfile:
    """Compute depth-wise attention evolution profile across all Transformer layers."""
    if not attention_weights:
        raise ValidationError("attention_weights list cannot be empty.")

    depth = len(attention_weights)
    layer_summaries: list[AttentionTensorSummary] = []
    layer_entropies: list[float] = []
    layer_diags: list[float] = []

    for layer_w in attention_weights:
        summary = summarize_attention_weights(layer_w, tolerance=tolerance)
        layer_summaries.append(summary)
        layer_entropies.append(summary.mean_entropy)
        layer_diags.append(summary.mean_diagonal_mass)

    num_heads = layer_summaries[0].num_heads

    # Determine entropy trend
    if depth >= 2:
        diff = layer_entropies[-1] - layer_entropies[0]
        if diff > 0.05:
            trend = "increasing"
        elif diff < -0.05:
            trend = "decreasing"
        else:
            trend = "stable"
    else:
        trend = "single_layer"

    return TransformerAttentionProfile(
        model_id=model_id,
        depth=depth,
        num_heads=num_heads,
        layer_summaries=layer_summaries,
        layer_mean_entropies=layer_entropies,
        layer_diagonal_masses=layer_diags,
        entropy_trend=trend,
    )


def compare_transformer_attention_profiles(
    profile_a: TransformerAttentionProfile,
    profile_b: TransformerAttentionProfile,
) -> dict[str, Any]:
    """Compare attention evolution across depth between two Transformer models."""
    if profile_a.depth != profile_b.depth:
        raise ValidationError(
            f"Cannot compare profiles with different depths: "
            f"{profile_a.depth} vs {profile_b.depth}."
        )

    layer_comparisons = [
        compare_attention_summaries(sum_a, sum_b)
        for sum_a, sum_b in zip(
            profile_a.layer_summaries, profile_b.layer_summaries, strict=True
        )
    ]

    return {
        "model_id_a": profile_a.model_id,
        "model_id_b": profile_b.model_id,
        "depth": profile_a.depth,
        "layer_comparisons": layer_comparisons,
        "layer_entropy_deltas": [
            b_ent - a_ent
            for a_ent, b_ent in zip(
                profile_a.layer_mean_entropies,
                profile_b.layer_mean_entropies,
                strict=True,
            )
        ],
        "layer_diagonal_mass_deltas": [
            b_diag - a_diag
            for a_diag, b_diag in zip(
                profile_a.layer_diagonal_masses,
                profile_b.layer_diagonal_masses,
                strict=True,
            )
        ],
    }
