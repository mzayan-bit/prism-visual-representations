"""Vision Transformer attention drift analysis comparing clean vs corrupted."""

from __future__ import annotations

import json
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from prism.core.errors import SerializationError
from prism.models.base import BaseVisionModel
from prism.models.transformer import VisionTransformer
from prism.representations.attention import (
    AttentionTensorSummary,
    summarize_attention_weights,
)


class LayerAttentionDrift(BaseModel):
    """Attention entropy and mass changes for a single transformer encoder block."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    layer_name: str = Field(description="Layer identifier name (e.g. 'encoder_0')")
    clean_entropy: float = Field(
        ge=0.0, description="Clean mean attention entropy (nats)"
    )
    corrupted_entropy: float = Field(
        ge=0.0, description="Corrupted mean attention entropy (nats)"
    )
    entropy_delta: float = Field(
        description="Entropy change: (corrupted - clean) in nats"
    )
    clean_diagonal_mass: float = Field(
        ge=0.0, le=1.0, description="Clean mean diagonal attention mass fraction"
    )
    corrupted_diagonal_mass: float = Field(
        ge=0.0, le=1.0, description="Corrupted mean diagonal attention mass fraction"
    )
    diagonal_mass_delta: float = Field(
        description="Diagonal mass change: (corrupted - clean)"
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert layer drift to dictionary."""
        return self.model_dump(mode="json")


class AttentionDriftSummary(BaseModel):
    """Summary of multi-head attention weight changes across transformer depth."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str = Field(description="Vision Transformer model identifier")
    num_layers: int = Field(ge=1, description="Number of attention layers evaluated")
    clean_overall_mean_entropy: float = Field(
        ge=0.0, description="Overall clean mean attention entropy"
    )
    corrupted_overall_mean_entropy: float = Field(
        ge=0.0, description="Overall corrupted mean attention entropy"
    )
    overall_entropy_delta: float = Field(
        description="Overall entropy change: (corrupted - clean)"
    )
    clean_overall_diagonal_mass: float = Field(
        ge=0.0, le=1.0, description="Overall clean diagonal attention mass fraction"
    )
    corrupted_overall_diagonal_mass: float = Field(
        ge=0.0, le=1.0, description="Overall corrupted diagonal attention mass fraction"
    )
    overall_diagonal_mass_delta: float = Field(
        description="Overall diagonal mass change: (corrupted - clean)"
    )
    layer_drifts: list[LayerAttentionDrift] = Field(
        description="Ordered sequence of per-layer attention drift summaries"
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert summary to dictionary."""
        return self.model_dump(mode="json")

    def to_json(self, indent: int | None = None) -> str:
        """Convert summary to JSON string."""
        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=True,
            ensure_ascii=True,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AttentionDriftSummary:
        """Create summary from dictionary."""
        try:
            return cls.model_validate(data)
        except Exception as exc:
            raise SerializationError(
                f"Failed to deserialize AttentionDriftSummary: {exc}"
            ) from exc


def compute_vit_attention_drift(
    model: BaseVisionModel,
    clean_inputs: Any,
    corrupted_inputs: Any,
) -> AttentionDriftSummary | None:
    """Extract and compare attention weights between clean and corrupted inputs."""
    if not isinstance(model, VisionTransformer):
        return None

    # Forward pass on clean inputs to capture attention weights
    _ = model.forward(clean_inputs)
    clean_weights_map: dict[str, list[list[list[list[float]]]]] = {}
    for block_idx, block in enumerate(model.encoder.blocks):
        layer_name = f"encoder_{block_idx}"
        if block.last_attention_weights is not None:
            clean_weights_map[layer_name] = block.last_attention_weights

    # Forward pass on corrupted inputs
    _ = model.forward(corrupted_inputs)
    corrupted_weights_map: dict[str, list[list[list[list[float]]]]] = {}
    for block_idx, block in enumerate(model.encoder.blocks):
        layer_name = f"encoder_{block_idx}"
        if block.last_attention_weights is not None:
            corrupted_weights_map[layer_name] = block.last_attention_weights

    layer_drifts: list[LayerAttentionDrift] = []
    clean_entropies: list[float] = []
    corrupted_entropies: list[float] = []
    clean_diags: list[float] = []
    corrupted_diags: list[float] = []

    for block_idx in range(len(model.encoder.blocks)):
        layer_name = f"encoder_{block_idx}"
        if layer_name in clean_weights_map and layer_name in corrupted_weights_map:
            c_summary: AttentionTensorSummary = summarize_attention_weights(
                clean_weights_map[layer_name]
            )
            cr_summary: AttentionTensorSummary = summarize_attention_weights(
                corrupted_weights_map[layer_name]
            )

            c_ent = c_summary.mean_entropy
            cr_ent = cr_summary.mean_entropy
            c_diag = c_summary.mean_diagonal_mass
            cr_diag = cr_summary.mean_diagonal_mass

            drift = LayerAttentionDrift(
                layer_name=layer_name,
                clean_entropy=c_ent,
                corrupted_entropy=cr_ent,
                entropy_delta=cr_ent - c_ent,
                clean_diagonal_mass=c_diag,
                corrupted_diagonal_mass=cr_diag,
                diagonal_mass_delta=cr_diag - c_diag,
            )
            layer_drifts.append(drift)

            clean_entropies.append(c_ent)
            corrupted_entropies.append(cr_ent)
            clean_diags.append(c_diag)
            corrupted_diags.append(cr_diag)

    if not layer_drifts:
        return None

    num_l = len(layer_drifts)
    c_overall_ent = sum(clean_entropies) / float(num_l)
    cr_overall_ent = sum(corrupted_entropies) / float(num_l)
    c_overall_diag = sum(clean_diags) / float(num_l)
    cr_overall_diag = sum(corrupted_diags) / float(num_l)

    return AttentionDriftSummary(
        model_id=model.model_id,
        num_layers=num_l,
        clean_overall_mean_entropy=c_overall_ent,
        corrupted_overall_mean_entropy=cr_overall_ent,
        overall_entropy_delta=cr_overall_ent - c_overall_ent,
        clean_overall_diagonal_mass=c_overall_diag,
        corrupted_overall_diagonal_mass=cr_overall_diag,
        overall_diagonal_mass_delta=cr_overall_diag - c_overall_diag,
        layer_drifts=layer_drifts,
    )
