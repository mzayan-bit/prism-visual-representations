"""Representation metadata and extracted embedding descriptors."""

from __future__ import annotations

import json
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from prism.core.errors import SerializationError
from prism.core.identifiers import ensure_valid_identifier


class RepresentationDescriptor(BaseModel):
    """Metadata describing an extracted representation tensor or feature bank."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    layer_name: str = Field(
        description="Source layer name (e.g. 'input', 'conv_0', 'final_spatial')"
    )
    feature_dim: int = Field(
        gt=0,
        description="Feature dimensionality of individual vectors or spatial elements",
    )
    num_samples: int = Field(
        ge=0,
        description="Number of sample vectors in this representation batch",
    )
    representation_kind: str = Field(
        default="vector",
        description="Category: 'vector', 'spatial', 'tokens', or 'attention'",
    )
    spatial_shape: tuple[int, int, int] | None = Field(
        default=None,
        description="Spatial dimensions (C, H, W) if representation_kind is 'spatial'",
    )
    tokens_shape: tuple[int, int] | None = Field(
        default=None,
        description="Token dimensions (T, D) if representation_kind is 'tokens'",
    )
    receptive_field: int | None = Field(
        default=None,
        ge=1,
        description="Effective receptive field size in input pixel coordinates",
    )
    sample_ids: list[str] | None = Field(
        default=None,
        description="Canonical sample IDs matching representation rows",
    )
    model_id: str = Field(
        description="Identifier of model that produced the representation"
    )
    is_training_mode: bool = Field(
        default=False,
        description="True if extracted during training mode, False for eval mode",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Auxiliary context (e.g. epoch, split, transform)",
    )

    @property
    def is_spatial(self) -> bool:
        """Return True if this descriptor represents a spatial feature map."""
        return self.representation_kind == "spatial"

    @property
    def is_tokens(self) -> bool:
        """Return True if this descriptor represents a sequence of token vectors."""
        return self.representation_kind == "tokens"

    @property
    def is_attention(self) -> bool:
        """Return True if this descriptor represents an attention pattern tensor."""
        return self.representation_kind == "attention"

    @field_validator("model_id")
    @classmethod
    def validate_model_id_field(cls, v: str) -> str:
        return ensure_valid_identifier(v, field_name="model_id")

    @field_validator("layer_name")
    @classmethod
    def validate_layer(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Layer name cannot be empty.")
        return v.strip().lower()

    @field_validator("representation_kind")
    @classmethod
    def validate_kind(cls, v: str) -> str:
        v_norm = v.strip().lower()
        if v_norm not in ("vector", "spatial", "tokens", "attention"):
            raise ValueError(
                f"representation_kind must be 'vector', 'spatial', 'tokens', "
                f"or 'attention', got '{v}'"
            )
        return v_norm

    def to_dict(self) -> dict[str, Any]:
        """Convert descriptor to dictionary."""
        return self.model_dump(mode="json")

    def to_json(self, indent: int | None = None) -> str:
        """Convert descriptor to JSON string."""
        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=True,
            ensure_ascii=True,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RepresentationDescriptor:
        """Create descriptor from dictionary."""
        try:
            return cls.model_validate(data)
        except Exception as exc:
            raise SerializationError(
                f"Failed to deserialize RepresentationDescriptor from dict: {exc}"
            ) from exc

    @classmethod
    def from_json(cls, json_str: str) -> RepresentationDescriptor:
        """Create descriptor from JSON string."""
        try:
            parsed = json.loads(json_str)
            return cls.from_dict(parsed)
        except Exception as exc:
            raise SerializationError(
                f"Failed to deserialize RepresentationDescriptor from JSON: {exc}"
            ) from exc


class RepresentationBatch(BaseModel):
    """Runtime container coupling a RepresentationDescriptor with feature data."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    descriptor: RepresentationDescriptor = Field(
        description="Metadata descriptor of the representation"
    )
    embeddings: list[Any] = Field(
        description="Extracted feature embedding [N, D] or spatial tensor [N, C, H, W]"
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert representation batch to dictionary."""
        return self.model_dump(mode="json")

    def to_json(self, indent: int | None = None) -> str:
        """Convert representation batch to JSON string."""
        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=True,
            ensure_ascii=True,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RepresentationBatch:
        """Create representation batch from dictionary."""
        try:
            return cls.model_validate(data)
        except Exception as exc:
            raise SerializationError(
                f"Failed to deserialize RepresentationBatch from dict: {exc}"
            ) from exc

    @classmethod
    def from_json(cls, json_str: str) -> RepresentationBatch:
        """Create representation batch from JSON string."""
        try:
            parsed = json.loads(json_str)
            return cls.from_dict(parsed)
        except Exception as exc:
            raise SerializationError(
                f"Failed to deserialize RepresentationBatch from JSON: {exc}"
            ) from exc
