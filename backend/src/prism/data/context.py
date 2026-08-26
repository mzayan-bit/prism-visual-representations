"""Data runtime execution context and audit trail."""

from __future__ import annotations

import json
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from prism.core.errors import SerializationError
from prism.experiments.hashing import compute_configuration_fingerprint


class DataRuntimeContext(BaseModel):
    """Immutable audit trail describing how dataset was prepared and batched."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_id: str = Field(description="Dataset identifier (e.g. 'ds-cifar10')")
    canonical_manifest_fingerprint: str = Field(
        description="Cryptographic SHA-256 fingerprint of canonical universe"
    )
    partition_manifest_fingerprint: str | None = Field(
        default=None,
        description="Cryptographic SHA-256 fingerprint of partition manifest",
    )
    subset_manifest_fingerprint: str | None = Field(
        default=None,
        description="Cryptographic SHA-256 fingerprint of nested subset manifest",
    )
    resolved_sample_count: int = Field(
        ge=0,
        description="Total number of samples successfully materialized in memory",
    )
    ordering_strategy: str = Field(
        description="Sampling or traversal ordering strategy"
    )
    ordering_fingerprint: str = Field(
        description="Cryptographic SHA-256 fingerprint of sample sequence"
    )
    batch_size: int = Field(
        ge=1,
        description="Configured batch size for iteration",
    )
    drop_last: bool = Field(
        default=False,
        description="Whether incomplete final batch is discarded",
    )
    backend_name: str = Field(
        default="in_memory",
        description="Materialization provider/backend identifier",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Warnings or notices recorded during data preparation",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Auxiliary context metadata",
    )
    schema_version: str = Field(
        default="1.0.0",
        description="Schema contract version",
    )

    def compute_fingerprint(self) -> str:
        """Compute deterministic SHA-256 fingerprint of this runtime context."""
        return compute_configuration_fingerprint(self)

    def to_dict(self) -> dict[str, Any]:
        """Convert runtime context to dictionary."""
        return self.model_dump(mode="json")

    def to_json(self, indent: int | None = None) -> str:
        """Convert runtime context to JSON string."""
        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=True,
            ensure_ascii=True,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DataRuntimeContext:
        """Create runtime context from dictionary."""
        try:
            return cls.model_validate(data)
        except Exception as exc:
            raise SerializationError(
                f"Failed to parse DataRuntimeContext from dict: {exc}"
            ) from exc

    @classmethod
    def from_json(cls, json_str: str) -> DataRuntimeContext:
        """Create runtime context from JSON string."""
        try:
            parsed = json.loads(json_str)
            return cls.from_dict(parsed)
        except Exception as exc:
            raise SerializationError(
                f"Failed to parse DataRuntimeContext from JSON: {exc}"
            ) from exc
