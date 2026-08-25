"""Artifact contracts and reference schemas for tracking run outputs."""

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from prism.core.enums import ArtifactType
from prism.core.identifiers import ensure_valid_identifier


def _utc_now() -> datetime:
    """Return timezone-aware current UTC datetime."""
    return datetime.now(timezone.utc)


class ArtifactReference(BaseModel):
    """Immutable reference to an artifact generated during experiment execution."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    artifact_id: str = Field(
        description="Unique artifact identifier (e.g. 'art-checkpoint-best')"
    )
    artifact_type: ArtifactType = Field(
        description="Category classification of the generated artifact"
    )
    logical_name: str = Field(
        description="Human-readable logical key (e.g. 'best_model_weights')"
    )
    uri: str = Field(description="Relative/absolute path, S3 URI, or storage locator")
    checksum: str | None = Field(
        default=None,
        description="SHA-256 cryptographic digest of the artifact file",
    )
    producing_run_id: str = Field(
        description="Identifier of the ExperimentRun that generated this"
    )
    created_at: datetime = Field(
        default_factory=_utc_now,
        description="UTC creation timestamp",
    )
    size_bytes: int | None = Field(
        default=None,
        ge=0,
        description="File size in bytes if available",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Contextual metadata (e.g. epoch, resolution)",
    )

    @field_validator("artifact_id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        return ensure_valid_identifier(v, field_name="artifact_id")

    @field_validator("producing_run_id")
    @classmethod
    def validate_run_id(cls, v: str) -> str:
        return ensure_valid_identifier(v, field_name="producing_run_id")

    @field_validator("logical_name", "uri")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty.")
        return v.strip()
