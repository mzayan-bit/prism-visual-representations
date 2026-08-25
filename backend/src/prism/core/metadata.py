"""Metadata schemas for tracking provenance, revisions, and environments."""

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def _utc_now() -> datetime:
    """Return timezone-aware current UTC datetime."""
    return datetime.now(timezone.utc)


class CreationMetadata(BaseModel):
    """Metadata detailing entity creation timestamp and author."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    created_at: datetime = Field(default_factory=_utc_now)
    created_by: str | None = Field(default=None)
    schema_version: str = Field(default="1.0.0")
    tags: list[str] = Field(default_factory=list)
    custom_metadata: dict[str, Any] = Field(default_factory=dict)


class CodeRevisionMetadata(BaseModel):
    """Metadata capturing the source code version control state."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    git_commit: str | None = Field(
        default=None,
        description="Full SHA-1/SHA-256 commit hash",
    )
    git_branch: str | None = Field(
        default=None,
        description="Active git branch name during execution",
    )
    is_dirty: bool = Field(
        default=False,
        description="True if working tree had uncommitted modifications",
    )
    repository_url: str | None = Field(
        default=None,
        description="Upstream repository remote URL",
    )


class EnvironmentMetadata(BaseModel):
    """Metadata capturing the runtime host environment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    python_version: str = Field(description="Runtime Python version string")
    os: str = Field(description="Operating system name and release")
    hardware: str | None = Field(
        default=None,
        description="CPU or GPU model descriptor",
    )
    cuda_version: str | None = Field(
        default=None,
        description="CUDA runtime version if available",
    )
    packages: dict[str, str] = Field(
        default_factory=dict,
        description="Key package versions (e.g. torch, torchvision)",
    )
