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
    short_commit: str | None = Field(
        default=None,
        description="Abbreviated commit hash (e.g. 7-8 chars)",
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
    modified_files: list[str] = Field(
        default_factory=list,
        description="Tracked modified file paths if working tree was dirty",
    )


class HardwareMetadata(BaseModel):
    """Metadata capturing hardware and compute backend capabilities."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cpu_count: int | None = Field(
        default=None,
        description="Number of logical CPU cores available",
    )
    cpu_architecture: str | None = Field(
        default=None,
        description="CPU architecture string (e.g. 'arm64', 'x86_64')",
    )
    cuda_available: bool = Field(
        default=False,
        description="Whether NVIDIA CUDA acceleration is available",
    )
    cuda_device_count: int = Field(
        default=0,
        ge=0,
        description="Number of detected CUDA GPU devices",
    )
    cuda_device_names: list[str] = Field(
        default_factory=list,
        description="Names of detected CUDA devices",
    )
    cuda_version: str | None = Field(
        default=None,
        description="CUDA runtime version string if available",
    )
    mps_available: bool = Field(
        default=False,
        description="Whether Apple Silicon Metal Performance Shaders is available",
    )
    mps_built: bool = Field(
        default=False,
        description="Whether PyTorch was compiled with MPS support",
    )
    compute_backend: str = Field(
        default="cpu",
        description="Primary active compute backend ('cpu', 'cuda', or 'mps')",
    )
    raw_details: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional non-standard hardware details",
    )


class EnvironmentMetadata(BaseModel):
    """Metadata capturing the runtime host environment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    python_version: str = Field(description="Runtime Python version string")
    python_implementation: str | None = Field(
        default=None,
        description="Python implementation name (e.g. 'CPython', 'PyPy')",
    )
    os: str = Field(description="Operating system name and release")
    platform: str | None = Field(
        default=None,
        description="Detailed platform descriptor string",
    )
    hardware: str | None = Field(
        default=None,
        description="Human-readable hardware summary descriptor",
    )
    hardware_info: HardwareMetadata | None = Field(
        default=None,
        description="Structured hardware capabilities and compute backends",
    )
    cuda_version: str | None = Field(
        default=None,
        description="CUDA runtime version if available",
    )
    packages: dict[str, str] = Field(
        default_factory=dict,
        description="Key package versions (e.g. torch, torchvision, pydantic)",
    )
