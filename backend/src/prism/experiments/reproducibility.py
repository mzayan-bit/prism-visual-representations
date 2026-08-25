"""Reproducibility configuration and audit requirements."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ReproducibilityConfiguration(BaseModel):
    """Configuration governing deterministic execution and provenance capture."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int = Field(
        default=42,
        ge=0,
        description="Master pseudo-random number generator seed",
    )
    deterministic: bool = Field(
        default=True,
        description="Enforce deterministic algorithms (e.g. cuDNN deterministic mode)",
    )
    capture_code_revision: bool = Field(
        default=True,
        description="Record git commit hash, branch, and dirty status",
    )
    capture_environment: bool = Field(
        default=True,
        description="Record host OS, hardware, and installed package versions",
    )
    capture_dataset_fingerprint: bool = Field(
        default=True,
        description="Require and record dataset partition digests",
    )
    hash_configuration: bool = Field(
        default=True,
        description="Compute and verify SHA-256 configuration hashes",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional reproducibility constraints",
    )
