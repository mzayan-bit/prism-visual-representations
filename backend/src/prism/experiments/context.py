"""Runtime context and prepared execution containers."""

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from prism.core.errors import SerializationError
from prism.core.identifiers import ensure_valid_identifier
from prism.core.metadata import (
    CodeRevisionMetadata,
    EnvironmentMetadata,
    HardwareMetadata,
)
from prism.experiments.reproducibility import ReproducibilityConfiguration
from prism.experiments.seeding import SeedInitializationResult


def _utc_now() -> datetime:
    """Return timezone-aware current UTC datetime."""
    return datetime.now(timezone.utc)


class PreparedExecution(BaseModel):
    """Immutable runtime execution context prepared prior to workload initiation.

    Combines experiment identity, configuration fingerprint, run ID,
    reproducibility parameters, multi-backend RNG seeds, environment
    snapshot, hardware capabilities, and git code revision provenance.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_id: str = Field(description="Referenced ExperimentDefinition identifier")
    run_id: str = Field(description="Associated ExperimentRun identifier")
    configuration_fingerprint: str = Field(
        description="SHA-256 semantic fingerprint of the experiment definition"
    )
    reproducibility: ReproducibilityConfiguration = Field(
        description="Governing reproducibility and audit settings"
    )
    seeding_result: SeedInitializationResult = Field(
        description="Structured outcome of multi-backend RNG initialization"
    )
    environment: EnvironmentMetadata = Field(
        description="Snapshot of the host runtime environment and dependencies"
    )
    hardware: HardwareMetadata = Field(
        description="Discovered hardware acceleration and compute capabilities"
    )
    code_revision: CodeRevisionMetadata = Field(
        description="Source code version control provenance"
    )
    initialized_at: datetime = Field(
        default_factory=_utc_now,
        description="UTC timestamp when execution environment was prepared",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Consolidated runtime warnings and non-fatal limitations",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional preparation metadata",
    )

    @field_validator("experiment_id")
    @classmethod
    def validate_exp_id(cls, v: str) -> str:
        return ensure_valid_identifier(v, field_name="experiment_id")

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, v: str) -> str:
        return ensure_valid_identifier(v, field_name="run_id")

    def get_reproducibility_report(self) -> dict[str, Any]:
        """Compile a transparent fact-based reproducibility capability report."""
        return {
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "configuration_fingerprint": self.configuration_fingerprint,
            "requested": {
                "seed": self.reproducibility.seed,
                "deterministic": self.reproducibility.deterministic,
                "capture_code_revision": self.reproducibility.capture_code_revision,
                "capture_environment": self.reproducibility.capture_environment,
            },
            "configured": {
                "seeded_backends": self.seeding_result.configured_backends,
                "deterministic_algorithms_configured": (
                    self.seeding_result.deterministic_algorithms_configured
                ),
                "primary_compute_backend": self.hardware.compute_backend,
            },
            "provenance": {
                "git_commit": self.code_revision.git_commit,
                "short_commit": self.code_revision.short_commit,
                "git_branch": self.code_revision.git_branch,
                "is_dirty": self.code_revision.is_dirty,
                "modified_files": self.code_revision.modified_files,
            },
            "environment": {
                "python_version": self.environment.python_version,
                "os": self.environment.os,
                "hardware": self.environment.hardware,
                "packages": self.environment.packages,
            },
            "warnings": self.warnings,
            "limitations": self.seeding_result.limitations,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize the prepared execution context to a JSON-compatible dictionary."""
        return self.model_dump(mode="json")

    def to_json(self, indent: int | None = None) -> str:
        """Serialize the prepared execution context to a JSON string."""
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PreparedExecution":
        """Deserialize a prepared execution context from a dictionary."""
        try:
            return cls.model_validate(data)
        except Exception as exc:
            raise SerializationError(
                f"Failed to deserialize PreparedExecution from dict: {exc}"
            ) from exc

    @classmethod
    def from_json(cls, json_str: str) -> "PreparedExecution":
        """Deserialize a prepared execution context from a JSON string."""
        try:
            return cls.model_validate_json(json_str)
        except Exception as exc:
            raise SerializationError(
                f"Failed to deserialize PreparedExecution from JSON: {exc}"
            ) from exc


# Type alias for clarity across research components
RuntimeContext = PreparedExecution
