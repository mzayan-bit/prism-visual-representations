"""Experiment run models and execution lifecycle tracking."""

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from prism.artifacts.contracts import ArtifactReference
from prism.core.enums import RunStatus
from prism.core.errors import SerializationError
from prism.core.identifiers import ensure_valid_identifier
from prism.core.metadata import CodeRevisionMetadata, EnvironmentMetadata
from prism.experiments.lifecycle import validate_transition
from prism.experiments.metrics import MetricRecord
from prism.experiments.reproducibility import ReproducibilityConfiguration


def _utc_now() -> datetime:
    """Return timezone-aware current UTC datetime."""
    return datetime.now(timezone.utc)


class FailureInfo(BaseModel):
    """Structured details capturing the cause and context of a run failure."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    error_type: str = Field(description="Class name or error category")
    error_message: str = Field(description="Human-readable exception message")
    traceback: str | None = Field(
        default=None,
        description="Formatted exception stack trace string if captured",
    )
    occurred_at: datetime = Field(
        default_factory=_utc_now,
        description="UTC timestamp when failure occurred",
    )


class ExperimentRun(BaseModel):
    """Represents an execution attempt of an experiment definition."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(description="Unique run identifier (e.g. 'run-a1b2c3d4e5f6')")
    experiment_id: str = Field(description="Referenced ExperimentDefinition identifier")
    status: RunStatus = Field(
        default=RunStatus.PLANNED,
        description="Current lifecycle status of the run",
    )
    created_at: datetime = Field(
        default_factory=_utc_now,
        description="UTC timestamp when run was planned/created",
    )
    started_at: datetime | None = Field(
        default=None,
        description="UTC timestamp when active execution started",
    )
    completed_at: datetime | None = Field(
        default=None,
        description="UTC timestamp when execution concluded",
    )
    configuration_fingerprint: str | None = Field(
        default=None,
        description="SHA-256 fingerprint snapshot of executing config",
    )
    reproducibility: ReproducibilityConfiguration | None = Field(
        default=None,
        description="Reproducibility settings used for this execution",
    )
    code_revision: CodeRevisionMetadata | None = Field(
        default=None,
        description="Code revision details at time of execution",
    )
    environment: EnvironmentMetadata | None = Field(
        default=None,
        description="Host system runtime environment details",
    )
    failure_info: FailureInfo | None = Field(
        default=None,
        description="Failure telemetry if run ended in FAILED status",
    )
    metric_records: list[MetricRecord] = Field(
        default_factory=list,
        description="Chronological log of scalar metric measurements",
    )
    artifact_references: list[ArtifactReference] = Field(
        default_factory=list,
        description="Manifest of output artifacts produced by this run",
    )
    summary_metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Consolidated final scalar summary metrics",
    )
    tags: list[str] = Field(default_factory=list, description="Categorization tags")
    notes: str = Field(default="", description="Operator or researcher notes")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional execution metadata",
    )

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, v: str) -> str:
        return ensure_valid_identifier(v, field_name="run_id")

    @field_validator("experiment_id")
    @classmethod
    def validate_exp_id(cls, v: str) -> str:
        return ensure_valid_identifier(v, field_name="experiment_id")

    def transition_to(
        self, target_status: RunStatus, reason: str | None = None
    ) -> None:
        """Validate and apply a lifecycle state transition."""
        validate_transition(
            self.status,
            target_status,
            run_id=self.run_id,
            reason=reason,
        )
        self.status = target_status

    def start(self) -> None:
        """Transition run to RUNNING and record started_at timestamp."""
        self.transition_to(RunStatus.RUNNING)
        self.started_at = _utc_now()

    def complete(self, summary_metrics: dict[str, float] | None = None) -> None:
        """Transition run to COMPLETED and record final metrics and timestamp."""
        self.transition_to(RunStatus.COMPLETED)
        self.completed_at = _utc_now()
        if summary_metrics:
            self.summary_metrics.update(summary_metrics)

    def fail(
        self,
        error_type: str,
        error_message: str,
        traceback: str | None = None,
    ) -> None:
        """Transition run to FAILED and record failure details."""
        self.transition_to(RunStatus.FAILED)
        self.completed_at = _utc_now()
        self.failure_info = FailureInfo(
            error_type=error_type,
            error_message=error_message,
            traceback=traceback,
            occurred_at=self.completed_at,
        )

    def cancel(self, reason: str = "") -> None:
        """Transition run to CANCELLED and record timestamp."""
        self.transition_to(RunStatus.CANCELLED, reason=reason)
        self.completed_at = _utc_now()
        if reason:
            self.notes = f"{self.notes} [Cancelled: {reason}]".strip()

    def add_metric(self, record: MetricRecord) -> None:
        """Append a metric record to the run log."""
        self.metric_records.append(record)

    def add_artifact(self, artifact: ArtifactReference) -> None:
        """Register an output artifact with this run."""
        self.artifact_references.append(artifact)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the experiment run to a JSON-compatible dictionary."""
        return self.model_dump(mode="json")

    def to_json(self, indent: int | None = None) -> str:
        """Serialize the experiment run to a JSON string."""
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentRun":
        """Deserialize an experiment run from a dictionary."""
        try:
            return cls.model_validate(data)
        except Exception as exc:
            raise SerializationError(
                f"Failed to deserialize ExperimentRun from dict: {exc}"
            ) from exc

    @classmethod
    def from_json(cls, json_str: str) -> "ExperimentRun":
        """Deserialize an experiment run from a JSON string."""
        try:
            return cls.model_validate_json(json_str)
        except Exception as exc:
            raise SerializationError(
                f"Failed to deserialize ExperimentRun from JSON: {exc}"
            ) from exc
