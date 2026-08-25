"""Structured evaluation report schemas summarizing experiment findings."""

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from prism.artifacts.contracts import ArtifactReference
from prism.core.errors import SerializationError
from prism.core.identifiers import ensure_valid_identifier
from prism.evaluation.configuration import EvaluationConfiguration
from prism.experiments.metrics import MetricRecord


def _utc_now() -> datetime:
    """Return timezone-aware current UTC datetime."""
    return datetime.now(timezone.utc)


class EvaluationReport(BaseModel):
    """Immutable report consolidating metrics, config, and artifacts for a run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    report_id: str = Field(
        description="Unique report identifier (e.g. 'rep-a1b2c3d4e5f6')"
    )
    experiment_id: str = Field(description="Referenced ExperimentDefinition identifier")
    run_id: str = Field(description="Referenced ExperimentRun identifier")
    evaluation_config: EvaluationConfiguration = Field(
        description="Configuration governing the evaluation protocol"
    )
    metric_records: list[MetricRecord] = Field(
        description="Detailed metric records collected during evaluation"
    )
    artifact_references: list[ArtifactReference] = Field(
        default_factory=list,
        description="Artifacts associated with this evaluation",
    )
    summary_metrics: dict[str, float] = Field(
        default_factory=dict,
        description="High-level scalar metric summary (e.g. {'top1_acc': 0.85})",
    )
    generated_at: datetime = Field(
        default_factory=_utc_now,
        description="UTC timestamp when the report was compiled",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context or qualitative analysis notes",
    )

    @field_validator("report_id")
    @classmethod
    def validate_report_id(cls, v: str) -> str:
        return ensure_valid_identifier(v, field_name="report_id")

    @field_validator("experiment_id")
    @classmethod
    def validate_exp_id(cls, v: str) -> str:
        return ensure_valid_identifier(v, field_name="experiment_id")

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, v: str) -> str:
        return ensure_valid_identifier(v, field_name="run_id")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the evaluation report to a JSON-compatible dictionary."""
        return self.model_dump(mode="json")

    def to_json(self, indent: int | None = None) -> str:
        """Serialize the evaluation report to a JSON string."""
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationReport":
        """Deserialize an evaluation report from a dictionary."""
        try:
            return cls.model_validate(data)
        except Exception as exc:
            raise SerializationError(
                f"Failed to deserialize EvaluationReport from dict: {exc}"
            ) from exc

    @classmethod
    def from_json(cls, json_str: str) -> "EvaluationReport":
        """Deserialize an evaluation report from a JSON string."""
        try:
            return cls.model_validate_json(json_str)
        except Exception as exc:
            raise SerializationError(
                f"Failed to deserialize EvaluationReport from JSON: {exc}"
            ) from exc
