"""Training result schemas capturing execution summaries."""

from __future__ import annotations

import json
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from prism.core.enums import RunStatus
from prism.core.errors import SerializationError
from prism.core.identifiers import ensure_valid_identifier
from prism.evaluation.reports import EvaluationReport
from prism.experiments.hashing import compute_configuration_fingerprint


class TrainingResult(BaseModel):
    """Immutable summary record produced at the conclusion of model training."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(description="Referenced ExperimentRun identifier")
    experiment_id: str = Field(description="Referenced ExperimentDefinition identifier")
    status: RunStatus = Field(description="Final execution lifecycle status")
    epochs_completed: int = Field(
        ge=0,
        description="Total training epochs successfully executed",
    )
    total_batches: int = Field(
        ge=0,
        description="Total number of batches processed during training",
    )
    total_examples: int = Field(
        ge=0,
        description="Total count of individual sample forward passes",
    )
    final_train_loss: float = Field(
        description="Final training epoch mean loss",
    )
    final_train_accuracy: float = Field(
        ge=0.0,
        le=1.0,
        description="Final training epoch classification accuracy",
    )
    evaluation_reports: list[EvaluationReport] = Field(
        default_factory=list,
        description="Evaluation reports compiled during or after training",
    )
    summary_metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Consolidated scalar metrics dictionary",
    )
    duration_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Wall-clock duration of active training in seconds",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Warnings or notices recorded during training",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Auxiliary training metadata",
    )
    schema_version: str = Field(
        default="1.0.0",
        description="Schema contract version",
    )

    @field_validator("run_id")
    @classmethod
    def validate_run_id_field(cls, v: str) -> str:
        return ensure_valid_identifier(v, field_name="run_id")

    @field_validator("experiment_id")
    @classmethod
    def validate_exp_id_field(cls, v: str) -> str:
        return ensure_valid_identifier(v, field_name="experiment_id")

    def compute_fingerprint(self) -> str:
        """Compute deterministic SHA-256 fingerprint of the result summary."""
        return compute_configuration_fingerprint(self)

    def to_dict(self) -> dict[str, Any]:
        """Convert training result to dictionary."""
        return self.model_dump(mode="json")

    def to_json(self, indent: int | None = None) -> str:
        """Convert training result to JSON string."""
        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=True,
            ensure_ascii=True,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingResult:
        """Create training result from dictionary."""
        try:
            return cls.model_validate(data)
        except Exception as exc:
            raise SerializationError(
                f"Failed to deserialize TrainingResult from dict: {exc}"
            ) from exc

    @classmethod
    def from_json(cls, json_str: str) -> TrainingResult:
        """Create training result from JSON string."""
        try:
            parsed = json.loads(json_str)
            return cls.from_dict(parsed)
        except Exception as exc:
            raise SerializationError(
                f"Failed to deserialize TrainingResult from JSON: {exc}"
            ) from exc
