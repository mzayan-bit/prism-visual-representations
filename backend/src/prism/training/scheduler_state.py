"""Serializable state contracts for reproducible learning rate scheduling."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from prism.core.errors import SerializationError


class SchedulerState(BaseModel):
    """Immutable, serializable snapshot of learning rate scheduler state."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schedule_type: str = Field(
        description="Type identifier of the scheduler (e.g. 'constant', 'cosine')"
    )
    initial_lr: float = Field(
        gt=0.0, description="Initial base learning rate"
    )
    current_lr: float = Field(
        ge=0.0, description="Current effective learning rate"
    )
    current_step: int = Field(
        ge=0, description="Total number of discrete steps advanced"
    )
    current_epoch: int = Field(
        ge=0, description="Current epoch index (0-indexed)"
    )
    total_steps: int | None = Field(
        default=None, description="Total planned training steps horizon"
    )
    total_epochs: int | None = Field(
        default=None, description="Total planned training epochs horizon"
    )
    step_unit: str = Field(
        default="epoch", description="Stepping unit ('epoch' or 'step')"
    )
    min_lr: float = Field(
        default=0.0, ge=0.0, description="Minimum learning rate floor"
    )
    hyperparameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Schedule-specific hyperparameters (gamma, step_size, etc.)",
    )
    warmup_progress: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Normalized warmup progress (0.0 to 1.0) if applicable",
    )
    is_warmup_completed: bool | None = Field(
        default=None,
        description="True if warmup phase has completed",
    )
    history: list[float] = Field(
        default_factory=list,
        description="Full historical progression of emitted learning rates",
    )
    composed_inner_state: dict[str, Any] | None = Field(
        default=None,
        description="Serialized state of inner scheduler for composed/warmup schedules",
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize scheduler state to JSON-compatible dictionary."""
        return self.model_dump(mode="json")

    def to_json(self, indent: int | None = None) -> str:
        """Serialize scheduler state to JSON string."""
        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=True,
            ensure_ascii=True,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SchedulerState:
        """Deserialize scheduler state from dictionary."""
        try:
            return cls.model_validate(data)
        except Exception as exc:
            raise SerializationError(
                f"Failed to deserialize SchedulerState from dict: {exc}"
            ) from exc

    @classmethod
    def from_json(cls, json_str: str) -> SchedulerState:
        """Deserialize scheduler state from JSON string."""
        try:
            parsed = json.loads(json_str)
            return cls.from_dict(parsed)
        except Exception as exc:
            raise SerializationError(
                f"Failed to deserialize SchedulerState from JSON: {exc}"
            ) from exc
