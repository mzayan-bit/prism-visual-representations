"""Controlled comparison contracts for scientific evaluation of model variants."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from prism.core.errors import SerializationError
from prism.core.identifiers import ensure_valid_identifier


class ControlledComparison(BaseModel):
    """Declarative specification for controlled model comparisons."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    comparison_id: str = Field(
        description="Unique identifier for the comparison (e.g. 'comp-linear-vs-mlp')"
    )
    name: str = Field(description="Human-readable comparison name")
    description: str = Field(
        default="", description="Scientific rationale and hypothesis"
    )
    baseline_experiment_id: str = Field(
        description="Identifier of the baseline experiment"
    )
    candidate_experiment_id: str = Field(
        description="Identifier of the candidate experiment"
    )
    varied_factors: dict[str, Any] = Field(
        description="Explicit mapping of factors that differed between experiments"
    )
    fixed_factors: dict[str, Any] = Field(
        description="Explicit mapping of strictly controlled invariant factors"
    )
    dataset_fingerprint: str = Field(
        description="SHA-256 fingerprint of underlying dataset manifest"
    )
    partition_fingerprint: str | None = Field(
        default=None,
        description="SHA-256 fingerprint of fixed partition split manifest",
    )
    subset_fingerprint: str | None = Field(
        default=None,
        description="SHA-256 fingerprint of nested budget subset manifest",
    )
    seed: int = Field(description="Reproducibility seed shared across comparison")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional scientific annotations"
    )

    @field_validator(
        "comparison_id", "baseline_experiment_id", "candidate_experiment_id"
    )
    @classmethod
    def validate_ids(cls, v: str) -> str:
        return ensure_valid_identifier(v)

    def compute_fingerprint(self) -> str:
        """Compute deterministic SHA-256 fingerprint of the comparison."""
        payload = {
            "comparison_id": self.comparison_id,
            "baseline_experiment_id": self.baseline_experiment_id,
            "candidate_experiment_id": self.candidate_experiment_id,
            "varied_factors": self.varied_factors,
            "fixed_factors": self.fixed_factors,
            "dataset_fingerprint": self.dataset_fingerprint,
            "partition_fingerprint": self.partition_fingerprint,
            "subset_fingerprint": self.subset_fingerprint,
            "seed": self.seed,
        }
        encoded = json.dumps(
            payload, sort_keys=True, ensure_ascii=True
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Serialize comparison to dictionary."""
        return self.model_dump(mode="json")

    def to_json(self, indent: int | None = None) -> str:
        """Serialize comparison to JSON string."""
        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=True,
            ensure_ascii=True,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ControlledComparison:
        """Deserialize comparison from dictionary."""
        try:
            return cls.model_validate(data)
        except Exception as exc:
            raise SerializationError(
                f"Failed to deserialize ControlledComparison from dict: {exc}"
            ) from exc

    @classmethod
    def from_json(cls, json_str: str) -> ControlledComparison:
        """Deserialize comparison from JSON string."""
        try:
            parsed = json.loads(json_str)
            return cls.from_dict(parsed)
        except Exception as exc:
            raise SerializationError(
                f"Failed to deserialize ControlledComparison from JSON: {exc}"
            ) from exc


def create_normalization_comparison(
    comparison_id: str,
    name: str,
    baseline_experiment_id: str,
    candidate_experiment_id: str,
    dataset_fingerprint: str,
    seed: int = 42,
    normalization_type: str = "batch_norm",
    norm_eps: float = 1e-5,
    norm_momentum: float = 0.1,
    norm_affine: bool = True,
    fixed_factors: dict[str, Any] | None = None,
    partition_fingerprint: str | None = None,
    subset_fingerprint: str | None = None,
    description: str = "Controlled comparison isolating normalization effect.",
) -> ControlledComparison:
    """Helper creating an auditable ControlledComparison for normalization studies."""
    varied = {
        "normalization": {
            "baseline": "none",
            "candidate": normalization_type,
        },
        "norm_eps": {
            "baseline": None,
            "candidate": norm_eps,
        },
        "norm_momentum": {
            "baseline": None,
            "candidate": norm_momentum,
        },
        "norm_affine": {
            "baseline": None,
            "candidate": norm_affine,
        },
    }

    base_fixed: dict[str, Any] = {
        "dataset_fingerprint": dataset_fingerprint,
        "seed": seed,
    }
    if fixed_factors:
        base_fixed.update(fixed_factors)

    return ControlledComparison(
        comparison_id=comparison_id,
        name=name,
        description=description,
        baseline_experiment_id=baseline_experiment_id,
        candidate_experiment_id=candidate_experiment_id,
        varied_factors=varied,
        fixed_factors=base_fixed,
        dataset_fingerprint=dataset_fingerprint,
        partition_fingerprint=partition_fingerprint,
        subset_fingerprint=subset_fingerprint,
        seed=seed,
    )
