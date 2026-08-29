"""Controlled architecture experiment suites and scientific factor audits.

This module deliberately orchestrates existing experiment definitions.  It does not
contain model mathematics or a second training configuration system.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from prism.core.enums import ModelFamily
from prism.core.errors import SerializationError, ValidationError
from prism.core.identifiers import ensure_valid_identifier
from prism.experiments.definitions import ExperimentDefinition


class ComparisonMode(str, Enum):
    """Policy governing which differences an architecture suite permits."""

    STRICT_CONTROLLED = "strict_controlled"
    ARCHITECTURE_APPROPRIATE = "architecture_appropriate"


class SuiteStatus(str, Enum):
    """Lifecycle states for a planned architecture comparison suite."""

    PLANNED = "planned"
    VALIDATED = "validated"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIALLY_COMPLETED = "partially_completed"
    FAILED = "failed"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _stable(value: Any) -> Any:
    """Convert typed model data to deterministic JSON-compatible values."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {
            str(k): _stable(v)
            for k, v in sorted(value.items(), key=lambda x: str(x[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_stable(v) for v in value]
    return value


class ExperimentFactorAudit(BaseModel):
    """Structured result of comparing typed factors across definitions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_ids: list[str]
    constant_factors: dict[str, Any] = Field(default_factory=dict)
    varied_factors: dict[str, dict[str, Any]] = Field(default_factory=dict)
    unexpected_differences: dict[str, dict[str, Any]] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    is_strictly_controlled: bool

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def to_json(self, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentFactorAudit:
        try:
            return cls.model_validate(data)
        except Exception as exc:
            raise SerializationError(f"Invalid ExperimentFactorAudit: {exc}") from exc

    @classmethod
    def from_json(cls, value: str) -> ExperimentFactorAudit:
        try:
            return cls.from_dict(json.loads(value))
        except SerializationError:
            raise
        except Exception as exc:
            raise SerializationError(f"Invalid factor audit JSON: {exc}") from exc


class ParameterCountAudit(BaseModel):
    """Exact trainable parameter count derived from a model parameter mapping."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str
    total_trainable_parameters: int = Field(ge=0)
    component_counts: dict[str, int] = Field(default_factory=dict)
    parameter_shapes: dict[str, tuple[int, ...]] = Field(default_factory=dict)
    parameter_matched: bool = False

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def to_json(self, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


class ArchitectureComparisonSuite(BaseModel):
    """Serializable plan for a controlled CNN/ResNet/ViT comparison."""

    model_config = ConfigDict(extra="forbid")

    suite_id: str
    suite_name: str
    research_question: str
    experiment_definitions: list[ExperimentDefinition]
    comparison_mode: ComparisonMode = ComparisonMode.STRICT_CONTROLLED
    controlled_factors: list[str] = Field(default_factory=list)
    intentionally_varied_factors: list[str] = Field(
        default_factory=lambda: ["model.family", "model.architecture"]
    )
    dataset_identity: str | None = None
    partition_identity: str | None = None
    subset_identity: str | None = None
    seed_policy: dict[str, Any] = Field(default_factory=dict)
    training_budget_policy: dict[str, Any] = Field(default_factory=dict)
    evaluation_policy: dict[str, Any] = Field(default_factory=dict)
    status: SuiteStatus = SuiteStatus.PLANNED
    warnings: list[str] = Field(default_factory=list)
    required_families: list[ModelFamily] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utc_now)

    @field_validator("suite_id")
    @classmethod
    def validate_suite_id(cls, value: str) -> str:
        return ensure_valid_identifier(value, field_name="suite_id")

    @field_validator("suite_name", "research_question")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Suite text fields cannot be empty.")
        return value.strip()

    @model_validator(mode="after")
    def validate_definitions(self) -> ArchitectureComparisonSuite:
        ids = [item.experiment_id for item in self.experiment_definitions]
        if len(ids) != len(set(ids)):
            raise ValidationError(f"Experiment IDs must be unique, got {ids}.")
        if not self.experiment_definitions:
            raise ValidationError(
                "Architecture suite requires at least one experiment."
            )
        missing = [
            family.value
            for family in self.required_families
            if family not in [e.model.family for e in self.experiment_definitions]
        ]
        if missing:
            self.warnings.append(
                f"Required architecture families are absent: {missing}"
            )
        return self

    def compute_fingerprint(self) -> str:
        payload = self.model_dump(
            mode="json",
            exclude={"created_at", "status", "warnings", "experiment_definitions"},
        )
        payload["experiment_definitions"] = [
            experiment.compute_fingerprint()
            for experiment in self.experiment_definitions
        ]
        encoded = json.dumps(
            _stable(payload), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def audit_factors(self) -> ExperimentFactorAudit:
        return audit_experiment_factors(
            self.experiment_definitions,
            declared_varied_factors=self.intentionally_varied_factors,
            comparison_mode=self.comparison_mode,
        )

    def validate_factors(self) -> ExperimentFactorAudit:
        audit = self.audit_factors()
        if (
            self.comparison_mode == ComparisonMode.STRICT_CONTROLLED
            and audit.unexpected_differences
        ):
            differences = ", ".join(sorted(audit.unexpected_differences))
            raise ValidationError(
                f"Strict architecture suite has undeclared differences: {differences}"
            )
        return audit

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def to_json(self, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArchitectureComparisonSuite:
        try:
            return cls.model_validate(data)
        except Exception as exc:
            raise SerializationError(
                f"Invalid ArchitectureComparisonSuite: {exc}"
            ) from exc

    @classmethod
    def from_json(cls, value: str) -> ArchitectureComparisonSuite:
        try:
            return cls.from_dict(json.loads(value))
        except SerializationError:
            raise
        except Exception as exc:
            raise SerializationError(f"Invalid suite JSON: {exc}") from exc


def _factor_map(experiment: ExperimentDefinition) -> dict[str, Any]:
    """Return typed, scientifically meaningful factors for one definition."""
    data = experiment.dataset
    controlled = data.controlled_data
    return {
        "dataset.dataset_id": data.dataset_id,
        "dataset.version": data.version,
        "dataset.fingerprint": data.fingerprint,
        "dataset.canonical_manifest_fingerprint": (
            controlled.canonical_manifest_fingerprint if controlled else None
        ),
        "dataset.partition_fingerprint": controlled.partition_manifest_fingerprint
        if controlled
        else None,
        "dataset.subset_fingerprint": controlled.subset_manifest_fingerprint
        if controlled
        else None,
        "dataset.budget_ratio": controlled.budget_ratio
        if controlled
        else data.subset_fraction,
        "dataset.preprocessing": data.preprocessing.model_dump(mode="json"),
        "dataset.augmentation": data.augmentation.model_dump(mode="json"),
        "reproducibility.seed": experiment.reproducibility.seed,
        "reproducibility.deterministic": experiment.reproducibility.deterministic,
        "model.family": experiment.model.family,
        "model.architecture": experiment.model.architecture,
        "model.input_shape": experiment.model.input_shape,
        "model.num_classes": experiment.model.num_classes,
        "model.initialization": experiment.model.initialization,
        "model.hyperparameters": experiment.model.hyperparameters,
        "model.metadata": experiment.model.metadata,
        "training.epochs": experiment.training.epochs,
        "training.batch_size": experiment.training.batch_size,
        "training.optimizer": experiment.training.optimizer.model_dump(mode="json"),
        "training.scheduler": experiment.training.scheduler.model_dump(mode="json"),
        "training.weight_decay": experiment.training.optimizer.weight_decay,
        "training.dropout": experiment.model.hyperparameters.get("dropout"),
        "evaluation.metrics": [
            m.model_dump(mode="json") for m in experiment.evaluation.metrics
        ],
        "evaluation.target_splits": experiment.evaluation.target_splits,
        "evaluation.batch_size": experiment.evaluation.batch_size,
    }


def audit_experiment_factors(
    experiments: Iterable[ExperimentDefinition],
    declared_varied_factors: Iterable[str] = (),
    comparison_mode: ComparisonMode = ComparisonMode.STRICT_CONTROLLED,
) -> ExperimentFactorAudit:
    """Compare typed factors and identify undeclared differences."""
    items = list(experiments)
    if not items:
        raise ValidationError("At least one experiment is required for factor audit.")
    maps = [_factor_map(item) for item in items]
    declared = set(declared_varied_factors)
    aliases = {
        "dataset": {"dataset.dataset_id", "dataset.version", "dataset.fingerprint"},
        "model_family": {"model.family"},
        "model_architecture": {"model.architecture"},
        "architecture": {"model.architecture"},
        "batch_size": {"training.batch_size", "evaluation.batch_size"},
        "optimizer": {"training.optimizer"},
        "scheduler": {"training.scheduler"},
        "seed": {"reproducibility.seed"},
        "deterministic": {"reproducibility.deterministic"},
        "preprocessing": {"dataset.preprocessing"},
    }
    declared_paths = set(declared)
    for alias in declared:
        declared_paths.update(aliases.get(alias, {alias}))
    all_keys = sorted(set().union(*(mapping.keys() for mapping in maps)))
    constant: dict[str, Any] = {}
    varied: dict[str, dict[str, Any]] = {}
    unexpected: dict[str, dict[str, Any]] = {}
    for key in all_keys:
        values = [_stable(mapping.get(key)) for mapping in maps]
        if all(value == values[0] for value in values[1:]):
            constant[key] = values[0]
            continue
        record = {
            item.experiment_id: value for item, value in zip(items, values, strict=True)
        }
        varied[key] = record
        if key not in declared_paths and key.split(".", 1)[0] not in declared_paths:
            unexpected[key] = record
    warnings: list[str] = []
    if comparison_mode == ComparisonMode.ARCHITECTURE_APPROPRIATE and unexpected:
        warnings.append(
            "Architecture-appropriate comparison contains explicitly "
            "auditable differences."
        )
    return ExperimentFactorAudit(
        experiment_ids=[item.experiment_id for item in items],
        constant_factors=constant,
        varied_factors=varied,
        unexpected_differences=unexpected,
        warnings=warnings,
        is_strictly_controlled=not unexpected,
    )


def _shape_and_count(value: Any) -> tuple[tuple[int, ...], int]:
    if isinstance(value, (list, tuple)):
        if not value:
            return (0,), 0
        child_shapes = [_shape_and_count(child) for child in value]
        shape = (len(value),) + child_shapes[0][0]
        if any(child_shape != child_shapes[0][0] for child_shape, _ in child_shapes):
            raise ValidationError("Ragged parameter tensors cannot be counted exactly.")
        return shape, sum(count for _, count in child_shapes)
    return (), 1


def count_model_parameters(
    model: Any,
    parameter_matched: bool = False,
) -> ParameterCountAudit:
    """Count every trainable scalar in an existing model parameter mapping."""
    if model is None or not hasattr(model, "get_parameters"):
        raise ValidationError("A model exposing get_parameters() is required.")
    params = model.get_parameters()
    if not isinstance(params, dict):
        raise ValidationError("get_parameters() must return a dictionary.")
    shapes: dict[str, tuple[int, ...]] = {}
    components: dict[str, int] = {}
    total = 0
    for name, value in sorted(params.items()):
        shape, count = _shape_and_count(value)
        shapes[name] = shape
        total += count
        component = name.split(".")[0].split("_")[0] or "model"
        components[component] = components.get(component, 0) + count
    model_id = str(
        getattr(
            model,
            "model_id",
            getattr(getattr(model, "spec", None), "model_id", "unknown"),
        )
    )
    return ParameterCountAudit(
        model_id=model_id,
        total_trainable_parameters=total,
        component_counts=components,
        parameter_shapes=shapes,
        parameter_matched=parameter_matched,
    )


def create_architecture_comparison_suite(
    suite_id: str,
    suite_name: str,
    research_question: str,
    experiments: Iterable[ExperimentDefinition],
    comparison_mode: ComparisonMode = ComparisonMode.STRICT_CONTROLLED,
    intentionally_varied_factors: Iterable[str] = (
        "model.family",
        "model.architecture",
        "model.hyperparameters",
    ),
    required_families: Iterable[ModelFamily] = (
        ModelFamily.CNN,
        ModelFamily.RESNET,
        ModelFamily.VISION_TRANSFORMER,
    ),
    **metadata: Any,
) -> ArchitectureComparisonSuite:
    """Construct a suite while preserving exact experiment specifications."""
    definitions = list(experiments)
    if not definitions:
        raise ValidationError("Architecture comparison suite requires experiments.")
    first = definitions[0]
    controlled = first.dataset.controlled_data
    suite = ArchitectureComparisonSuite(
        suite_id=suite_id,
        suite_name=suite_name,
        research_question=research_question,
        experiment_definitions=definitions,
        comparison_mode=comparison_mode,
        intentionally_varied_factors=list(intentionally_varied_factors),
        required_families=list(required_families),
        dataset_identity=first.dataset.dataset_id,
        partition_identity=controlled.partition_manifest_fingerprint
        if controlled
        else None,
        subset_identity=controlled.subset_manifest_fingerprint if controlled else None,
        seed_policy={
            "seeds": sorted({item.reproducibility.seed for item in definitions})
        },
        training_budget_policy={
            "epochs": sorted({item.training.epochs for item in definitions})
        },
        evaluation_policy={
            "target_splits": sorted(
                {
                    split
                    for item in definitions
                    for split in item.evaluation.target_splits
                }
            )
        },
        metadata=metadata,
    )
    suite.controlled_factors = sorted(suite.audit_factors().constant_factors)
    return suite


# Repository-friendly aliases for generic suite consumers.
ExperimentSuite = ArchitectureComparisonSuite
ControlledExperimentSuite = ArchitectureComparisonSuite
