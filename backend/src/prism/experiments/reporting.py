"""Execution, aggregation, and reporting for controlled architecture studies."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable
from statistics import mean, stdev
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from prism.core.enums import ModelFamily, RunStatus
from prism.core.errors import SerializationError, ValidationError
from prism.experiments.architecture import (
    ArchitectureComparisonSuite,
    ComparisonMode,
    ExperimentFactorAudit,
    ParameterCountAudit,
    SuiteStatus,
)
from prism.experiments.definitions import ExperimentDefinition
from prism.experiments.metrics import MetricRecord
from prism.representations.attention import (
    TransformerAttentionProfile,
    compute_transformer_attention_profile,
)
from prism.representations.summaries import (
    FeatureDistributionSummary,
    compute_distribution_summary,
)
from prism.training.gradient_flow import ModelGradientFlowSummary
from prism.training.results import TrainingResult


class TrainingCurvePoint(BaseModel):
    """Metrics observed at one epoch for future visualization."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    epoch: int = Field(ge=0)
    train_loss: float | None = None
    train_accuracy: float | None = None
    validation_loss: float | None = None
    validation_accuracy: float | None = None
    learning_rate: float | None = None


class TrainingCurveSummary(BaseModel):
    """Deterministically aligned epoch series from real metric records."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_id: str
    points: list[TrainingCurvePoint] = Field(default_factory=list)


class ConvergenceSummary(BaseModel):
    """Quantitative convergence descriptors without model-quality interpretation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    threshold: float | None = None
    first_epoch_reaching_threshold: int | None = None
    best_validation_accuracy: float | None = None
    best_validation_epoch: int | None = None
    final_validation_accuracy: float | None = None
    loss_improvement: float | None = None
    generalization_gap: float | None = None


class ArchitectureMetricSummary(BaseModel):
    """One auditable row in the cross-architecture metric table."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_id: str
    model_family: ModelFamily
    architecture: str
    final_train_loss: float | None = None
    final_validation_loss: float | None = None
    final_validation_accuracy: float | None = None
    test_accuracy: float | None = None
    best_validation_accuracy: float | None = None
    best_validation_epoch: int | None = None
    total_epochs: int | None = None
    parameter_count: int | None = None
    final_learning_rate: float | None = None
    training_status: RunStatus


class ArchitectureRunResult(BaseModel):
    """Serializable result envelope returned by a suite execution callback."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    experiment_id: str
    training_result: TrainingResult | None = None
    parameter_audit: ParameterCountAudit | None = None
    gradient_flow: ModelGradientFlowSummary | None = None
    representation_summary: FeatureDistributionSummary | None = None
    representation_dimension: int | None = None
    attention_profile: TransformerAttentionProfile | None = None
    attention_status: str = "not_applicable"
    error_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def status(self) -> RunStatus:
        if self.training_result is not None:
            return self.training_result.status
        return RunStatus.FAILED if self.error_message else RunStatus.PLANNED


class PairwiseArchitectureDelta(BaseModel):
    """Quantitative deltas for a pair of completed architecture runs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    left_experiment_id: str
    right_experiment_id: str
    validation_accuracy_delta: float | None = None
    validation_loss_delta: float | None = None
    parameter_count_ratio: float | None = None
    convergence_epoch_delta: int | None = None
    gradient_norm_delta: float | None = None
    representation_dimension_delta: int | None = None
    attention_comparison: dict[str, Any] | str = "not_applicable"


class ArchitectureComparisonReport(BaseModel):
    """Comprehensive, compact report for an architecture comparison suite."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    suite_id: str
    suite_fingerprint: str
    research_question: str
    comparison_mode: ComparisonMode
    generated_at: str
    factor_audit: ExperimentFactorAudit
    metric_summaries: list[ArchitectureMetricSummary] = Field(default_factory=list)
    curve_summaries: list[TrainingCurveSummary] = Field(default_factory=list)
    convergence_summaries: dict[str, ConvergenceSummary] = Field(default_factory=dict)
    run_results: list[ArchitectureRunResult] = Field(default_factory=list)
    pairwise_comparisons: list[PairwiseArchitectureDelta] = Field(default_factory=list)
    completed_experiment_ids: list[str] = Field(default_factory=list)
    failed_experiment_ids: list[str] = Field(default_factory=list)
    skipped_experiment_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def to_json(self, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArchitectureComparisonReport:
        try:
            return cls.model_validate(data)
        except Exception as exc:
            raise SerializationError(
                f"Invalid ArchitectureComparisonReport: {exc}"
            ) from exc

    @classmethod
    def from_json(cls, value: str) -> ArchitectureComparisonReport:
        try:
            return cls.from_dict(json.loads(value))
        except SerializationError:
            raise
        except Exception as exc:
            raise SerializationError(
                f"Invalid architecture report JSON: {exc}"
            ) from exc


def _metric_at(
    records: Iterable[MetricRecord], name: str, epoch: int | None = None
) -> float | None:
    matches = [
        r
        for r in records
        if r.metric_name == name and (epoch is None or r.epoch == epoch)
    ]
    if not matches:
        return None
    return matches[-1].value


def _all_records(result: TrainingResult) -> list[MetricRecord]:
    return [
        *result.metric_records,
        *[
            record
            for report in result.evaluation_reports
            for record in report.metric_records
        ],
    ]


def summarize_training_curve(
    experiment_id: str, result: TrainingResult
) -> TrainingCurveSummary:
    """Align actual training/evaluation records by epoch."""
    grouped: dict[int, dict[str, float | None]] = {}
    records = _all_records(result)
    for record in records:
        if not isinstance(record, MetricRecord) or record.epoch is None:
            continue
        key = record.metric_name
        aliases = {
            "train_top1_accuracy": "train_accuracy",
            "val_top1_accuracy": "validation_accuracy",
            "val_loss": "validation_loss",
            "train_loss": "train_loss",
            "learning_rate": "learning_rate",
        }
        if key in aliases:
            grouped.setdefault(record.epoch, {})[aliases[key]] = record.value
    points = [
        TrainingCurvePoint(epoch=epoch, **grouped[epoch]) for epoch in sorted(grouped)
    ]
    return TrainingCurveSummary(experiment_id=experiment_id, points=points)


def summarize_metrics(
    experiment: ExperimentDefinition,
    result: TrainingResult,
    parameter_count: int | None = None,
) -> ArchitectureMetricSummary:
    records = _all_records(result)
    validation_accuracy = _metric_at(records, "val_top1_accuracy")
    validation_loss = _metric_at(records, "val_loss")
    test_accuracy = _metric_at(records, "test_top1_accuracy")
    curve = summarize_training_curve(experiment.experiment_id, result)
    best = max(
        (point for point in curve.points if point.validation_accuracy is not None),
        key=lambda point: point.validation_accuracy or -math.inf,
        default=None,
    )
    return ArchitectureMetricSummary(
        experiment_id=experiment.experiment_id,
        model_family=experiment.model.family,
        architecture=experiment.model.architecture,
        final_train_loss=result.final_train_loss,
        final_validation_loss=validation_loss,
        final_validation_accuracy=validation_accuracy,
        test_accuracy=test_accuracy,
        best_validation_accuracy=best.validation_accuracy if best else None,
        best_validation_epoch=best.epoch if best else None,
        total_epochs=result.epochs_completed,
        parameter_count=parameter_count,
        final_learning_rate=result.summary_metrics.get("final_learning_rate"),
        training_status=result.status,
    )


def summarize_convergence(
    experiment_id: str,
    result: TrainingResult,
    threshold: float | None = None,
) -> ConvergenceSummary:
    curve = summarize_training_curve(experiment_id, result)
    validation_points = [p for p in curve.points if p.validation_accuracy is not None]
    best = max(
        validation_points,
        key=lambda p: p.validation_accuracy or -math.inf,
        default=None,
    )
    first = next(
        (
            p.epoch
            for p in validation_points
            if threshold is not None and (p.validation_accuracy or 0.0) >= threshold
        ),
        None,
    )
    losses = [p.train_loss for p in curve.points if p.train_loss is not None]
    final_train = losses[-1] if losses else result.final_train_loss
    first_train = losses[0] if losses else None
    train_acc = _metric_at(result.metric_records, "train_top1_accuracy")
    val_acc = best.validation_accuracy if best else None
    return ConvergenceSummary(
        threshold=threshold,
        first_epoch_reaching_threshold=first,
        best_validation_accuracy=best.validation_accuracy if best else None,
        best_validation_epoch=best.epoch if best else None,
        final_validation_accuracy=validation_points[-1].validation_accuracy
        if validation_points
        else None,
        loss_improvement=(first_train - final_train)
        if first_train is not None
        else None,
        generalization_gap=(train_acc - val_acc)
        if train_acc is not None and val_acc is not None
        else None,
    )


def summarize_model_outputs(
    model: Any,
    inputs: Any,
    family: ModelFamily,
) -> tuple[FeatureDistributionSummary, int, TransformerAttentionProfile | None, str]:
    """Summarize one model's final representation without updating parameters."""
    was_training = model.is_training
    model.eval()
    try:
        layer = (
            "cls_representation"
            if family == ModelFamily.VISION_TRANSFORMER
            else "final_hidden"
        )
        representation = model.extract_representations(inputs, layer=layer)
        summary = compute_distribution_summary(representation)
        dimension = summary.tensor_shape[-1] if summary.tensor_shape else None
        if dimension is None:
            raise ValidationError("Final representation has no feature dimension.")
        profile = None
        status = "not_applicable"
        if family == ModelFamily.VISION_TRANSFORMER:
            weights = model.get_attention_weights()
            profile = compute_transformer_attention_profile(
                weights, model_id=model.model_id
            )
            status = "available"
        return summary, dimension, profile, status
    finally:
        model.train(was_training)


def pairwise_delta(
    left: ArchitectureRunResult,
    right: ArchitectureRunResult,
    left_convergence: ConvergenceSummary | None = None,
    right_convergence: ConvergenceSummary | None = None,
) -> PairwiseArchitectureDelta:
    """Compute only meaningful pairwise deltas; unavailable values remain null."""
    left_training = left.training_result
    right_training = right.training_result
    left_val = (
        _metric_at(
            _all_records(left_training),
            "val_top1_accuracy",
        )
        if left_training
        else None
    )
    right_val = (
        _metric_at(
            _all_records(right_training),
            "val_top1_accuracy",
        )
        if right_training
        else None
    )
    left_loss = (
        _metric_at(
            _all_records(left_training),
            "val_loss",
        )
        if left_training
        else None
    )
    right_loss = (
        _metric_at(
            _all_records(right_training),
            "val_loss",
        )
        if right_training
        else None
    )
    ratio = None
    if (
        left.parameter_audit
        and right.parameter_audit
        and left.parameter_audit.total_trainable_parameters
    ):
        ratio = (
            right.parameter_audit.total_trainable_parameters
            / left.parameter_audit.total_trainable_parameters
        )
    attention: dict[str, Any] | str = "not_applicable"
    if left.attention_profile and right.attention_profile:
        attention = {
            "layer_entropy_delta": [
                b - a
                for a, b in zip(
                    left.attention_profile.layer_mean_entropies,
                    right.attention_profile.layer_mean_entropies,
                    strict=False,
                )
            ]
        }
    return PairwiseArchitectureDelta(
        left_experiment_id=left.experiment_id,
        right_experiment_id=right.experiment_id,
        validation_accuracy_delta=(right_val - left_val)
        if left_val is not None and right_val is not None
        else None,
        validation_loss_delta=(right_loss - left_loss)
        if left_loss is not None and right_loss is not None
        else None,
        parameter_count_ratio=ratio,
        convergence_epoch_delta=(
            right_convergence.best_validation_epoch
            - left_convergence.best_validation_epoch
        )
        if left_convergence
        and right_convergence
        and left_convergence.best_validation_epoch is not None
        and right_convergence.best_validation_epoch is not None
        else None,
        gradient_norm_delta=(
            right.gradient_flow.global_grad_norm_l2
            - left.gradient_flow.global_grad_norm_l2
        )
        if left.gradient_flow and right.gradient_flow
        else None,
        representation_dimension_delta=(
            right.representation_dimension - left.representation_dimension
        )
        if right.representation_dimension is not None
        and left.representation_dimension is not None
        else None,
        attention_comparison=attention,
    )


class ExperimentSuiteRunner:
    """Sequential, failure-isolating runner around existing execution machinery."""

    def __init__(self, fail_fast: bool = False) -> None:
        self.fail_fast = fail_fast

    def run(
        self,
        suite: ArchitectureComparisonSuite,
        execute_experiment: Callable[[ExperimentDefinition], ArchitectureRunResult],
        convergence_threshold: float | None = None,
    ) -> ArchitectureComparisonReport:
        audit = suite.validate_factors()
        suite.status = SuiteStatus.RUNNING
        results: list[ArchitectureRunResult] = []
        warnings = list(suite.warnings) + list(audit.warnings)
        for experiment in suite.experiment_definitions:
            try:
                result = execute_experiment(experiment)
                if result.experiment_id != experiment.experiment_id:
                    raise ValidationError(
                        "Execution result experiment ID does not match definition."
                    )
            except Exception as exc:
                result = ArchitectureRunResult(
                    experiment_id=experiment.experiment_id, error_message=str(exc)
                )
                warnings.append(f"{experiment.experiment_id} failed: {exc}")
                results.append(result)
                if self.fail_fast:
                    break
                continue
            results.append(result)
        completed = [
            r.experiment_id for r in results if r.status == RunStatus.COMPLETED
        ]
        failed = [r.experiment_id for r in results if r.status == RunStatus.FAILED]
        skipped = [
            e.experiment_id
            for e in suite.experiment_definitions
            if e.experiment_id not in {r.experiment_id for r in results}
        ]
        suite.status = (
            SuiteStatus.COMPLETED
            if not failed and not skipped
            else SuiteStatus.PARTIALLY_COMPLETED
        )
        metrics = []
        curves = []
        convergence: dict[str, ConvergenceSummary] = {}
        for result in results:
            experiment = next(
                e
                for e in suite.experiment_definitions
                if e.experiment_id == result.experiment_id
            )
            if result.training_result is not None:
                metrics.append(
                    summarize_metrics(
                        experiment,
                        result.training_result,
                        result.parameter_audit.total_trainable_parameters
                        if result.parameter_audit
                        else None,
                    )
                )
                curves.append(
                    summarize_training_curve(
                        experiment.experiment_id, result.training_result
                    )
                )
                convergence[experiment.experiment_id] = summarize_convergence(
                    experiment.experiment_id,
                    result.training_result,
                    convergence_threshold,
                )
        pairwise = [
            pairwise_delta(
                results[i],
                results[j],
                convergence.get(results[i].experiment_id),
                convergence.get(results[j].experiment_id),
            )
            for i in range(len(results))
            for j in range(i + 1, len(results))
        ]
        return ArchitectureComparisonReport(
            suite_id=suite.suite_id,
            suite_fingerprint=suite.compute_fingerprint(),
            research_question=suite.research_question,
            comparison_mode=suite.comparison_mode,
            generated_at=_utc_timestamp(),
            factor_audit=audit,
            metric_summaries=metrics,
            curve_summaries=curves,
            convergence_summaries=convergence,
            run_results=results,
            pairwise_comparisons=pairwise,
            completed_experiment_ids=completed,
            failed_experiment_ids=failed,
            skipped_experiment_ids=skipped,
            warnings=warnings,
        )


def _utc_timestamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


class SampleEfficiencyPlan(BaseModel):
    """Declarative nested data-budget study plan."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    budget_ratios: list[float]
    subset_fingerprints: dict[str, str] = Field(default_factory=dict)
    seed: int = 42

    @classmethod
    def create(
        cls,
        budget_ratios: Iterable[float],
        subset_fingerprints: dict[float, str] | None = None,
        seed: int = 42,
    ) -> SampleEfficiencyPlan:
        budgets = sorted({float(budget) for budget in budget_ratios})
        if not budgets or any(budget <= 0.0 or budget > 1.0 for budget in budgets):
            raise ValidationError("Data budgets must be in the interval (0, 1].")
        fingerprints = {
            str(budget): fingerprint
            for budget, fingerprint in (subset_fingerprints or {}).items()
        }
        return cls(budget_ratios=budgets, subset_fingerprints=fingerprints, seed=seed)


class SampleEfficiencyRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model_family: ModelFamily
    data_budget: float
    validation_accuracy: float | None = None
    test_accuracy: float | None = None
    representation_dimension: int | None = None
    parameter_count: int | None = None


class SampleEfficiencySummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    records: list[SampleEfficiencyRecord] = Field(default_factory=list)
    missing_budgets: list[float] = Field(default_factory=list)

    @classmethod
    def from_records(
        cls, plan: SampleEfficiencyPlan, records: Iterable[SampleEfficiencyRecord]
    ) -> SampleEfficiencySummary:
        ordered = sorted(
            records, key=lambda item: (item.model_family.value, item.data_budget)
        )
        present = {item.data_budget for item in ordered}
        return cls(
            records=ordered,
            missing_budgets=[
                budget for budget in plan.budget_ratios if budget not in present
            ],
        )


class RepeatedSeedPlan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    seeds: list[int]

    @field_validator("seeds")
    @classmethod
    def validate_seeds(cls, value: list[int]) -> list[int]:
        if not value or len(value) != len(set(value)):
            raise ValueError("Repeated-seed plan requires unique seeds.")
        return sorted(value)


class RepeatedMetricAggregate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    metric_name: str
    mean: float | None = None
    standard_deviation: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    run_count: int = Field(ge=0)
    single_run_std_policy: str = "not_applicable"


def aggregate_repeated_metric(
    metric_name: str, values: Iterable[float | None]
) -> RepeatedMetricAggregate:
    finite = [
        float(value) for value in values if value is not None and math.isfinite(value)
    ]
    if not finite:
        return RepeatedMetricAggregate(metric_name=metric_name, run_count=0)
    return RepeatedMetricAggregate(
        metric_name=metric_name,
        mean=mean(finite),
        standard_deviation=stdev(finite) if len(finite) > 1 else 0.0,
        minimum=min(finite),
        maximum=max(finite),
        run_count=len(finite),
        single_run_std_policy="zero_for_single_observation"
        if len(finite) == 1
        else "sample_standard_deviation",
    )
