"""Robustness evaluation engine, severity curves, failure taxonomy, and reports."""

from __future__ import annotations

import json
import math
from enum import Enum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from prism.core.errors import SerializationError
from prism.core.identifiers import ensure_valid_identifier
from prism.data.materialized import MaterializedDataset
from prism.models.base import BaseVisionModel
from prism.representations.geometry import (
    DistanceMetric,
    RepresentationDataset,
    SpatialVectorizationPolicy,
    VectorNormalizationPolicy,
)
from prism.robustness.attention_drift import (
    AttentionDriftSummary,
    compute_vit_attention_drift,
)
from prism.robustness.corruptions import (
    CorruptedDatasetView,
    CorruptionSpecification,
    CorruptionType,
)
from prism.robustness.drift import (
    RepresentationDriftSummary,
    SampleRepresentationDrift,
    compute_representation_drift,
)
from prism.robustness.geometry_drift import (
    GeometryDriftReport,
    compute_geometry_drift,
)
from prism.training.loss import SoftmaxCrossEntropyLoss, compute_accuracy


class RobustnessFailureCategory(str, Enum):
    """Observed failure taxonomy categories under distribution shift."""

    PREDICTION_FLIP = "prediction_flip"
    CLEAN_CORRECT_TO_CORRUPTED_WRONG = "clean_correct_to_corrupted_wrong"
    HIGH_REPRESENTATION_DRIFT = "high_representation_drift"
    NEIGHBORHOOD_BREAKDOWN = "neighborhood_breakdown"
    CLASS_CENTROID_DRIFT = "class_centroid_drift"
    ATTENTION_PATTERN_SHIFT = "attention_pattern_shift"


class RobustnessFailureRecord(BaseModel):
    """Categorized failure instance observed under corruption."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_id: str = Field(description="Sample identifier")
    category: RobustnessFailureCategory = Field(description="Taxonomy category")
    description: str = Field(description="Human-readable description of failure")
    severity: int = Field(ge=1, le=5, description="Observed severity level")
    metrics: dict[str, float] = Field(
        default_factory=dict, description="Quantitative failure metrics"
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert record to dictionary."""
        return self.model_dump(mode="json")


class CorruptionEvaluationSummary(BaseModel):
    """Quantitative performance and drift summary for a corruption and severity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    corruption_type: CorruptionType = Field(description="Corruption family")
    severity: int = Field(ge=1, le=5, description="Severity level")
    num_samples: int = Field(ge=0, description="Evaluated sample count")
    clean_accuracy: float = Field(ge=0.0, le=1.0, description="Clean top-1 accuracy")
    corrupted_accuracy: float = Field(
        ge=0.0, le=1.0, description="Corrupted top-1 accuracy"
    )
    absolute_accuracy_drop: float = Field(
        description="Clean accuracy minus corrupted accuracy"
    )
    relative_accuracy_drop: float = Field(
        description="Relative drop: (clean_acc - corrupt_acc) / (clean_acc + 1e-12)"
    )
    clean_loss: float = Field(ge=0.0, description="Clean mean cross-entropy loss")
    corrupted_loss: float = Field(
        ge=0.0, description="Corrupted mean cross-entropy loss"
    )
    loss_increase: float = Field(description="Corrupted loss minus clean loss")
    predictions_changed_count: int = Field(
        ge=0, description="Number of samples whose prediction changed"
    )
    prediction_consistency_fraction: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of samples with unchanged prediction",
    )
    representation_drift: RepresentationDriftSummary = Field(
        description="Representation drift summary"
    )
    geometry_drift: GeometryDriftReport = Field(
        description="Geometric and manifold drift report"
    )
    attention_drift: AttentionDriftSummary | None = Field(
        default=None, description="ViT attention drift summary if applicable"
    )
    failure_counts_by_category: dict[str, int] = Field(
        default_factory=dict,
        description="Count of flagged failures grouped by taxonomy",
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert summary to dictionary."""
        return self.model_dump(mode="json")


class CorruptionSeverityCurve(BaseModel):
    """Trajectory of performance and drift across increasing corruption severities."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    corruption_type: CorruptionType = Field(description="Corruption family")
    severities: list[int] = Field(
        description="Ordered evaluated severity levels (e.g. [1, 2, 3, 4, 5])"
    )
    clean_accuracy: float = Field(ge=0.0, le=1.0, description="Clean baseline accuracy")
    accuracy_trajectory: list[float] = Field(
        description="Corrupted accuracy at each severity level"
    )
    loss_trajectory: list[float] = Field(
        description="Corrupted loss at each severity level"
    )
    representation_drift_trajectory: list[float] = Field(
        description="Mean Euclidean representation drift at each severity level"
    )
    neighbor_consistency_trajectory: list[float] = Field(
        description="k-NN label consistency at each severity level"
    )
    centroid_displacement_trajectory: list[float] = Field(
        description="Mean class centroid displacement at each severity level"
    )
    total_accuracy_drop: float = Field(
        description="Accuracy drop from clean baseline to maximum evaluated severity"
    )
    mean_accuracy: float = Field(
        ge=0.0, le=1.0, description="Average accuracy across all evaluated severities"
    )
    area_under_curve: float = Field(
        description="Normalized trapezoidal area under accuracy severity curve [0, 1]"
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert curve to dictionary."""
        return self.model_dump(mode="json")


class CorruptionSuite(BaseModel):
    """Declarative specification of a multi-corruption robustness experiment suite."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    suite_id: str = Field(description="Unique corruption suite identifier")
    name: str = Field(description="Descriptive suite title")
    corruption_types: list[CorruptionType] = Field(
        description="List of corruption families to evaluate"
    )
    severities: list[int] = Field(
        default_factory=lambda: [1, 2, 3, 4, 5],
        description="Severity levels to test per corruption",
    )
    eval_split: str = Field(default="test", description="Evaluation dataset split")
    layer_name: str = Field(
        default="final_hidden",
        description="Model layer to extract representations from",
    )
    seed: int = Field(default=42, description="Random seed for corruptions")
    k_neighbors: int = Field(
        default=5, ge=1, description="Number of nearest neighbors for geometry"
    )
    pca_components: int = Field(
        default=2, ge=1, description="Number of PCA components for shared basis"
    )

    @field_validator("suite_id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        return ensure_valid_identifier(v, field_name="suite_id")


class RobustnessExperimentReport(BaseModel):
    """Report covering clean baselines, corruptions, drift, and failures."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_id: str = Field(description="Unique experiment identifier")
    model_id: str = Field(description="Evaluated model identifier")
    model_family: str = Field(description="Model family (CNN, ResNet, ViT)")
    dataset_id: str = Field(description="Evaluated dataset identifier")
    eval_split: str = Field(description="Evaluated split name")
    layer_name: str = Field(description="Representation extraction layer")
    num_samples: int = Field(ge=0, description="Total sample count")
    clean_accuracy: float = Field(ge=0.0, le=1.0, description="Clean top-1 accuracy")
    clean_loss: float = Field(ge=0.0, description="Clean cross-entropy loss")
    evaluations: dict[str, CorruptionEvaluationSummary] = Field(
        description="Evaluation summaries keyed by '{corruption}::sev{severity}'"
    )
    severity_curves: dict[str, CorruptionSeverityCurve] = Field(
        description="Severity curves keyed by corruption type"
    )
    sample_drifts: dict[str, list[SampleRepresentationDrift]] = Field(
        default_factory=dict,
        description="Per-sample drift lists keyed by '{corruption}::sev{severity}'",
    )
    flagged_failures: list[RobustnessFailureRecord] = Field(
        default_factory=list,
        description="Flagged failure records categorized by taxonomy",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Scientific warnings (e.g. small sample counts)",
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        return self.model_dump(mode="json")

    def to_json(self, indent: int | None = None) -> str:
        """Convert report to JSON string."""
        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=True,
            ensure_ascii=True,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RobustnessExperimentReport:
        """Create report from dictionary."""
        try:
            return cls.model_validate(data)
        except Exception as exc:
            raise SerializationError(
                f"Failed to deserialize RobustnessExperimentReport: {exc}"
            ) from exc


class ArchitectureRobustnessSummary(BaseModel):
    """Summary of robustness metrics for an individual architecture."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    architecture: str = Field(description="Architecture name ('cnn', 'resnet', 'vit')")
    model_family: str = Field(description="Model family enum")
    model_id: str = Field(description="Model identifier")
    clean_accuracy: float = Field(ge=0.0, le=1.0, description="Clean baseline accuracy")
    mean_corrupted_accuracy: float = Field(
        ge=0.0, le=1.0, description="Mean corrupted accuracy across all corruptions"
    )
    mean_accuracy_drop: float = Field(
        description="Mean accuracy drop across all corruptions"
    )
    mean_representation_drift: float = Field(
        ge=0.0, description="Mean Euclidean representation drift"
    )
    mean_neighbor_overlap: float = Field(
        ge=0.0, le=1.0, description="Mean top-k neighbor retention fraction"
    )
    mean_centroid_displacement: float = Field(
        ge=0.0, description="Mean class centroid displacement"
    )
    total_parameters: int | None = Field(
        default=None, description="Total parameter count"
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert summary to dictionary."""
        return self.model_dump(mode="json")


class CrossArchitectureRobustnessReport(BaseModel):
    """Comparative report evaluating multiple architectures under corruptions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    comparison_id: str = Field(description="Unique comparison identifier")
    name: str = Field(description="Comparison title")
    dataset_id: str = Field(description="Evaluated dataset identifier")
    architectures: dict[str, ArchitectureRobustnessSummary] = Field(
        description="Architecture summaries keyed by label"
    )
    detailed_reports: dict[str, RobustnessExperimentReport] = Field(
        default_factory=dict,
        description="Detailed experiment reports per architecture",
    )
    coordinate_space_note: str = Field(
        default=(
            "Note: Each architecture's representation space is independent. "
            "Comparisons evaluate scalar robustness invariants rather than "
            "direct coordinate overlays."
        ),
        description="Scientific methodology note",
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        return self.model_dump(mode="json")

    def to_json(self, indent: int | None = None) -> str:
        """Convert report to JSON string."""
        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=True,
            ensure_ascii=True,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CrossArchitectureRobustnessReport:
        """Create report from dictionary."""
        try:
            return cls.model_validate(data)
        except Exception as exc:
            raise SerializationError(
                f"Failed to deserialize CrossArchitectureRobustnessReport: {exc}"
            ) from exc


# -----------------------------------------------------------------------------
# Robustness Suite Runner
# -----------------------------------------------------------------------------


def _compute_sample_losses(
    logits: list[list[float]], targets: list[int]
) -> list[float]:
    """Compute per-sample cross-entropy losses."""
    losses: list[float] = []
    for idx, row in enumerate(logits):
        t = targets[idx] if 0 <= targets[idx] < len(row) else 0
        max_v = max(row)
        exp_sum = sum(math.exp(v - max_v) for v in row)
        log_prob = (row[t] - max_v) - math.log(max(1e-12, exp_sum))
        losses.append(-log_prob)
    return losses


class RobustnessSuiteRunner:
    """Orchestrates evaluation of frozen models on corruption suites."""

    def __init__(self) -> None:
        self.loss_fn = SoftmaxCrossEntropyLoss()

    def run_suite(
        self,
        model: BaseVisionModel,
        clean_dataset: MaterializedDataset,
        suite: CorruptionSuite,
        experiment_id: str = "exp-robustness",
    ) -> RobustnessExperimentReport:
        """Evaluate a frozen model on clean inputs and all suite corruptions."""
        sample_ids = clean_dataset.sample_ids
        targets_raw = clean_dataset.targets
        targets = [
            int(t) if isinstance(t, (int, str)) and str(t).isdigit() else 0
            for t in targets_raw
        ]
        clean_images = [sample.data for sample in clean_dataset]
        num_samples = len(clean_dataset)

        # 1. Clean Baseline Evaluation
        clean_logits = model.forward(clean_images)
        clean_loss_mean, _ = self.loss_fn(clean_logits, targets)
        clean_loss_sample = _compute_sample_losses(clean_logits, targets)
        clean_acc = compute_accuracy(clean_logits, targets)
        clean_preds = [
            int(max(range(len(row)), key=lambda c: row[c])) for row in clean_logits
        ]

        # Extract clean representations
        clean_raw_rep = model.extract_representations(
            clean_images, layer=suite.layer_name
        )
        clean_rep_ds = RepresentationDataset.from_raw_representations(
            raw_embeddings=clean_raw_rep,
            sample_ids=sample_ids,
            labels=targets,
            experiment_id=experiment_id,
            model_id=model.model_id,
            layer_name=suite.layer_name,
            spatial_policy=SpatialVectorizationPolicy.GLOBAL_AVERAGE_POOL,
            norm_policy=VectorNormalizationPolicy.NONE,
        )

        evaluations: dict[str, CorruptionEvaluationSummary] = {}
        sample_drifts_map: dict[str, list[SampleRepresentationDrift]] = {}
        flagged_failures: list[RobustnessFailureRecord] = []
        severity_curves: dict[str, CorruptionSeverityCurve] = {}
        warnings: list[str] = []

        if num_samples < 5:
            warnings.append(
                f"Small sample count ({num_samples}) may produce noisy statistics."
            )

        # 2. Iterate across corruptions and severities
        for c_type in suite.corruption_types:
            acc_traj: list[float] = []
            loss_traj: list[float] = []
            drift_traj: list[float] = []
            cons_traj: list[float] = []
            cent_disp_traj: list[float] = []

            for sev in suite.severities:
                eval_key = f"{c_type.value}::sev{sev}"
                c_spec = CorruptionSpecification(
                    corruption_type=c_type,
                    severity=sev,
                    seed=suite.seed,
                )
                corrupted_view = CorruptedDatasetView(
                    base_dataset=clean_dataset,
                    corruption_spec=c_spec,
                )
                corrupted_images = [sample.data for sample in corrupted_view]

                # Forward pass on corrupted images
                corr_logits = model.forward(corrupted_images)
                corr_loss_mean, _ = self.loss_fn(corr_logits, targets)
                corr_loss_sample = _compute_sample_losses(corr_logits, targets)
                corr_acc = compute_accuracy(corr_logits, targets)
                corr_preds = [
                    int(max(range(len(row)), key=lambda c: row[c]))
                    for row in corr_logits
                ]

                # Extract corrupted representations
                corr_raw_rep = model.extract_representations(
                    corrupted_images, layer=suite.layer_name
                )
                corr_rep_ds = RepresentationDataset.from_raw_representations(
                    raw_embeddings=corr_raw_rep,
                    sample_ids=sample_ids,
                    labels=targets,
                    experiment_id=experiment_id,
                    model_id=model.model_id,
                    layer_name=suite.layer_name,
                    spatial_policy=SpatialVectorizationPolicy.GLOBAL_AVERAGE_POOL,
                    norm_policy=VectorNormalizationPolicy.NONE,
                )

                # Representation drift
                drift_summary, sample_records = compute_representation_drift(
                    clean_dataset=clean_rep_ds,
                    corrupted_dataset=corr_rep_ds,
                    clean_predictions=clean_preds,
                    corrupted_predictions=corr_preds,
                    clean_losses=clean_loss_sample,
                    corrupted_losses=corr_loss_sample,
                )
                sample_drifts_map[eval_key] = sample_records

                # Geometry drift
                geom_drift = compute_geometry_drift(
                    clean_dataset=clean_rep_ds,
                    corrupted_dataset=corr_rep_ds,
                    k=suite.k_neighbors,
                    n_pca_components=suite.pca_components,
                    metric=DistanceMetric.EUCLIDEAN,
                )

                # ViT attention drift
                attn_drift = compute_vit_attention_drift(
                    model=model,
                    clean_inputs=clean_images,
                    corrupted_inputs=corrupted_images,
                )

                # Count failures by taxonomy
                pred_changes = sum(1 for r in sample_records if r.prediction_changed)
                clean_corr_to_wrong = sum(
                    1
                    for r in sample_records
                    if r.clean_correct and not r.corrupted_correct
                )
                mean_d = drift_summary.mean_euclidean_drift
                std_d = drift_summary.std_euclidean_drift
                high_drift_count = sum(
                    1 for r in sample_records if r.euclidean_drift > mean_d + std_d
                )

                cat_flip = RobustnessFailureCategory.PREDICTION_FLIP.value
                cat_wrong = (
                    RobustnessFailureCategory.CLEAN_CORRECT_TO_CORRUPTED_WRONG.value
                )
                cat_drift = RobustnessFailureCategory.HIGH_REPRESENTATION_DRIFT.value
                cat_neigh = RobustnessFailureCategory.NEIGHBORHOOD_BREAKDOWN.value
                cat_cent = RobustnessFailureCategory.CLASS_CENTROID_DRIFT.value

                failure_counts: dict[str, int] = {
                    cat_flip: pred_changes,
                    cat_wrong: clean_corr_to_wrong,
                    cat_drift: high_drift_count,
                    cat_neigh: int(
                        (
                            1.0
                            - geom_drift.neighborhood_drift.mean_neighbor_overlap_ratio
                        )
                        * num_samples
                    ),
                    cat_cent: len(geom_drift.class_centroid_drifts),
                }

                # Flag individual failure records
                for rec in sample_records:
                    if rec.clean_correct and not rec.corrupted_correct:
                        desc = (
                            f"Flip: {rec.clean_prediction} -> "
                            f"{rec.corrupted_prediction} ({c_type.value} sev {sev})"
                        )
                        flagged_failures.append(
                            RobustnessFailureRecord(
                                sample_id=rec.sample_id,
                                category=RobustnessFailureCategory.CLEAN_CORRECT_TO_CORRUPTED_WRONG,
                                description=desc,
                                severity=sev,
                                metrics={
                                    "drift": rec.euclidean_drift,
                                    "cosine_similarity": rec.cosine_similarity,
                                    "loss_increase": rec.corrupted_loss
                                    - rec.clean_loss,
                                },
                            )
                        )

                eval_summary = CorruptionEvaluationSummary(
                    corruption_type=c_type,
                    severity=sev,
                    num_samples=num_samples,
                    clean_accuracy=clean_acc,
                    corrupted_accuracy=corr_acc,
                    absolute_accuracy_drop=clean_acc - corr_acc,
                    relative_accuracy_drop=(clean_acc - corr_acc) / (clean_acc + 1e-12),
                    clean_loss=clean_loss_mean,
                    corrupted_loss=corr_loss_mean,
                    loss_increase=corr_loss_mean - clean_loss_mean,
                    predictions_changed_count=pred_changes,
                    prediction_consistency_fraction=1.0
                    - (float(pred_changes) / float(max(1, num_samples))),
                    representation_drift=drift_summary,
                    geometry_drift=geom_drift,
                    attention_drift=attn_drift,
                    failure_counts_by_category=failure_counts,
                )
                evaluations[eval_key] = eval_summary

                acc_traj.append(corr_acc)
                loss_traj.append(corr_loss_mean)
                drift_traj.append(drift_summary.mean_euclidean_drift)
                cons_traj.append(
                    geom_drift.neighborhood_drift.corrupted_mean_label_consistency
                )
                cent_disp_traj.append(geom_drift.mean_centroid_displacement)

            # Compute AUC for severity curve
            auc = 0.0
            if len(acc_traj) > 1:
                for idx in range(len(acc_traj) - 1):
                    auc += 0.5 * (acc_traj[idx] + acc_traj[idx + 1])
                auc /= float(len(acc_traj) - 1)
            elif len(acc_traj) == 1:
                auc = acc_traj[0]

            curve = CorruptionSeverityCurve(
                corruption_type=c_type,
                severities=list(suite.severities),
                clean_accuracy=clean_acc,
                accuracy_trajectory=acc_traj,
                loss_trajectory=loss_traj,
                representation_drift_trajectory=drift_traj,
                neighbor_consistency_trajectory=cons_traj,
                centroid_displacement_trajectory=cent_disp_traj,
                total_accuracy_drop=clean_acc - acc_traj[-1] if acc_traj else 0.0,
                mean_accuracy=sum(acc_traj) / float(max(1, len(acc_traj))),
                area_under_curve=auc,
            )
            severity_curves[c_type.value] = curve

        return RobustnessExperimentReport(
            experiment_id=experiment_id,
            model_id=model.model_id,
            model_family=model.spec.family.value,
            dataset_id=clean_dataset.dataset_id,
            eval_split=suite.eval_split,
            layer_name=suite.layer_name,
            num_samples=num_samples,
            clean_accuracy=clean_acc,
            clean_loss=clean_loss_mean,
            evaluations=evaluations,
            severity_curves=severity_curves,
            sample_drifts=sample_drifts_map,
            flagged_failures=flagged_failures,
            warnings=warnings,
        )


def compare_architecture_robustness(
    models: dict[str, BaseVisionModel],
    clean_dataset: MaterializedDataset,
    suite: CorruptionSuite,
    comparison_id: str = "comp-arch-robustness",
    name: str = "Cross-Architecture Robustness & Distribution Shift Benchmark",
) -> CrossArchitectureRobustnessReport:
    """Evaluate and compare robustness across CNN, ResNet, and ViT models."""
    runner = RobustnessSuiteRunner()
    arch_summaries: dict[str, ArchitectureRobustnessSummary] = {}
    detailed_reports: dict[str, RobustnessExperimentReport] = {}

    for arch_key, model in models.items():
        report = runner.run_suite(
            model=model,
            clean_dataset=clean_dataset,
            suite=suite,
            experiment_id=f"exp-robustness-{arch_key}",
        )
        detailed_reports[arch_key] = report

        all_corr_accs = [ev.corrupted_accuracy for ev in report.evaluations.values()]
        all_acc_drops = [
            ev.absolute_accuracy_drop for ev in report.evaluations.values()
        ]
        all_drifts = [
            ev.representation_drift.mean_euclidean_drift
            for ev in report.evaluations.values()
        ]
        all_overlaps = [
            ev.geometry_drift.neighborhood_drift.mean_neighbor_overlap_ratio
            for ev in report.evaluations.values()
        ]
        all_disps = [
            ev.geometry_drift.mean_centroid_displacement
            for ev in report.evaluations.values()
        ]

        param_count: int | None = None
        try:
            param_count = sum(
                len(p) if isinstance(p, list) else 1
                for p in model.get_parameters().values()
            )
        except Exception:
            param_count = None

        summary = ArchitectureRobustnessSummary(
            architecture=arch_key,
            model_family=model.spec.family.value,
            model_id=model.model_id,
            clean_accuracy=report.clean_accuracy,
            mean_corrupted_accuracy=sum(all_corr_accs)
            / float(max(1, len(all_corr_accs))),
            mean_accuracy_drop=sum(all_acc_drops) / float(max(1, len(all_acc_drops))),
            mean_representation_drift=sum(all_drifts) / float(max(1, len(all_drifts))),
            mean_neighbor_overlap=sum(all_overlaps) / float(max(1, len(all_overlaps))),
            mean_centroid_displacement=sum(all_disps) / float(max(1, len(all_disps))),
            total_parameters=param_count,
        )
        arch_summaries[arch_key] = summary

    return CrossArchitectureRobustnessReport(
        comparison_id=comparison_id,
        name=name,
        dataset_id=clean_dataset.dataset_id,
        architectures=arch_summaries,
        detailed_reports=detailed_reports,
    )
