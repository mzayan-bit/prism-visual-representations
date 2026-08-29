"""Cross-architecture representation geometry comparison and seed aggregation."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from prism.core.errors import SerializationError, ValidationError
from prism.models.base import BaseVisionModel
from prism.representations.geometry import (
    DistanceMetric,
    RepresentationDataset,
    SpatialVectorizationPolicy,
    VectorNormalizationPolicy,
)
from prism.representations.reports import (
    RepresentationGeometryReport,
    analyze_representation_geometry,
)


class ArchitectureGeometrySummary(BaseModel):
    """Scalar geometric summary for a single architecture's representation space."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    architecture: str = Field(
        description="Architecture name (e.g. 'cnn', 'resnet', 'vit')"
    )
    model_family: str = Field(description="Model family enum value")
    model_id: str = Field(description="Model identifier")
    layer_name: str = Field(description="Probed representation layer")
    feature_dim: int = Field(gt=0, description="Representation dimensionality")
    mean_vector_norm: float = Field(ge=0.0, description="Mean L2 vector norm")
    intra_class_compactness: float = Field(
        ge=0.0, description="Average intra-class sample-to-centroid distance"
    )
    inter_class_separation: float = Field(
        ge=0.0, description="Average inter-class centroid-to-centroid distance"
    )
    separation_to_compactness_ratio: float = Field(
        ge=0.0, description="Separation / compactness ratio"
    )
    neighbor_label_consistency: float = Field(
        ge=0.0, le=1.0, description="k-NN same-class neighborhood consistency"
    )
    pca_first_two_variance_ratio: float = Field(
        ge=0.0,
        le=1.0,
        description="Variance ratio explained by top 2 PCA components",
    )
    total_parameters: int | None = Field(
        default=None, description="Trainable parameter count if available"
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert summary to dictionary."""
        return self.model_dump(mode="json")


class RepeatedSeedGeometryMetric(BaseModel):
    """Aggregated statistics across multiple seeds for a geometry metric."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    metric_name: str = Field(description="Metric name")
    mean: float = Field(description="Mean value across seeds")
    std: float = Field(ge=0.0, description="Standard deviation across seeds")
    min_value: float = Field(description="Minimum value across seeds")
    max_value: float = Field(description="Maximum value across seeds")
    num_seeds: int = Field(ge=1, description="Number of seed runs aggregated")

    def to_dict(self) -> dict[str, Any]:
        """Convert metric to dictionary."""
        return self.model_dump(mode="json")


class CrossArchitectureGeometryReport(BaseModel):
    """Comparative report evaluating geometry across architecture families."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    comparison_id: str = Field(description="Unique comparison identifier")
    name: str = Field(description="Descriptive comparison title")
    dataset_fingerprint: str = Field(description="Fingerprint of evaluation dataset")
    data_budget: float = Field(
        ge=0.0, le=1.0, description="Data budget ratio evaluated"
    )
    distance_metric: DistanceMetric = Field(description="Distance metric used")
    normalization_policy: VectorNormalizationPolicy = Field(
        description="Normalization policy"
    )
    architectures: dict[str, ArchitectureGeometrySummary] = Field(
        description="Geometry summaries keyed by architecture label"
    )
    detailed_reports: dict[str, RepresentationGeometryReport] = Field(
        default_factory=dict,
        description="Full geometry reports per architecture",
    )
    repeated_seed_aggregates: dict[str, list[RepeatedSeedGeometryMetric]] | None = (
        Field(
            default=None,
            description="Aggregated metrics per architecture across seeds",
        )
    )
    coordinate_space_note: str = Field(
        default=(
            "Note: Principal component projection spaces are fitted independently "
            "per architecture. Comparisons evaluate scalar geometric invariants "
            "rather than direct coordinate overlaps."
        ),
        description="Methodological disclaimer on independent coordinate spaces",
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
    def from_dict(cls, data: dict[str, Any]) -> CrossArchitectureGeometryReport:
        """Create report from dictionary."""
        try:
            return cls.model_validate(data)
        except Exception as exc:
            raise SerializationError(
                f"Failed to deserialize CrossArchitectureGeometryReport "
                f"from dict: {exc}"
            ) from exc


def compare_architecture_geometries(
    models: dict[str, BaseVisionModel],
    inputs: Any,
    sample_ids: list[str],
    labels: Sequence[int | str],
    comparison_id: str = "comp-arch-geometry",
    name: str = "Cross-Architecture Representation Geometry Comparison",
    dataset_fingerprint: str = "eval-dataset-fingerprint",
    data_budget: float = 1.0,
    distance_metric: DistanceMetric | str = DistanceMetric.EUCLIDEAN,
    normalization_policy: VectorNormalizationPolicy
    | str = VectorNormalizationPolicy.NONE,
    spatial_policy: SpatialVectorizationPolicy
    | str = SpatialVectorizationPolicy.GLOBAL_AVERAGE_POOL,
    k: int = 5,
    layer_map: dict[str, str] | None = None,
    class_names: list[str] | None = None,
) -> CrossArchitectureGeometryReport:
    """Compare representation geometry across vision architectures.

    Parameters
    ----------
    models : dict[str, BaseVisionModel]
        Dict mapping architecture key ('cnn', 'resnet', 'vit') to model.
    inputs : Any
        Evaluation batch [N, C, H, W].
    sample_ids : list[str]
        Aligned sample IDs.
    labels : list[int | str]
        Aligned labels.
    comparison_id : str
        Comparison ID.
    name : str
        Comparison title.
    dataset_fingerprint : str
        Dataset fingerprint.
    data_budget : float
        Data budget ratio.
    distance_metric : DistanceMetric | str
        Distance metric.
    normalization_policy : VectorNormalizationPolicy | str
        Normalization policy.
    spatial_policy : SpatialVectorizationPolicy | str
        Spatial pooling policy.
    k : int
        Number of nearest neighbors.
    layer_map : dict[str, str] | None
        Optional mapping from architecture key to layer name.
    class_names : list[str] | None
        Class names.

    Returns
    -------
    CrossArchitectureGeometryReport
        Comprehensive comparative geometry report.
    """
    if not models:
        raise ValidationError(
            "Must provide at least one model for geometry comparison."
        )

    metric_enum = (
        DistanceMetric(distance_metric)
        if isinstance(distance_metric, str)
        else distance_metric
    )
    norm_enum = (
        VectorNormalizationPolicy(normalization_policy)
        if isinstance(normalization_policy, str)
        else normalization_policy
    )

    summaries: dict[str, ArchitectureGeometrySummary] = {}
    detailed_reports: dict[str, RepresentationGeometryReport] = {}

    for arch_key, model in models.items():
        target_layer = "final_hidden"
        if layer_map and arch_key in layer_map:
            target_layer = layer_map[arch_key]
        elif hasattr(model, "spec") and getattr(model.spec, "family", None):
            fam = str(model.spec.family.value).lower()
            if "transformer" in fam or "vit" in fam:
                target_layer = "cls_representation"

        raw_repr = model.extract_representations(inputs, layer=target_layer)

        dataset = RepresentationDataset.from_raw_representations(
            raw_embeddings=raw_repr,
            sample_ids=sample_ids,
            labels=labels,
            experiment_id=f"{comparison_id}-{arch_key}",
            model_id=model.model_id,
            layer_name=target_layer,
            spatial_policy=spatial_policy,
            norm_policy=norm_enum,
            class_names=class_names,
        )

        rep_report = analyze_representation_geometry(
            dataset=dataset,
            k=k,
            metric=metric_enum,
            n_pca_components=2,
        )
        detailed_reports[arch_key] = rep_report

        fam_str = (
            str(model.spec.family.value)
            if hasattr(model, "spec") and hasattr(model.spec, "family")
            else "custom"
        )
        arch_name = (
            model.spec.architecture
            if hasattr(model, "spec") and hasattr(model.spec, "architecture")
            else arch_key
        )

        pca_var_ratio = (
            rep_report.pca_projection.cumulative_explained_variance[1]
            if len(rep_report.pca_projection.cumulative_explained_variance) >= 2
            else (
                rep_report.pca_projection.cumulative_explained_variance[0]
                if rep_report.pca_projection.cumulative_explained_variance
                else 0.0
            )
        )

        param_count = None
        if hasattr(model, "get_parameters"):
            params = model.get_parameters()
            param_count = sum(
                len(v)
                if isinstance(v, list) and not isinstance(v[0], list)
                else (sum(len(r) for r in v) if isinstance(v, list) else 1)
                for v in params.values()
                if v is not None and isinstance(v, list)
            )

        summary = ArchitectureGeometrySummary(
            architecture=arch_name,
            model_family=fam_str,
            model_id=model.model_id,
            layer_name=target_layer,
            feature_dim=dataset.feature_dim,
            mean_vector_norm=rep_report.vector_norms.mean_norm,
            intra_class_compactness=rep_report.centroid_geometry.mean_intra_class_distance,
            inter_class_separation=rep_report.centroid_geometry.mean_inter_class_centroid_distance,
            separation_to_compactness_ratio=rep_report.centroid_geometry.separation_to_compactness_ratio,
            neighbor_label_consistency=rep_report.neighborhood_geometry.mean_label_consistency,
            pca_first_two_variance_ratio=pca_var_ratio,
            total_parameters=param_count,
        )
        summaries[arch_key] = summary

    return CrossArchitectureGeometryReport(
        comparison_id=comparison_id,
        name=name,
        dataset_fingerprint=dataset_fingerprint,
        data_budget=data_budget,
        distance_metric=metric_enum,
        normalization_policy=norm_enum,
        architectures=summaries,
        detailed_reports=detailed_reports,
    )


def aggregate_repeated_seed_geometry(
    seed_reports: list[RepresentationGeometryReport],
) -> list[RepeatedSeedGeometryMetric]:
    """Aggregate scalar representation geometry metrics across seed runs.

    Parameters
    ----------
    seed_reports : list[RepresentationGeometryReport]
        List of geometry reports from identical configurations across seeds.

    Returns
    -------
    list[RepeatedSeedGeometryMetric]
        Statistical mean, std, min, max per scalar metric.
    """
    if not seed_reports:
        return []

    num_seeds = len(seed_reports)

    # Extract scalar series
    compactness_vals = [
        r.centroid_geometry.mean_intra_class_distance for r in seed_reports
    ]
    separation_vals = [
        r.centroid_geometry.mean_inter_class_centroid_distance for r in seed_reports
    ]
    ratio_vals = [
        r.centroid_geometry.separation_to_compactness_ratio for r in seed_reports
    ]
    consistency_vals = [
        r.neighborhood_geometry.mean_label_consistency for r in seed_reports
    ]
    pca_var_vals = [
        (
            r.pca_projection.cumulative_explained_variance[1]
            if len(r.pca_projection.cumulative_explained_variance) >= 2
            else (
                r.pca_projection.cumulative_explained_variance[0]
                if r.pca_projection.cumulative_explained_variance
                else 0.0
            )
        )
        for r in seed_reports
    ]

    def _make_metric(name: str, values: list[float]) -> RepeatedSeedGeometryMetric:
        mean_v = sum(values) / float(num_seeds)
        var_v = (
            sum((x - mean_v) ** 2 for x in values) / float(num_seeds)
            if num_seeds > 1
            else 0.0
        )
        std_v = math.sqrt(max(0.0, var_v))
        return RepeatedSeedGeometryMetric(
            metric_name=name,
            mean=mean_v,
            std=std_v,
            min_value=min(values),
            max_value=max(values),
            num_seeds=num_seeds,
        )

    return [
        _make_metric("intra_class_compactness", compactness_vals),
        _make_metric("inter_class_separation", separation_vals),
        _make_metric("separation_to_compactness_ratio", ratio_vals),
        _make_metric("neighbor_label_consistency", consistency_vals),
        _make_metric("pca_first_two_variance_ratio", pca_var_vals),
    ]
