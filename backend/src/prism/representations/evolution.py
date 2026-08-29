"""Layer-wise representation geometry evolution across network depth."""

from __future__ import annotations

import json
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


class LayerGeometryPoint(BaseModel):
    """Geometric summary metrics for an individual architectural layer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    layer_name: str = Field(description="Layer identifier name")
    depth_index: int = Field(
        ge=0, description="Sequential depth index (0 is earliest layer)"
    )
    feature_dim: int = Field(
        gt=0, description="Feature dimensionality of representation"
    )
    original_shape: tuple[int, ...] | None = Field(
        default=None, description="Original tensor shape before vectorization"
    )
    mean_intra_class_distance: float = Field(
        ge=0.0, description="Intra-class compactness at this layer"
    )
    mean_inter_class_centroid_distance: float = Field(
        ge=0.0, description="Inter-class separation at this layer"
    )
    separation_to_compactness_ratio: float = Field(
        ge=0.0, description="Separation / compactness ratio at this layer"
    )
    mean_label_consistency: float = Field(
        ge=0.0, le=1.0, description="k-NN neighborhood label consistency"
    )
    pca_first_two_variance_ratio: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of variance captured by first 2 PCA components",
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert point to dictionary."""
        return self.model_dump(mode="json")


class LayerGeometryProfile(BaseModel):
    """Complete profile of representation geometry evolution across depth."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_id: str = Field(description="Associated experiment identifier")
    model_id: str = Field(description="Model identifier")
    architecture: str = Field(description="Model architecture name")
    distance_metric: DistanceMetric = Field(description="Distance metric used")
    normalization_policy: VectorNormalizationPolicy = Field(
        description="Normalization policy applied"
    )
    layer_points: list[LayerGeometryPoint] = Field(
        description="Ordered sequence of layer geometric points"
    )
    compactness_trend: list[float] = Field(
        description="Intra-class compactness values across depth"
    )
    separation_trend: list[float] = Field(
        description="Inter-class separation values across depth"
    )
    consistency_trend: list[float] = Field(
        description="Neighborhood label consistency values across depth"
    )
    ratio_trend: list[float] = Field(
        description="Separation-to-compactness ratios across depth"
    )
    detailed_reports: dict[str, RepresentationGeometryReport] = Field(
        default_factory=dict,
        description="Full geometry reports keyed by layer name",
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert profile to dictionary."""
        return self.model_dump(mode="json")

    def to_json(self, indent: int | None = None) -> str:
        """Convert profile to JSON string."""
        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=True,
            ensure_ascii=True,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LayerGeometryProfile:
        """Create profile from dictionary."""
        try:
            return cls.model_validate(data)
        except Exception as exc:
            raise SerializationError(
                f"Failed to deserialize LayerGeometryProfile from dict: {exc}"
            ) from exc


def analyze_layer_geometry_profile(
    model: BaseVisionModel,
    inputs: Any,
    sample_ids: list[str],
    labels: list[int | str],
    layers: list[str],
    experiment_id: str = "exp-layer-profile",
    distance_metric: DistanceMetric | str = DistanceMetric.EUCLIDEAN,
    normalization_policy: VectorNormalizationPolicy
    | str = VectorNormalizationPolicy.NONE,
    spatial_policy: SpatialVectorizationPolicy
    | str = SpatialVectorizationPolicy.GLOBAL_AVERAGE_POOL,
    k: int = 5,
    class_names: list[str] | None = None,
) -> LayerGeometryProfile:
    """Extract and analyze representation geometry across an ordered sequence of layers.

    Parameters
    ----------
    model : BaseVisionModel
        Trained or instantiated vision model.
    inputs : Any
        Evaluation batch of images [N, C, H, W].
    sample_ids : list[str]
        Aligned sample IDs.
    labels : list[int | str]
        Aligned category labels.
    layers : list[str]
        Ordered list of layer names to probe.
    experiment_id : str
        Associated experiment ID.
    distance_metric : DistanceMetric | str
        Distance metric.
    normalization_policy : VectorNormalizationPolicy | str
        Normalization policy.
    spatial_policy : SpatialVectorizationPolicy | str
        Spatial vectorization policy.
    k : int
        Number of nearest neighbors.
    class_names : list[str] | None
        Class names list.

    Returns
    -------
    LayerGeometryProfile
        Complete layer-wise evolution profile.
    """
    if not layers:
        raise ValidationError("Must provide at least one layer to analyze.")

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

    layer_points: list[LayerGeometryPoint] = []
    compactness_trend: list[float] = []
    separation_trend: list[float] = []
    consistency_trend: list[float] = []
    ratio_trend: list[float] = []
    reports_map: dict[str, RepresentationGeometryReport] = {}

    for depth_idx, layer_name in enumerate(layers):
        raw_repr = model.extract_representations(inputs, layer=layer_name)

        dataset = RepresentationDataset.from_raw_representations(
            raw_embeddings=raw_repr,
            sample_ids=sample_ids,
            labels=labels,
            experiment_id=experiment_id,
            model_id=model.model_id,
            layer_name=layer_name,
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
        reports_map[layer_name] = rep_report

        intra_d = rep_report.centroid_geometry.mean_intra_class_distance
        inter_d = rep_report.centroid_geometry.mean_inter_class_centroid_distance
        ratio_d = rep_report.centroid_geometry.separation_to_compactness_ratio
        cons_d = rep_report.neighborhood_geometry.mean_label_consistency

        pca_var_ratio = (
            rep_report.pca_projection.cumulative_explained_variance[1]
            if len(rep_report.pca_projection.cumulative_explained_variance) >= 2
            else (
                rep_report.pca_projection.cumulative_explained_variance[0]
                if rep_report.pca_projection.cumulative_explained_variance
                else 0.0
            )
        )

        point = LayerGeometryPoint(
            layer_name=layer_name,
            depth_index=depth_idx,
            feature_dim=dataset.feature_dim,
            original_shape=dataset.original_shape,
            mean_intra_class_distance=intra_d,
            mean_inter_class_centroid_distance=inter_d,
            separation_to_compactness_ratio=ratio_d,
            mean_label_consistency=cons_d,
            pca_first_two_variance_ratio=pca_var_ratio,
        )
        layer_points.append(point)
        compactness_trend.append(intra_d)
        separation_trend.append(inter_d)
        consistency_trend.append(cons_d)
        ratio_trend.append(ratio_d)

    arch_name = (
        model.spec.architecture
        if hasattr(model, "spec") and hasattr(model.spec, "architecture")
        else model.model_id
    )

    return LayerGeometryProfile(
        experiment_id=experiment_id,
        model_id=model.model_id,
        architecture=arch_name,
        distance_metric=metric_enum,
        normalization_policy=norm_enum,
        layer_points=layer_points,
        compactness_trend=compactness_trend,
        separation_trend=separation_trend,
        consistency_trend=consistency_trend,
        ratio_trend=ratio_trend,
        detailed_reports=reports_map,
    )
