"""Comprehensive representation geometry reports and end-to-end analyzers."""

from __future__ import annotations

import json
import math
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from prism.core.errors import SerializationError, ValidationError
from prism.representations.centroids import (
    CentroidGeometryReport,
    compute_centroid_geometry,
)
from prism.representations.geometry import (
    DistanceMetric,
    RepresentationDataset,
    SpatialVectorizationPolicy,
    VectorNormalizationPolicy,
)
from prism.representations.neighborhood import (
    CandidateFailureCase,
    NeighborhoodGeometrySummary,
    compute_neighborhood_geometry,
)
from prism.representations.pca import (
    ProjectionResult,
    compute_pca_projection,
)


class VectorNormSummary(BaseModel):
    """Statistical properties of representation vector magnitudes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mean_norm: float = Field(ge=0.0, description="Mean L2 norm across vectors")
    std_norm: float = Field(ge=0.0, description="Standard deviation of L2 norms")
    min_norm: float = Field(ge=0.0, description="Minimum L2 norm observed")
    max_norm: float = Field(ge=0.0, description="Maximum L2 norm observed")

    def to_dict(self) -> dict[str, Any]:
        """Convert vector norm summary to dictionary."""
        return self.model_dump(mode="json")


def _compute_vector_norms(vectors: list[list[float]]) -> VectorNormSummary:
    """Compute L2 norm distribution summary across a batch of vectors."""
    if not vectors:
        return VectorNormSummary(
            mean_norm=0.0, std_norm=0.0, min_norm=0.0, max_norm=0.0
        )

    norms = [math.sqrt(sum(x * x for x in row)) for row in vectors]
    n = len(norms)
    mean_v = sum(norms) / float(n)
    var_v = sum((x - mean_v) ** 2 for x in norms) / float(n) if n > 1 else 0.0
    std_v = math.sqrt(max(0.0, var_v))

    return VectorNormSummary(
        mean_norm=mean_v,
        std_norm=std_v,
        min_norm=min(norms),
        max_norm=max(norms),
    )


class RepresentationGeometryReport(BaseModel):
    """Complete, serializable geometric analysis report for a representation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_id: str = Field(description="Experiment identifier")
    model_id: str = Field(description="Model identifier")
    layer_name: str = Field(description="Representation layer name")
    num_samples: int = Field(ge=0, description="Number of analyzed samples")
    feature_dim: int = Field(gt=0, description="Representation feature dimension")
    num_classes: int = Field(ge=1, description="Number of distinct classes")
    class_names: list[str] = Field(default_factory=list, description="Class name list")
    source_split: str = Field(default="test", description="Data split")
    spatial_transformation: SpatialVectorizationPolicy = Field(
        description="Spatial vectorization policy"
    )
    normalization_policy: VectorNormalizationPolicy = Field(
        description="Vector normalization policy"
    )
    distance_metric: DistanceMetric = Field(description="Distance metric used")
    vector_norms: VectorNormSummary = Field(description="Vector norm statistics")
    centroid_geometry: CentroidGeometryReport = Field(
        description="Centroid, intra-class compactness, and inter-class separation"
    )
    neighborhood_geometry: NeighborhoodGeometrySummary = Field(
        description="k-NN geometry and neighborhood consistency summary"
    )
    pca_projection: ProjectionResult = Field(
        description="2D/3D PCA projection result with explained variance"
    )
    candidate_failures: list[CandidateFailureCase] = Field(
        default_factory=list,
        description="Identified geometrically ambiguous or failure samples",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description=("Scientific or data quality warnings (e.g. small class size)"),
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Auxiliary run metadata"
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
    def from_dict(cls, data: dict[str, Any]) -> RepresentationGeometryReport:
        """Create report from dictionary."""
        try:
            return cls.model_validate(data)
        except Exception as exc:
            raise SerializationError(
                f"Failed to deserialize RepresentationGeometryReport from dict: {exc}"
            ) from exc

    @classmethod
    def from_json(cls, json_str: str) -> RepresentationGeometryReport:
        """Create report from JSON string."""
        try:
            parsed = json.loads(json_str)
            return cls.from_dict(parsed)
        except Exception as exc:
            raise SerializationError(
                f"Failed to deserialize RepresentationGeometryReport from JSON: {exc}"
            ) from exc


def analyze_representation_geometry(
    dataset: RepresentationDataset,
    k: int = 5,
    metric: DistanceMetric | str = DistanceMetric.EUCLIDEAN,
    n_pca_components: int = 2,
    consistency_threshold: float = 0.5,
) -> RepresentationGeometryReport:
    """Execute complete representation geometry analysis on RepresentationDataset.

    Parameters
    ----------
    dataset : RepresentationDataset
        Prepared representation dataset.
    k : int
        Number of nearest neighbors for neighborhood consistency.
    metric : DistanceMetric | str
        Distance metric.
    n_pca_components : int
        Number of PCA projection dimensions (default 2).
    consistency_threshold : float
        Threshold for candidate failure detection.

    Returns
    -------
    RepresentationGeometryReport
        Comprehensive geometry report.
    """
    if dataset.num_samples == 0:
        raise ValidationError(
            "Cannot analyze representation geometry on empty dataset."
        )

    metric_enum = DistanceMetric(metric) if isinstance(metric, str) else metric
    warnings: list[str] = []

    # 1. Vector Norms
    norm_summary = _compute_vector_norms(dataset.vectors)

    # 2. Centroid Geometry
    centroid_report = compute_centroid_geometry(dataset, metric=metric_enum)

    # 3. Neighborhood Geometry
    neighborhood_summary = compute_neighborhood_geometry(
        dataset=dataset,
        k=k,
        metric=metric_enum,
        centroid_report=centroid_report,
        consistency_threshold=consistency_threshold,
    )

    # 4. PCA Projection
    pca_result = compute_pca_projection(
        dataset=dataset,
        n_components=n_pca_components,
    )

    # 5. Scientific Quality Warnings
    for c_id, c_sum in centroid_report.class_centroids.items():
        if c_sum.sample_count < 3:
            warnings.append(
                f"Class '{c_id}' has only {c_sum.sample_count} samples; "
                f"compactness statistics may have high estimation variance."
            )

    if (
        pca_result.cumulative_explained_variance
        and len(pca_result.cumulative_explained_variance) >= 2
        and pca_result.cumulative_explained_variance[1] < 0.20
    ):
        warnings.append(
            f"First 2 PCA components explain only "
            f"{pca_result.cumulative_explained_variance[1]:.1%} of variance; "
            f"2D visualization captures a limited subspace of representation geometry."
        )

    return RepresentationGeometryReport(
        experiment_id=dataset.experiment_id,
        model_id=dataset.model_id,
        layer_name=dataset.layer_name,
        num_samples=dataset.num_samples,
        feature_dim=dataset.feature_dim,
        num_classes=dataset.num_classes,
        class_names=dataset.class_names,
        source_split=dataset.source_split,
        spatial_transformation=dataset.spatial_transformation,
        normalization_policy=dataset.normalization_policy,
        distance_metric=metric_enum,
        vector_norms=norm_summary,
        centroid_geometry=centroid_report,
        neighborhood_geometry=neighborhood_summary,
        pca_projection=pca_result,
        candidate_failures=neighborhood_summary.candidate_failures,
        warnings=warnings,
        metadata=dataset.metadata,
    )
