"""Geometric drift analysis comparing clean vs corrupted manifolds and shared PCA."""

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
    compute_distance,
)
from prism.representations.neighborhood import (
    compute_neighborhood_geometry,
)
from prism.representations.pca import PrincipalComponentAnalysis


class SharedPCAProjectionResult(BaseModel):
    """Projection where clean and corrupted representations share a basis."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    original_dim: int = Field(gt=0, description="Original feature dimensionality")
    projected_dim: int = Field(
        default=2, description="Projected dimensionality (e.g. 2, 3)"
    )
    num_samples: int = Field(ge=0, description="Number of projected samples")
    sample_ids: list[str] = Field(description="Aligned sample identifiers")
    labels: list[int | str] = Field(description="Aligned category labels")
    clean_coordinates: list[list[float]] = Field(
        description="2D/3D coordinates of clean representations [N, K]"
    )
    corrupted_coordinates: list[list[float]] = Field(
        description="2D/3D coordinates of corrupted representations [N, K]"
    )
    displacement_vectors: list[list[float]] = Field(
        description="Displacement vectors: (corrupted - clean) [N, K]"
    )
    displacement_magnitudes: list[float] = Field(
        description="Euclidean length of displacement vectors in projected space"
    )
    explained_variance_ratio: list[float] = Field(
        description="Fraction of clean variance captured by each principal component"
    )
    cumulative_explained_variance: list[float] = Field(
        description="Cumulative variance captured by principal components"
    )
    basis_note: str = Field(
        default="PCA basis fitted on clean representations and reused for corrupted.",
        description="Methodological note on shared coordinate basis",
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert projection to dictionary."""
        return self.model_dump(mode="json")


class ClassCentroidDriftSummary(BaseModel):
    """Summary of class centroid displacement and dispersion changes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    class_label: str = Field(description="Class identifier")
    clean_centroid_norm: float = Field(ge=0.0, description="L2 norm of clean centroid")
    corrupted_centroid_norm: float = Field(
        ge=0.0, description="L2 norm of corrupted centroid"
    )
    centroid_displacement: float = Field(
        ge=0.0,
        description="Euclidean distance between clean and corrupted centroid vectors",
    )
    cosine_similarity: float = Field(
        ge=-1.0,
        le=1.0,
        description="Cosine similarity between clean and corrupted centroid",
    )
    clean_intra_compactness: float = Field(
        ge=0.0, description="Clean intra-class average sample-to-centroid distance"
    )
    corrupted_intra_compactness: float = Field(
        ge=0.0,
        description="Corrupted intra-class average sample-to-centroid distance",
    )
    compactness_delta: float = Field(
        description="Change in compactness: (corrupted - clean)"
    )
    clean_competing_separation: float = Field(
        ge=0.0, description="Clean distance to closest foreign class centroid"
    )
    corrupted_competing_separation: float = Field(
        ge=0.0, description="Corrupted distance to closest foreign class centroid"
    )
    competing_separation_delta: float = Field(
        description="Change in competing separation: (corrupted - clean)"
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert summary to dictionary."""
        return self.model_dump(mode="json")


class NeighborhoodDriftSummary(BaseModel):
    """Summary of neighborhood structure degradation under corruption."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    k: int = Field(ge=1, description="Number of nearest neighbors evaluated")
    mean_neighbor_overlap_ratio: float = Field(
        ge=0.0,
        le=1.0,
        description="Mean fraction of top-k clean neighbors retained under corruption",
    )
    clean_mean_label_consistency: float = Field(
        ge=0.0, le=1.0, description="Clean top-k same-class neighbor fraction"
    )
    corrupted_mean_label_consistency: float = Field(
        ge=0.0, le=1.0, description="Corrupted top-k same-class neighbor fraction"
    )
    label_consistency_delta: float = Field(
        description="Change in label consistency: (corrupted - clean)"
    )
    nearest_neighbor_label_flip_fraction: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of samples whose rank-1 neighbor class changed",
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert summary to dictionary."""
        return self.model_dump(mode="json")


class GeometryDriftReport(BaseModel):
    """Comprehensive report on geometric and manifold degradation under corruption."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    clean_centroid_report: CentroidGeometryReport = Field(
        description="Clean centroid geometry report"
    )
    corrupted_centroid_report: CentroidGeometryReport = Field(
        description="Corrupted centroid geometry report"
    )
    mean_centroid_displacement: float = Field(
        ge=0.0, description="Mean displacement across all class centroids"
    )
    class_centroid_drifts: dict[str, ClassCentroidDriftSummary] = Field(
        description="Per-class centroid drift summaries"
    )
    neighborhood_drift: NeighborhoodDriftSummary = Field(
        description="Neighborhood degradation summary"
    )
    shared_pca: SharedPCAProjectionResult = Field(
        description="Shared PCA projection result"
    )
    separation_to_compactness_ratio_delta: float = Field(
        description="Change in separation-to-compactness ratio: (corrupted - clean)"
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
    def from_dict(cls, data: dict[str, Any]) -> GeometryDriftReport:
        """Create report from dictionary."""
        try:
            return cls.model_validate(data)
        except Exception as exc:
            raise SerializationError(
                f"Failed to deserialize GeometryDriftReport: {exc}"
            ) from exc


def compute_geometry_drift(
    clean_dataset: RepresentationDataset,
    corrupted_dataset: RepresentationDataset,
    k: int = 5,
    n_pca_components: int = 2,
    metric: DistanceMetric | str = DistanceMetric.EUCLIDEAN,
) -> GeometryDriftReport:
    """Compute manifold, centroid, neighborhood, and shared PCA drift under corruption.

    Parameters
    ----------
    clean_dataset : RepresentationDataset
        Clean representation dataset.
    corrupted_dataset : RepresentationDataset
        Corrupted representation dataset aligned by sample IDs.
    k : int
        Number of nearest neighbors to evaluate.
    n_pca_components : int
        Number of PCA components to project.
    metric : DistanceMetric | str
        Distance metric.

    Returns
    -------
    GeometryDriftReport
        Comprehensive geometry drift report.
    """
    n = clean_dataset.num_samples
    if corrupted_dataset.num_samples != n:
        raise ValidationError(
            f"Sample count mismatch: {n} clean vs {corrupted_dataset.num_samples}."
        )

    metric_enum = DistanceMetric(metric) if isinstance(metric, str) else metric

    # 1. Clean and Corrupted Centroid Geometry
    clean_cent = compute_centroid_geometry(clean_dataset, metric=metric_enum)
    corr_cent = compute_centroid_geometry(corrupted_dataset, metric=metric_enum)

    class_drifts: dict[str, ClassCentroidDriftSummary] = {}
    displacements: list[float] = []

    for cls_k, c_clean in clean_cent.class_centroids.items():
        if cls_k in corr_cent.class_centroids:
            c_corr = corr_cent.class_centroids[cls_k]
            disp = compute_distance(
                c_clean.centroid, c_corr.centroid, metric=DistanceMetric.EUCLIDEAN
            )
            sim = compute_distance(
                c_clean.centroid,
                c_corr.centroid,
                metric=DistanceMetric.COSINE_SIMILARITY,
            )
            c_clean_sep = (
                c_clean.distance_to_nearest_competing_centroid
                if c_clean.distance_to_nearest_competing_centroid is not None
                else 0.0
            )
            c_corr_sep = (
                c_corr.distance_to_nearest_competing_centroid
                if c_corr.distance_to_nearest_competing_centroid is not None
                else 0.0
            )
            comp_delta = (
                c_corr.intra_class_mean_distance - c_clean.intra_class_mean_distance
            )
            sep_delta = c_corr_sep - c_clean_sep

            drift_summary = ClassCentroidDriftSummary(
                class_label=cls_k,
                clean_centroid_norm=c_clean.centroid_norm,
                corrupted_centroid_norm=c_corr.centroid_norm,
                centroid_displacement=disp,
                cosine_similarity=sim,
                clean_intra_compactness=c_clean.intra_class_mean_distance,
                corrupted_intra_compactness=c_corr.intra_class_mean_distance,
                compactness_delta=comp_delta,
                clean_competing_separation=c_clean_sep,
                corrupted_competing_separation=c_corr_sep,
                competing_separation_delta=sep_delta,
            )
            class_drifts[cls_k] = drift_summary
            displacements.append(disp)

    mean_disp = sum(displacements) / float(max(1, len(displacements)))

    # 2. Neighborhood Drift
    clean_neigh = compute_neighborhood_geometry(clean_dataset, k=k, metric=metric_enum)
    corr_neigh = compute_neighborhood_geometry(
        corrupted_dataset, k=k, metric=metric_enum
    )

    # Compute top-k neighbor overlap and rank-1 flips
    overlap_ratios: list[float] = []
    rank1_flips = 0

    clean_neigh_map = clean_neigh.sample_neighborhoods
    corr_neigh_map = corr_neigh.sample_neighborhoods

    for sid in clean_dataset.sample_ids:
        if sid in clean_neigh_map and sid in corr_neigh_map:
            c_entry = clean_neigh_map[sid]
            cr_entry = corr_neigh_map[sid]
            c_set = {nb.neighbor_sample_id for nb in c_entry.neighbors}
            cr_set = {nb.neighbor_sample_id for nb in cr_entry.neighbors}
            if len(c_set) > 0:
                overlap = len(c_set.intersection(cr_set)) / float(len(c_set))
                overlap_ratios.append(overlap)

            # Check rank 1 neighbor label
            if c_entry.neighbors and cr_entry.neighbors:
                c_rank1_label = c_entry.neighbors[0].neighbor_label
                cr_rank1_label = cr_entry.neighbors[0].neighbor_label
                if c_rank1_label != cr_rank1_label:
                    rank1_flips += 1

    mean_overlap = sum(overlap_ratios) / float(max(1, len(overlap_ratios)))
    flip_fraction = float(rank1_flips) / float(max(1, n))

    neigh_drift = NeighborhoodDriftSummary(
        k=k,
        mean_neighbor_overlap_ratio=mean_overlap,
        clean_mean_label_consistency=clean_neigh.mean_label_consistency,
        corrupted_mean_label_consistency=corr_neigh.mean_label_consistency,
        label_consistency_delta=(
            corr_neigh.mean_label_consistency - clean_neigh.mean_label_consistency
        ),
        nearest_neighbor_label_flip_fraction=flip_fraction,
    )

    # 3. Shared PCA Projection
    pca = PrincipalComponentAnalysis(n_components=n_pca_components)
    clean_coords = pca.fit_transform(clean_dataset.vectors)
    corr_coords = pca.transform(corrupted_dataset.vectors)

    disp_vecs: list[list[float]] = []
    disp_mags: list[float] = []

    for i in range(n):
        dv = [
            corr_coords[i][j] - clean_coords[i][j] for j in range(len(clean_coords[i]))
        ]
        mag = math.sqrt(sum(x * x for x in dv))
        disp_vecs.append(dv)
        disp_mags.append(mag)

    shared_pca = SharedPCAProjectionResult(
        original_dim=clean_dataset.feature_dim,
        projected_dim=n_pca_components,
        num_samples=n,
        sample_ids=list(clean_dataset.sample_ids),
        labels=list(clean_dataset.labels),
        clean_coordinates=clean_coords,
        corrupted_coordinates=corr_coords,
        displacement_vectors=disp_vecs,
        displacement_magnitudes=disp_mags,
        explained_variance_ratio=pca.explained_variance_ratio or [],
        cumulative_explained_variance=pca.cumulative_explained_variance or [],
    )

    ratio_delta = (
        corr_cent.separation_to_compactness_ratio
        - clean_cent.separation_to_compactness_ratio
    )

    return GeometryDriftReport(
        clean_centroid_report=clean_cent,
        corrupted_centroid_report=corr_cent,
        mean_centroid_displacement=mean_disp,
        class_centroid_drifts=class_drifts,
        neighborhood_drift=neigh_drift,
        shared_pca=shared_pca,
        separation_to_compactness_ratio_delta=ratio_delta,
    )
