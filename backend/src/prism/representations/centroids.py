"""Class centroid geometry, intra-class compactness, and inter-class separation."""

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
from prism.representations.geometry import (
    DistanceMetric,
    RepresentationDataset,
    compute_distance,
)


class ClassCentroidSummary(BaseModel):
    """Geometric and compactness properties of a single class representation cluster."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    class_id: str = Field(description="Unique class identifier or label string")
    class_name: str = Field(description="Human readable class name")
    sample_count: int = Field(ge=1, description="Number of samples in this class")
    centroid: list[float] = Field(description="Mean feature vector for this class [D]")
    centroid_norm: float = Field(
        ge=0.0, description="L2 norm of the class centroid vector"
    )
    intra_class_mean_distance: float = Field(
        ge=0.0, description="Mean distance from class samples to the class centroid"
    )
    intra_class_std_distance: float = Field(
        ge=0.0, description="Standard deviation of distance from samples to centroid"
    )
    intra_class_max_distance: float = Field(
        ge=0.0, description="Maximum distance from any class sample to centroid"
    )
    intra_class_radius_90: float = Field(
        ge=0.0, description="90th percentile distance covering 90% of class samples"
    )
    nearest_competing_class: str | None = Field(
        default=None, description="Class ID of the closest foreign class centroid"
    )
    distance_to_nearest_competing_centroid: float | None = Field(
        default=None, description="Distance to closest foreign class centroid"
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert centroid summary to dictionary."""
        return self.model_dump(mode="json")


class CentroidGeometryReport(BaseModel):
    """Comprehensive summary of intra-class compactness and inter-class separation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    metric: DistanceMetric = Field(description="Distance metric used for calculations")
    class_order: list[str] = Field(description="Ordered list of class IDs")
    class_centroids: dict[str, ClassCentroidSummary] = Field(
        description="Per-class centroid summaries keyed by class ID"
    )
    centroid_distance_matrix: list[list[float]] = Field(
        description="Pairwise distance matrix between class centroids [C, C]"
    )
    mean_intra_class_distance: float = Field(
        ge=0.0,
        description="Average intra-class sample-to-centroid distance",
    )
    mean_inter_class_centroid_distance: float = Field(
        ge=0.0,
        description="Average distance between distinct class centroids",
    )
    min_inter_class_centroid_distance: float = Field(
        ge=0.0,
        description="Minimum distance between distinct class centroids",
    )
    separation_to_compactness_ratio: float = Field(
        ge=0.0,
        description="Ratio of inter-class separation to intra-class compactness",
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
    def from_dict(cls, data: dict[str, Any]) -> CentroidGeometryReport:
        """Create report from dictionary."""
        try:
            return cls.model_validate(data)
        except Exception as exc:
            raise SerializationError(
                f"Failed to deserialize CentroidGeometryReport from dict: {exc}"
            ) from exc


def compute_centroid_geometry(
    dataset: RepresentationDataset,
    metric: DistanceMetric | str = DistanceMetric.EUCLIDEAN,
    eps: float = 1e-12,
) -> CentroidGeometryReport:
    """Compute class centroids, intra-class compactness, and inter-class separation.

    Parameters
    ----------
    dataset : RepresentationDataset
        Input representation dataset.
    metric : DistanceMetric | str
        Distance metric.
    eps : float
        Small numerical stability epsilon.

    Returns
    -------
    CentroidGeometryReport
        Calculated centroid geometry report.
    """
    if dataset.num_samples == 0:
        raise ValidationError("Cannot compute centroid geometry on empty dataset.")

    metric_enum = DistanceMetric(metric) if isinstance(metric, str) else metric
    d = dataset.feature_dim

    # Group sample indices by class
    class_groups: dict[str, list[int]] = {}
    class_names_map: dict[str, str] = {}

    for idx, label in enumerate(dataset.labels):
        c_key = str(label)
        if c_key not in class_groups:
            class_groups[c_key] = []
            c_name = (
                dataset.class_names[len(class_groups) - 1]
                if len(dataset.class_names) >= len(class_groups)
                else f"class_{c_key}"
            )
            class_names_map[c_key] = c_name
        class_groups[c_key].append(idx)

    class_order = sorted(class_groups.keys())
    num_classes = len(class_order)

    # 1. Compute Centroid Vectors
    centroids: dict[str, list[float]] = {}
    for c_id in class_order:
        indices = class_groups[c_id]
        k_samples = len(indices)
        c_vec = [0.0] * d
        for idx in indices:
            row = dataset.vectors[idx]
            for j in range(d):
                c_vec[j] += row[j]
        for j in range(d):
            c_vec[j] /= float(k_samples)
        centroids[c_id] = c_vec

    # 2. Compute Centroid-to-Centroid Distances
    centroid_dist_matrix: list[list[float]] = [
        [0.0] * num_classes for _ in range(num_classes)
    ]
    inter_distances: list[float] = []

    for i in range(num_classes):
        c1 = class_order[i]
        for j in range(i + 1, num_classes):
            c2 = class_order[j]
            dist = compute_distance(centroids[c1], centroids[c2], metric=metric_enum)
            centroid_dist_matrix[i][j] = dist
            centroid_dist_matrix[j][i] = dist
            inter_distances.append(dist)

    # 3. Compute Intra-Class Compactness per Class
    class_summaries: dict[str, ClassCentroidSummary] = {}
    all_intra_distances: list[float] = []

    for c_idx, c_id in enumerate(class_order):
        c_vec = centroids[c_id]
        indices = class_groups[c_id]
        sample_count = len(indices)

        # Distances to own centroid
        dists: list[float] = []
        for idx in indices:
            d_val = compute_distance(dataset.vectors[idx], c_vec, metric=metric_enum)
            dists.append(d_val)
            all_intra_distances.append(d_val)

        mean_d = sum(dists) / sample_count
        var_d = (
            sum((x - mean_d) ** 2 for x in dists) / sample_count
            if sample_count > 1
            else 0.0
        )
        std_d = math.sqrt(max(0.0, var_d))
        max_d = max(dists) if dists else 0.0

        sorted_d = sorted(dists)
        r90_idx = min(sample_count - 1, math.ceil(0.90 * sample_count) - 1)
        r90 = sorted_d[r90_idx] if sorted_d else 0.0

        c_norm = math.sqrt(sum(x * x for x in c_vec))

        # Nearest competing class
        nearest_competing: str | None = None
        min_comp_dist: float | None = None

        if num_classes > 1:
            for other_idx, other_id in enumerate(class_order):
                if other_idx == c_idx:
                    continue
                dist_to_other = centroid_dist_matrix[c_idx][other_idx]
                if min_comp_dist is None or dist_to_other < min_comp_dist:
                    min_comp_dist = dist_to_other
                    nearest_competing = other_id

        class_summaries[c_id] = ClassCentroidSummary(
            class_id=c_id,
            class_name=class_names_map[c_id],
            sample_count=sample_count,
            centroid=c_vec,
            centroid_norm=c_norm,
            intra_class_mean_distance=mean_d,
            intra_class_std_distance=std_d,
            intra_class_max_distance=max_d,
            intra_class_radius_90=r90,
            nearest_competing_class=nearest_competing,
            distance_to_nearest_competing_centroid=min_comp_dist,
        )

    # 4. Overall Aggregate Statistics
    overall_mean_intra = (
        sum(all_intra_distances) / len(all_intra_distances)
        if all_intra_distances
        else 0.0
    )
    overall_mean_inter = (
        sum(inter_distances) / len(inter_distances) if inter_distances else 0.0
    )
    min_inter = min(inter_distances) if inter_distances else 0.0

    sep_to_comp = overall_mean_inter / (overall_mean_intra + eps)

    return CentroidGeometryReport(
        metric=metric_enum,
        class_order=class_order,
        class_centroids=class_summaries,
        centroid_distance_matrix=centroid_dist_matrix,
        mean_intra_class_distance=overall_mean_intra,
        mean_inter_class_centroid_distance=overall_mean_inter,
        min_inter_class_centroid_distance=min_inter,
        separation_to_compactness_ratio=sep_to_comp,
    )
