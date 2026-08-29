"""Nearest-neighbor geometry, label consistency, and candidate failure discovery."""

from __future__ import annotations

import json
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


class NearestNeighborEntry(BaseModel):
    """Ranked nearest neighbor entry for a query sample."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rank: int = Field(ge=1, description="Rank index (1 is closest neighbor)")
    neighbor_sample_id: str = Field(description="Sample ID of neighbor")
    neighbor_label: int | str = Field(description="Class label of neighbor")
    distance: float = Field(ge=0.0, description="Distance from query to neighbor")
    same_class: bool = Field(
        description="True if neighbor shares identical label with query"
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert entry to dictionary."""
        return self.model_dump(mode="json")


class SampleNeighborhood(BaseModel):
    """Local neighborhood context and nearest neighbors of an individual sample."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query_sample_id: str = Field(description="Sample ID of query representation")
    query_label: int | str = Field(description="Ground truth label of query sample")
    neighbors: list[NearestNeighborEntry] = Field(
        description="List of top-k nearest neighbors in ascending distance order"
    )
    same_class_fraction: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of top-k neighbors sharing the query class label",
    )
    distance_to_own_centroid: float | None = Field(
        default=None, description="Distance from query sample to its class centroid"
    )
    nearest_competing_centroid_distance: float | None = Field(
        default=None,
        description="Distance from query sample to the closest foreign class centroid",
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert neighborhood to dictionary."""
        return self.model_dump(mode="json")


class CandidateFailureCase(BaseModel):
    """Identified candidate failure or geometrically ambiguous sample."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_id: str = Field(description="Sample identifier")
    label: int | str = Field(description="True class label")
    failure_kind: str = Field(
        description="Category: 'cross_class_neighbor', 'low_consistency', "
        "'centroid_outlier', 'closer_to_foreign_centroid'"
    )
    description: str = Field(description="Human readable explanation")
    metric_value: float = Field(description="Quantitative metric triggering detection")

    def to_dict(self) -> dict[str, Any]:
        """Convert failure case to dictionary."""
        return self.model_dump(mode="json")


class NeighborhoodGeometrySummary(BaseModel):
    """Statistical summary of k-NN geometry and neighborhood consistency."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    k: int = Field(ge=1, description="Number of nearest neighbors evaluated")
    metric: DistanceMetric = Field(description="Distance metric used")
    mean_label_consistency: float = Field(
        ge=0.0,
        le=1.0,
        description="Mean fraction of same-class neighbors across all samples",
    )
    median_label_consistency: float = Field(
        ge=0.0,
        le=1.0,
        description="Median fraction of same-class neighbors across all samples",
    )
    per_class_label_consistency: dict[str, float] = Field(
        description="Average neighborhood consistency per class"
    )
    candidate_failures: list[CandidateFailureCase] = Field(
        default_factory=list,
        description="List of detected candidate failure samples",
    )
    sample_neighborhoods: dict[str, SampleNeighborhood] = Field(
        default_factory=dict,
        description="Neighborhood records keyed by sample ID",
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert summary to dictionary."""
        return self.model_dump(mode="json")

    def to_json(self, indent: int | None = None) -> str:
        """Convert summary to JSON string."""
        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=True,
            ensure_ascii=True,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NeighborhoodGeometrySummary:
        """Create summary from dictionary."""
        try:
            return cls.model_validate(data)
        except Exception as exc:
            raise SerializationError(
                f"Failed to deserialize NeighborhoodGeometrySummary from dict: {exc}"
            ) from exc


def compute_neighborhood_geometry(
    dataset: RepresentationDataset,
    k: int = 5,
    metric: DistanceMetric | str = DistanceMetric.EUCLIDEAN,
    centroid_report: CentroidGeometryReport | None = None,
    consistency_threshold: float = 0.5,
) -> NeighborhoodGeometrySummary:
    """Compute exact in-memory nearest neighbors and neighborhood consistency.

    Parameters
    ----------
    dataset : RepresentationDataset
        Input representation dataset.
    k : int
        Number of nearest neighbors to retrieve (excluding self).
    metric : DistanceMetric | str
        Distance metric.
    centroid_report : CentroidGeometryReport | None
        Optional precomputed centroid geometry report.
    consistency_threshold : float
        Threshold for flagging low neighborhood consistency candidate failures.

    Returns
    -------
    NeighborhoodGeometrySummary
        Calculated neighborhood geometry summary.
    """
    n = dataset.num_samples
    if n == 0:
        raise ValidationError("Cannot compute neighborhood geometry on empty dataset.")

    metric_enum = DistanceMetric(metric) if isinstance(metric, str) else metric
    actual_k = min(k, max(1, n - 1))

    # Compute centroids if not provided
    if centroid_report is None:
        centroid_report = compute_centroid_geometry(dataset, metric=metric_enum)

    sample_neighborhoods: dict[str, SampleNeighborhood] = {}
    consistencies: list[float] = []
    class_consistencies: dict[str, list[float]] = {}
    candidate_failures: list[CandidateFailureCase] = []

    for i in range(n):
        q_id = dataset.sample_ids[i]
        q_label = dataset.labels[i]
        q_vec = dataset.vectors[i]
        q_label_str = str(q_label)

        if q_label_str not in class_consistencies:
            class_consistencies[q_label_str] = []

        # Find distances to all other samples
        cand_list: list[tuple[float, str, int | str]] = []
        for j in range(n):
            if i == j:
                continue
            d_val = compute_distance(q_vec, dataset.vectors[j], metric=metric_enum)
            cand_list.append((d_val, dataset.sample_ids[j], dataset.labels[j]))

        # Sort deterministically: primary by distance, secondary by sample_id
        cand_list.sort(key=lambda item: (item[0], item[1]))

        # Take top-k
        top_k = cand_list[:actual_k]
        entries: list[NearestNeighborEntry] = []
        same_class_count = 0

        for rank_idx, (d_val, n_id, n_lbl) in enumerate(top_k, start=1):
            is_same = n_lbl == q_label
            if is_same:
                same_class_count += 1
            entries.append(
                NearestNeighborEntry(
                    rank=rank_idx,
                    neighbor_sample_id=n_id,
                    neighbor_label=n_lbl,
                    distance=d_val,
                    same_class=is_same,
                )
            )

        same_frac = float(same_class_count) / float(actual_k) if actual_k > 0 else 1.0
        consistencies.append(same_frac)
        class_consistencies[q_label_str].append(same_frac)

        # Centroid distances for this sample
        d_to_own: float | None = None
        d_to_foreign_min: float | None = None

        if q_label_str in centroid_report.class_centroids:
            own_cent = centroid_report.class_centroids[q_label_str].centroid
            d_to_own = compute_distance(q_vec, own_cent, metric=metric_enum)

            for c_id, c_summary in centroid_report.class_centroids.items():
                if c_id == q_label_str:
                    continue
                d_c = compute_distance(q_vec, c_summary.centroid, metric=metric_enum)
                if d_to_foreign_min is None or d_c < d_to_foreign_min:
                    d_to_foreign_min = d_c

        sample_nh = SampleNeighborhood(
            query_sample_id=q_id,
            query_label=q_label,
            neighbors=entries,
            same_class_fraction=same_frac,
            distance_to_own_centroid=d_to_own,
            nearest_competing_centroid_distance=d_to_foreign_min,
        )
        sample_neighborhoods[q_id] = sample_nh

        # Failure Case Checks:
        # 1. Rank-1 neighbor is a different class
        if entries and not entries[0].same_class:
            n0_id = entries[0].neighbor_sample_id
            n0_lbl = entries[0].neighbor_label
            n0_d = entries[0].distance
            candidate_failures.append(
                CandidateFailureCase(
                    sample_id=q_id,
                    label=q_label,
                    failure_kind="cross_class_neighbor",
                    description=(
                        f"Nearest neighbor ({n0_id}) belongs to "
                        f"foreign class '{n0_lbl}' at distance {n0_d:.4f}."
                    ),
                    metric_value=n0_d,
                )
            )

        # 2. Low neighborhood consistency (< threshold)
        if same_frac < consistency_threshold:
            candidate_failures.append(
                CandidateFailureCase(
                    sample_id=q_id,
                    label=q_label,
                    failure_kind="low_consistency",
                    description=(
                        f"Neighborhood consistency ({same_frac:.2%}) is below "
                        f"threshold {consistency_threshold:.2%} across "
                        f"{actual_k} nearest neighbors."
                    ),
                    metric_value=same_frac,
                )
            )

        # 3. Closer to foreign class centroid than own class centroid
        if (
            d_to_own is not None
            and d_to_foreign_min is not None
            and d_to_foreign_min < d_to_own
        ):
            candidate_failures.append(
                CandidateFailureCase(
                    sample_id=q_id,
                    label=q_label,
                    failure_kind="closer_to_foreign_centroid",
                    description=(
                        f"Sample is closer to a foreign centroid "
                        f"({d_to_foreign_min:.4f}) than own centroid "
                        f"({d_to_own:.4f})."
                    ),
                    metric_value=d_to_own - d_to_foreign_min,
                )
            )

        # 4. Centroid outlier (> intra-class mean + 2 * std)
        if d_to_own is not None and q_label_str in centroid_report.class_centroids:
            c_info = centroid_report.class_centroids[q_label_str]
            cutoff = (
                c_info.intra_class_mean_distance + 2.0 * c_info.intra_class_std_distance
            )
            if d_to_own > cutoff and c_info.sample_count > 3:
                candidate_failures.append(
                    CandidateFailureCase(
                        sample_id=q_id,
                        label=q_label,
                        failure_kind="centroid_outlier",
                        description=(
                            f"Sample distance to own centroid ({d_to_own:.4f}) "
                            f"exceeds 2-sigma cutoff ({cutoff:.4f})."
                        ),
                        metric_value=d_to_own,
                    )
                )

    # Aggregates
    mean_cons = sum(consistencies) / len(consistencies) if consistencies else 0.0
    sorted_cons = sorted(consistencies)
    med_cons = sorted_cons[len(sorted_cons) // 2] if sorted_cons else 0.0

    per_class_cons: dict[str, float] = {}
    for c_id, vals in class_consistencies.items():
        per_class_cons[c_id] = sum(vals) / len(vals) if vals else 0.0

    return NeighborhoodGeometrySummary(
        k=actual_k,
        metric=metric_enum,
        mean_label_consistency=mean_cons,
        median_label_consistency=med_cons,
        per_class_label_consistency=per_class_cons,
        candidate_failures=candidate_failures,
        sample_neighborhoods=sample_neighborhoods,
    )
