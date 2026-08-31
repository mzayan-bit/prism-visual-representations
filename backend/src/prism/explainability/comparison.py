"""Cross-method attribution agreement and comparison reports."""

from __future__ import annotations

import json
import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from prism.core.errors import SerializationError, ValidationError
from prism.explainability.attribution import (
    AttributionMethod,
    AttributionResult,
    AttributionStatistics,
)


def compute_map_cosine_similarity(
    map_a: list[list[float]],
    map_b: list[list[float]],
    eps: float = 1e-12,
) -> float:
    """Compute vector cosine similarity between flattened 2D spatial maps.

    Args:
        map_a: First 2D attribution map [H, W].
        map_b: Second 2D attribution map [H, W].
        eps: Numerical stabilizer for zero vectors.

    Returns:
        Cosine similarity in [-1.0, 1.0].
    """
    if not map_a or not map_b:
        raise ValidationError("Maps must be non-empty.")

    ha, wa = len(map_a), len(map_a[0])
    hb, wb = len(map_b), len(map_b[0])
    if ha != hb or wa != wb:
        raise ValidationError(
            f"Shape mismatch in cosine similarity: ({ha}, {wa}) vs ({hb}, {wb})."
        )

    flat_a = [map_a[r][c] for r in range(ha) for c in range(wa)]
    flat_b = [map_b[r][c] for r in range(hb) for c in range(wb)]

    dot_prod = sum(a * b for a, b in zip(flat_a, flat_b, strict=False))
    norm_a = math.sqrt(sum(a * a for a in flat_a))
    norm_b = math.sqrt(sum(b * b for b in flat_b))

    if norm_a < eps or norm_b < eps:
        return 0.0

    cos_val = dot_prod / (norm_a * norm_b)
    return float(max(-1.0, min(1.0, cos_val)))


def create_top_percent_mask(
    map_2d: list[list[float]],
    top_percent: float = 0.10,
) -> list[list[int]]:
    """Create deterministic binary mask of top-p% pixels by attribution magnitude.

    Ties are broken deterministically by (row, col) raster order.

    Args:
        map_2d: 2D attribution map [H, W].
        top_percent: Fraction of pixels in [0, 1] to select (e.g. 0.10 for top 10%).

    Returns:
        2D binary integer matrix [H, W] with 1 at top positions and 0 elsewhere.
    """
    if not map_2d or not map_2d[0]:
        raise ValidationError("map_2d must be non-empty.")

    h = len(map_2d)
    w = len(map_2d[0])
    total_pixels = h * w
    k = max(1, math.ceil(top_percent * total_pixels))

    indexed_values: list[tuple[float, int, int]] = []
    for r in range(h):
        for c in range(w):
            indexed_values.append((abs(map_2d[r][c]), r, c))

    # Sort primarily by absolute value descending, secondarily by row asc, col asc
    indexed_values.sort(key=lambda item: (-item[0], item[1], item[2]))

    selected_coords = {(r, c) for _, r, c in indexed_values[:k]}

    mask: list[list[int]] = []
    for r in range(h):
        row = [1 if (r, c) in selected_coords else 0 for c in range(w)]
        mask.append(row)

    return mask


def compute_top_percent_overlap(
    map_a: list[list[float]],
    map_b: list[list[float]],
    top_percent: float = 0.10,
) -> float:
    """Compute Jaccard overlap (Intersection / Union) of top-p% attribution masks.

    Args:
        map_a: First 2D attribution map [H, W].
        map_b: Second 2D attribution map [H, W].
        top_percent: Percentile fraction (default 0.10).

    Returns:
        Jaccard overlap coefficient in [0.0, 1.0].
    """
    mask_a = create_top_percent_mask(map_a, top_percent=top_percent)
    mask_b = create_top_percent_mask(map_b, top_percent=top_percent)

    h = len(mask_a)
    w = len(mask_a[0])

    intersection_count = 0
    union_count = 0

    for r in range(h):
        for c in range(w):
            val_a = mask_a[r][c]
            val_b = mask_b[r][c]
            if val_a == 1 and val_b == 1:
                intersection_count += 1
            if val_a == 1 or val_b == 1:
                union_count += 1

    if union_count == 0:
        return 1.0

    return float(intersection_count / float(union_count))


def compute_center_of_mass_displacement(
    stats_a: AttributionStatistics,
    stats_b: AttributionStatistics,
) -> float:
    """Compute Euclidean distance between center-of-mass spatial coordinates."""
    dr = stats_a.center_of_mass_row - stats_b.center_of_mass_row
    dc = stats_a.center_of_mass_col - stats_b.center_of_mass_col
    return float(math.sqrt(dr * dr + dc * dc))


class MethodAgreementResult(BaseModel):
    """Pairwise agreement metrics between two attribution methods."""

    model_config = ConfigDict(extra="forbid")

    method_a: AttributionMethod = Field(description="First attribution method")
    method_b: AttributionMethod = Field(description="Second attribution method")
    cosine_similarity: float = Field(
        description="Cosine similarity between normalized attribution heatmaps"
    )
    top_10_percent_overlap: float = Field(
        description="Jaccard overlap of top-10% binary attribution masks"
    )
    top_25_percent_overlap: float = Field(
        description="Jaccard overlap of top-25% binary attribution masks"
    )
    center_of_mass_displacement: float = Field(
        description="Euclidean distance between spatial centers of mass"
    )
    concentration_difference: float = Field(
        description="Absolute difference in spatial concentration |C_a - C_b|"
    )


class AttributionComparisonReport(BaseModel):
    """Comparative report analyzing multiple attribution signals."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(description="Sample identifier")
    model_id: str = Field(description="Model identifier")
    architecture: str = Field(description="Model architecture family")
    target_class: int = Field(description="Target class explained across methods")
    predicted_class: int = Field(description="Model's top-1 predicted class")
    true_class: int | None = Field(default=None, description="Ground truth label")
    results: dict[str, AttributionResult] = Field(
        description="AttributionResult indexed by method key"
    )
    pairwise_agreements: list[MethodAgreementResult] = Field(
        description="List of all evaluated pairwise method agreements"
    )
    cosine_similarity_matrix: dict[str, dict[str, float | None]] = Field(
        description="Pairwise cosine similarity lookup matrix"
    )
    top_10_overlap_matrix: dict[str, dict[str, float | None]] = Field(
        description="Pairwise top-10% mask overlap lookup matrix"
    )
    mean_cross_method_agreement: float = Field(
        description="Average pairwise cosine similarity across all method pairs"
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Descriptive warnings regarding disagreement or missing signals",
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
    def from_dict(cls, data: dict[str, Any]) -> AttributionComparisonReport:
        """Construct report from dictionary."""
        try:
            return cls.model_validate(data)
        except Exception as exc:
            raise SerializationError(
                f"Failed to deserialize AttributionComparisonReport: {exc}"
            ) from exc


def compare_attributions(
    results: list[AttributionResult],
) -> AttributionComparisonReport:
    """Generate comparative agreement report across a suite of attribution results.

    Args:
        results: List of AttributionResult objects computed for the same sample.

    Returns:
        Comprehensive AttributionComparisonReport.
    """
    if not results:
        raise ValidationError("Cannot compare an empty list of AttributionResults.")

    sample_id = results[0].sample_id
    model_id = results[0].model_id
    architecture = results[0].architecture
    target_class = results[0].target_class
    predicted_class = results[0].predicted_class
    true_class = results[0].true_class

    result_dict: dict[str, AttributionResult] = {}
    for res in results:
        result_dict[res.method.value] = res

    method_keys = list(result_dict.keys())
    pairwise_agreements: list[MethodAgreementResult] = []
    cos_matrix: dict[str, dict[str, float | None]] = {m: {} for m in method_keys}
    top_10_matrix: dict[str, dict[str, float | None]] = {m: {} for m in method_keys}

    # Initialize diagonals to 1.0
    for m in method_keys:
        cos_matrix[m][m] = 1.0
        top_10_matrix[m][m] = 1.0

    pair_similarities: list[float] = []

    for i in range(len(method_keys)):
        for j in range(i + 1, len(method_keys)):
            m_a = method_keys[i]
            m_b = method_keys[j]
            res_a = result_dict[m_a]
            res_b = result_dict[m_b]

            cos_sim = compute_map_cosine_similarity(
                res_a.normalized_attribution_map,
                res_b.normalized_attribution_map,
            )
            top_10_ovlp = compute_top_percent_overlap(
                res_a.normalized_attribution_map,
                res_b.normalized_attribution_map,
                top_percent=0.10,
            )
            top_25_ovlp = compute_top_percent_overlap(
                res_a.normalized_attribution_map,
                res_b.normalized_attribution_map,
                top_percent=0.25,
            )
            com_disp = compute_center_of_mass_displacement(
                res_a.statistics, res_b.statistics
            )
            conc_diff = abs(
                res_a.statistics.concentration_score
                - res_b.statistics.concentration_score
            )

            agreement = MethodAgreementResult(
                method_a=res_a.method,
                method_b=res_b.method,
                cosine_similarity=cos_sim,
                top_10_percent_overlap=top_10_ovlp,
                top_25_percent_overlap=top_25_ovlp,
                center_of_mass_displacement=com_disp,
                concentration_difference=conc_diff,
            )
            pairwise_agreements.append(agreement)

            cos_matrix[m_a][m_b] = cos_sim
            cos_matrix[m_b][m_a] = cos_sim
            top_10_matrix[m_a][m_b] = top_10_ovlp
            top_10_matrix[m_b][m_a] = top_10_ovlp
            pair_similarities.append(cos_sim)

    mean_agreement = (
        float(sum(pair_similarities) / len(pair_similarities))
        if pair_similarities
        else 1.0
    )

    warnings: list[str] = []
    if mean_agreement < 0.20 and len(method_keys) > 1:
        warnings.append("high_method_disagreement")

    return AttributionComparisonReport(
        sample_id=sample_id,
        model_id=model_id,
        architecture=architecture,
        target_class=target_class,
        predicted_class=predicted_class,
        true_class=true_class,
        results=result_dict,
        pairwise_agreements=pairwise_agreements,
        cosine_similarity_matrix=cos_matrix,
        top_10_overlap_matrix=top_10_matrix,
        mean_cross_method_agreement=mean_agreement,
        warnings=warnings,
    )
