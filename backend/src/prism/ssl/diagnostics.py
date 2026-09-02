"""Diagnostics for representation collapse, alignment, and uniformity in SSL."""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RepresentationCollapseSummary(BaseModel):
    """Structured diagnostics assessing representation diversity and collapse."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    total_dimensions: int = Field(..., description="Dimensionality of feature space D")
    mean_feature_std: float = Field(
        ..., description="Average standard deviation across all feature dimensions"
    )
    near_zero_variance_dimensions: int = Field(
        ..., description="Count of dimensions with variance below collapse threshold"
    )
    near_zero_variance_fraction: float = Field(
        ..., description="Fraction of feature dimensions that are collapsed"
    )
    distinct_sample_cosine_spread: float = Field(
        ..., description="Average pairwise cosine similarity across distinct samples"
    )
    mean_positive_alignment_distance: float = Field(
        ..., description="Mean squared Euclidean distance between positive pair views"
    )
    is_collapsed: bool = Field(
        ..., description="True if representation meets dimensional collapse criteria"
    )
    warnings: list[str] = Field(
        default_factory=list, description="Methodological and diagnostic warnings"
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert summary to dictionary."""
        return self.model_dump()


def compute_collapse_diagnostics(
    representations: list[list[float]],
    variance_threshold: float = 1e-4,
    positive_views: tuple[list[list[float]], list[list[float]]] | None = None,
) -> RepresentationCollapseSummary:
    """Compute dimensional variance, feature standard deviation, and collapse status."""
    n_samples = len(representations)
    if n_samples == 0:
        return RepresentationCollapseSummary(
            total_dimensions=0,
            mean_feature_std=0.0,
            near_zero_variance_dimensions=0,
            near_zero_variance_fraction=1.0,
            distinct_sample_cosine_spread=1.0,
            mean_positive_alignment_distance=0.0,
            is_collapsed=True,
            warnings=["No representation samples provided."],
        )

    dim = len(representations[0])
    stds: list[float] = []
    near_zero_count = 0

    for d in range(dim):
        vals = [representations[i][d] for i in range(n_samples)]
        mean_val = sum(vals) / float(n_samples)
        var = sum((x - mean_val) ** 2 for x in vals) / float(n_samples)
        std = math.sqrt(max(0.0, var))
        stds.append(std)
        if var < variance_threshold:
            near_zero_count += 1

    mean_std = sum(stds) / float(dim) if dim > 0 else 0.0
    near_zero_fraction = float(near_zero_count) / float(dim) if dim > 0 else 0.0

    # Distinct sample pairwise cosine similarity
    distinct_sims: list[float] = []
    max_pairs = min(n_samples, 20)
    for i in range(max_pairs):
        norm_i = math.sqrt(sum(x * x for x in representations[i])) + 1e-8
        for j in range(i + 1, max_pairs):
            norm_j = math.sqrt(sum(x * x for x in representations[j])) + 1e-8
            dot = sum(representations[i][k] * representations[j][k] for k in range(dim))
            distinct_sims.append(dot / (norm_i * norm_j))

    spread = sum(distinct_sims) / float(len(distinct_sims)) if distinct_sims else 0.0

    # Positive pair alignment distance
    pos_dist = 0.0
    if positive_views is not None and len(positive_views[0]) > 0:
        v_a, v_b = positive_views
        p_n = min(len(v_a), len(v_b))
        dists: list[float] = []
        for i in range(p_n):
            sq_d = sum((v_a[i][k] - v_b[i][k]) ** 2 for k in range(dim))
            dists.append(sq_d)
        pos_dist = sum(dists) / float(p_n) if dists else 0.0

    warnings: list[str] = []
    is_collapsed = near_zero_fraction > 0.8 or mean_std < 1e-3

    if near_zero_fraction > 0.5:
        warnings.append(
            f"High dimensional collapse: {near_zero_fraction:.1%} "
            f"of feature dimensions have near-zero variance."
        )
    if spread > 0.95:
        warnings.append(
            f"Representations lack angular spread: mean pairwise "
            f"cosine similarity is {spread:.3f}."
        )

    return RepresentationCollapseSummary(
        total_dimensions=dim,
        mean_feature_std=mean_std,
        near_zero_variance_dimensions=near_zero_count,
        near_zero_variance_fraction=near_zero_fraction,
        distinct_sample_cosine_spread=spread,
        mean_positive_alignment_distance=pos_dist,
        is_collapsed=is_collapsed,
        warnings=warnings,
    )
