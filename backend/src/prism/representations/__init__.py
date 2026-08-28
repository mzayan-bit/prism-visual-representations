"""Representation extraction, feature distribution summaries, and stability analysis."""

from prism.representations.contracts import (
    RepresentationBatch,
    RepresentationDescriptor,
)
from prism.representations.summaries import (
    FeatureDistributionSummary,
    compare_distribution_summaries,
    compute_distribution_summary,
)

__all__ = [
    "FeatureDistributionSummary",
    "RepresentationBatch",
    "RepresentationDescriptor",
    "compare_distribution_summaries",
    "compute_distribution_summary",
]
