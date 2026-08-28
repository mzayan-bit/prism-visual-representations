"""Representation extraction, feature distribution summaries, and stability analysis."""

from prism.representations.attention import (
    AttentionHeadSummary,
    AttentionTensorSummary,
    summarize_attention_weights,
)
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
    "AttentionHeadSummary",
    "AttentionTensorSummary",
    "FeatureDistributionSummary",
    "RepresentationBatch",
    "RepresentationDescriptor",
    "compare_distribution_summaries",
    "compute_distribution_summary",
    "summarize_attention_weights",
]
