"""Representation extraction, feature distribution summaries, and stability analysis."""

from prism.representations.attention import (
    AttentionHeadSummary,
    AttentionPattern,
    AttentionTensorSummary,
    AttentionWeightSummary,
    TransformerAttentionProfile,
    compare_attention_summaries,
    compare_transformer_attention_profiles,
    compute_attention_entropy,
    compute_attention_summary,
    compute_diagonal_attention_mass,
    compute_transformer_attention_profile,
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
    "AttentionPattern",
    "AttentionTensorSummary",
    "AttentionWeightSummary",
    "FeatureDistributionSummary",
    "RepresentationBatch",
    "RepresentationDescriptor",
    "TransformerAttentionProfile",
    "compare_attention_summaries",
    "compare_distribution_summaries",
    "compare_transformer_attention_profiles",
    "compute_attention_entropy",
    "compute_attention_summary",
    "compute_diagonal_attention_mass",
    "compute_distribution_summary",
    "compute_transformer_attention_profile",
    "summarize_attention_weights",
]
