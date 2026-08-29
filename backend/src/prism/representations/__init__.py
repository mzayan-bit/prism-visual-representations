"""Representation extraction, feature distribution summaries, and geometry."""

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
from prism.representations.centroids import (
    CentroidGeometryReport,
    ClassCentroidSummary,
    compute_centroid_geometry,
)
from prism.representations.contracts import (
    RepresentationBatch,
    RepresentationDescriptor,
)
from prism.representations.geometry import (
    DistanceMetric,
    RepresentationDataset,
    SpatialVectorizationPolicy,
    VectorNormalizationPolicy,
    compute_distance,
    compute_pairwise_distances,
    normalize_vectors,
    vectorize_spatial_features,
)
from prism.representations.neighborhood import (
    CandidateFailureCase,
    NearestNeighborEntry,
    NeighborhoodGeometrySummary,
    SampleNeighborhood,
    compute_neighborhood_geometry,
)
from prism.representations.pca import (
    PrincipalComponentAnalysis,
    ProjectionResult,
    compute_pca_projection,
)
from prism.representations.reports import (
    RepresentationGeometryReport,
    VectorNormSummary,
    analyze_representation_geometry,
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
    "CandidateFailureCase",
    "CentroidGeometryReport",
    "ClassCentroidSummary",
    "DistanceMetric",
    "FeatureDistributionSummary",
    "NearestNeighborEntry",
    "NeighborhoodGeometrySummary",
    "PrincipalComponentAnalysis",
    "ProjectionResult",
    "RepresentationBatch",
    "RepresentationDataset",
    "RepresentationDescriptor",
    "RepresentationGeometryReport",
    "SampleNeighborhood",
    "SpatialVectorizationPolicy",
    "TransformerAttentionProfile",
    "VectorNormSummary",
    "VectorNormalizationPolicy",
    "analyze_representation_geometry",
    "compare_attention_summaries",
    "compare_distribution_summaries",
    "compare_transformer_attention_profiles",
    "compute_attention_entropy",
    "compute_attention_summary",
    "compute_centroid_geometry",
    "compute_diagonal_attention_mass",
    "compute_distance",
    "compute_distribution_summary",
    "compute_neighborhood_geometry",
    "compute_pairwise_distances",
    "compute_pca_projection",
    "compute_transformer_attention_profile",
    "normalize_vectors",
    "summarize_attention_weights",
    "vectorize_spatial_features",
]
