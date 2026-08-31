"""PRISM Explainability & Visual Attribution Subsystem."""

from prism.explainability.attention_attribution import (
    compute_vit_attention_attribution,
)
from prism.explainability.attribution import (
    AttributionMethod,
    AttributionNormalizationPolicy,
    AttributionResult,
    AttributionSpecification,
    AttributionStatistics,
    ChannelReductionPolicy,
    OcclusionFillPolicy,
    TargetClassMode,
    ViTAttentionHeadPolicy,
    compute_attribution_statistics,
    normalize_attribution_map,
    reduce_channels,
)
from prism.explainability.comparison import (
    AttributionComparisonReport,
    MethodAgreementResult,
    compare_attributions,
    compute_center_of_mass_displacement,
    compute_map_cosine_similarity,
    compute_top_percent_overlap,
    create_top_percent_mask,
)
from prism.explainability.drift import (
    AttributionDriftSummary,
    compute_attribution_drift,
)
from prism.explainability.failures import (
    ExplanationFailureCategory,
    ExplanationFailureFlag,
    flag_explanation_failures,
)
from prism.explainability.grad_cam import (
    compute_grad_cam,
    upsample_bilinear_2d,
)
from prism.explainability.gradients import (
    compute_gradient_x_input,
    compute_input_gradient_saliency,
)
from prism.explainability.occlusion import (
    compute_occlusion_sensitivity,
)

__all__ = [
    "AttributionComparisonReport",
    "AttributionDriftSummary",
    "AttributionMethod",
    "AttributionNormalizationPolicy",
    "AttributionResult",
    "AttributionSpecification",
    "AttributionStatistics",
    "ChannelReductionPolicy",
    "ExplanationFailureCategory",
    "ExplanationFailureFlag",
    "MethodAgreementResult",
    "OcclusionFillPolicy",
    "TargetClassMode",
    "ViTAttentionHeadPolicy",
    "compare_attributions",
    "compute_attribution_drift",
    "compute_attribution_statistics",
    "compute_center_of_mass_displacement",
    "compute_grad_cam",
    "compute_gradient_x_input",
    "compute_input_gradient_saliency",
    "compute_map_cosine_similarity",
    "compute_occlusion_sensitivity",
    "compute_top_percent_overlap",
    "compute_vit_attention_attribution",
    "create_top_percent_mask",
    "flag_explanation_failures",
    "normalize_attribution_map",
    "reduce_channels",
    "upsample_bilinear_2d",
]
