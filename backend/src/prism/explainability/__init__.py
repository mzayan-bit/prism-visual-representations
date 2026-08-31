"""PRISM Explainability & Visual Attribution Subsystem."""

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
from prism.explainability.gradients import (
    compute_gradient_x_input,
    compute_input_gradient_saliency,
)
from prism.explainability.occlusion import (
    compute_occlusion_sensitivity,
)

__all__ = [
    "AttributionMethod",
    "AttributionNormalizationPolicy",
    "AttributionResult",
    "AttributionSpecification",
    "AttributionStatistics",
    "ChannelReductionPolicy",
    "OcclusionFillPolicy",
    "TargetClassMode",
    "ViTAttentionHeadPolicy",
    "compute_attribution_statistics",
    "compute_gradient_x_input",
    "compute_input_gradient_saliency",
    "compute_occlusion_sensitivity",
    "normalize_attribution_map",
    "reduce_channels",
]
