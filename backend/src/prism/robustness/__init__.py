"""Robustness evaluation under corruptions, distribution shifts, and perturbations."""

from prism.robustness.attention_drift import (
    AttentionDriftSummary,
    LayerAttentionDrift,
    compute_vit_attention_drift,
)
from prism.robustness.corruptions import (
    SEVERITY_PARAMETER_MAPS,
    CorruptedDatasetView,
    CorruptionSpecification,
    CorruptionType,
    apply_brightness_shift,
    apply_contrast_shift,
    apply_corruption,
    apply_gaussian_noise,
    apply_rectangular_occlusion,
    apply_resolution_degradation,
    apply_spatial_blur,
)
from prism.robustness.drift import (
    RepresentationDriftSummary,
    SampleRepresentationDrift,
    compute_representation_drift,
)
from prism.robustness.evaluation import (
    ArchitectureRobustnessSummary,
    CorruptionEvaluationSummary,
    CorruptionSeverityCurve,
    CorruptionSuite,
    CrossArchitectureRobustnessReport,
    RobustnessExperimentReport,
    RobustnessFailureCategory,
    RobustnessFailureRecord,
    RobustnessSuiteRunner,
    compare_architecture_robustness,
)
from prism.robustness.geometry_drift import (
    ClassCentroidDriftSummary,
    GeometryDriftReport,
    NeighborhoodDriftSummary,
    SharedPCAProjectionResult,
    compute_geometry_drift,
)

__all__ = [
    "SEVERITY_PARAMETER_MAPS",
    "ArchitectureRobustnessSummary",
    "AttentionDriftSummary",
    "ClassCentroidDriftSummary",
    "CorruptedDatasetView",
    "CorruptionEvaluationSummary",
    "CorruptionSeverityCurve",
    "CorruptionSpecification",
    "CorruptionSuite",
    "CorruptionType",
    "CrossArchitectureRobustnessReport",
    "GeometryDriftReport",
    "LayerAttentionDrift",
    "NeighborhoodDriftSummary",
    "RepresentationDriftSummary",
    "RobustnessExperimentReport",
    "RobustnessFailureCategory",
    "RobustnessFailureRecord",
    "RobustnessSuiteRunner",
    "SampleRepresentationDrift",
    "SharedPCAProjectionResult",
    "apply_brightness_shift",
    "apply_contrast_shift",
    "apply_corruption",
    "apply_gaussian_noise",
    "apply_rectangular_occlusion",
    "apply_resolution_degradation",
    "apply_spatial_blur",
    "compare_architecture_robustness",
    "compute_geometry_drift",
    "compute_representation_drift",
    "compute_vit_attention_drift",
]
