"""Uncertainty, calibration, and OOD analysis domain package."""

from __future__ import annotations

from prism.uncertainty.calibration import (
    compute_brier_score,
    compute_calibration_report,
    compute_confidence_subset_summary,
    compute_expected_calibration_error,
    compute_maximum_calibration_error,
    compute_negative_log_likelihood,
    compute_reliability_bins,
)
from prism.uncertainty.contracts import (
    CalibrationReport,
    CalibrationSample,
    ClassCalibrationSummary,
    ConfidenceSubsetSummary,
    CorruptionUncertaintyCurve,
    OODBinaryEvaluationSummary,
    OODReferenceSet,
    OODSample,
    OODScoreResult,
    PredictionFlipUncertainty,
    PredictiveDistribution,
    ReliabilityBin,
    RepresentationConfidenceRelationship,
    TemperatureScalingResult,
    UncertaintyAnalysisReport,
)
from prism.uncertainty.corruptions import evaluate_corruption_uncertainty
from prism.uncertainty.enums import (
    BinningStrategy,
    CalibrationMode,
    ConfidenceMetric,
    OODCategory,
    OODScoreMethod,
    ThresholdPolicy,
    UncertaintyFailureType,
)
from prism.uncertainty.failures import (
    UncertaintyFailureRecord,
    detect_uncertainty_failures,
)
from prism.uncertainty.metrics import (
    compute_aupr,
    compute_auroc,
    evaluate_ood_binary_classification,
    select_ood_threshold,
)
from prism.uncertainty.ood_scores import (
    compute_class_centroid_ood_score,
    compute_energy_ood_score,
    compute_entropy_ood_score,
    compute_knn_ood_score,
    compute_max_softmax_ood_score,
    score_ood_sample,
)
from prism.uncertainty.probabilities import (
    batch_predictive_distributions,
    compute_logit_margin,
    compute_normalized_entropy,
    compute_predictive_distribution,
    compute_predictive_entropy,
    compute_probability_margin,
    compute_stable_softmax,
)
from prism.uncertainty.reference_set import (
    build_ood_reference_set,
    compute_class_centroids,
    compute_intra_class_radii,
)
from prism.uncertainty.relationships import (
    compute_representation_confidence_relationships,
)
from prism.uncertainty.reports import compile_uncertainty_analysis_report
from prism.uncertainty.runner import (
    UncertaintyAnalysisConfig,
    UncertaintyAnalysisRunner,
)
from prism.uncertainty.synthetic import (
    SyntheticOODSpec,
    generate_synthetic_ood_dataset,
)
from prism.uncertainty.temperature import (
    apply_temperature_scaling,
    evaluate_calibrated_predictions,
    fit_temperature_scaling,
)

__all__ = [
    "BinningStrategy",
    "CalibrationMode",
    "CalibrationReport",
    "CalibrationSample",
    "ClassCalibrationSummary",
    "ConfidenceMetric",
    "ConfidenceSubsetSummary",
    "CorruptionUncertaintyCurve",
    "OODBinaryEvaluationSummary",
    "OODCategory",
    "OODReferenceSet",
    "OODSample",
    "OODScoreMethod",
    "OODScoreResult",
    "PredictionFlipUncertainty",
    "PredictiveDistribution",
    "ReliabilityBin",
    "RepresentationConfidenceRelationship",
    "SyntheticOODSpec",
    "TemperatureScalingResult",
    "ThresholdPolicy",
    "UncertaintyAnalysisConfig",
    "UncertaintyAnalysisReport",
    "UncertaintyAnalysisRunner",
    "UncertaintyFailureRecord",
    "UncertaintyFailureType",
    "apply_temperature_scaling",
    "batch_predictive_distributions",
    "build_ood_reference_set",
    "compile_uncertainty_analysis_report",
    "compute_aupr",
    "compute_auroc",
    "compute_brier_score",
    "compute_calibration_report",
    "compute_class_centroid_ood_score",
    "compute_class_centroids",
    "compute_confidence_subset_summary",
    "compute_energy_ood_score",
    "compute_entropy_ood_score",
    "compute_expected_calibration_error",
    "compute_intra_class_radii",
    "compute_knn_ood_score",
    "compute_logit_margin",
    "compute_max_softmax_ood_score",
    "compute_maximum_calibration_error",
    "compute_negative_log_likelihood",
    "compute_normalized_entropy",
    "compute_predictive_distribution",
    "compute_predictive_entropy",
    "compute_probability_margin",
    "compute_reliability_bins",
    "compute_representation_confidence_relationships",
    "compute_stable_softmax",
    "detect_uncertainty_failures",
    "evaluate_calibrated_predictions",
    "evaluate_corruption_uncertainty",
    "evaluate_ood_binary_classification",
    "fit_temperature_scaling",
    "generate_synthetic_ood_dataset",
    "score_ood_sample",
    "select_ood_threshold",
]
