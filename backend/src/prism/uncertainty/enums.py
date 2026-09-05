"""Enumerations for uncertainty, calibration, and OOD analysis."""

from __future__ import annotations

from enum import Enum


class ConfidenceMetric(str, Enum):
    """Predictive confidence and uncertainty metrics."""

    MAX_PROBABILITY = "max_probability"
    PREDICTIVE_ENTROPY = "predictive_entropy"
    LOGIT_MARGIN = "logit_margin"
    PROBABILITY_MARGIN = "probability_margin"


class BinningStrategy(str, Enum):
    """Strategy for partitioning confidence scores into reliability diagram bins."""

    EQUAL_WIDTH = "equal_width"
    EQUAL_FREQUENCY = "equal_frequency"


class CalibrationMode(str, Enum):
    """Model evaluation calibration state."""

    UNCALIBRATED = "uncalibrated"
    TEMPERATURE_SCALED = "temperature_scaled"


class OODCategory(str, Enum):
    """Taxonomy category for evaluation samples."""

    IN_DISTRIBUTION = "in_distribution"
    OUT_OF_DISTRIBUTION = "out_of_distribution"
    CORRUPTED_IN_DISTRIBUTION = "corrupted_in_distribution"
    NEAR_OOD = "near_ood"


class OODScoreMethod(str, Enum):
    """Methods for computing sample-level out-of-distribution novelty scores."""

    MAX_SOFTMAX_PROBABILITY = "max_softmax_probability"
    PREDICTIVE_ENTROPY = "predictive_entropy"
    NEAREST_CLASS_CENTROID_DISTANCE = "nearest_class_centroid_distance"
    KNN_REPRESENTATION_DISTANCE = "knn_representation_distance"
    ENERGY_SCORE = "energy_score"


class ThresholdPolicy(str, Enum):
    """Policy for selecting binary OOD detection decision thresholds."""

    FIXED = "fixed"
    VALIDATION_QUANTILE = "validation_quantile"
    TARGET_ID_TPR = "target_id_tpr"


class UncertaintyFailureType(str, Enum):
    """Diagnostic categorization of uncertainty, calibration, and OOD failures."""

    HIGH_CONFIDENCE_ERROR = "high_confidence_error"
    HIGH_CONFIDENCE_OOD = "high_confidence_ood"
    LOW_CONFIDENCE_CORRECT = "low_confidence_correct"
    CALIBRATION_OUTLIER = "calibration_outlier"
    OOD_NEAR_KNOWN_STRUCTURE = "ood_near_known_structure"
    ID_REPRESENTATION_OUTLIER = "id_representation_outlier"
    CORRUPTION_OVERCONFIDENCE = "corruption_overconfidence"
    NON_MONOTONIC_UNCERTAINTY = "non_monotonic_uncertainty"
