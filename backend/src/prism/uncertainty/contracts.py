"""Contracts and schemas for uncertainty, calibration, and OOD analysis."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from prism.core.errors import ValidationError
from prism.uncertainty.enums import (
    BinningStrategy,
    OODCategory,
    OODScoreMethod,
    ThresholdPolicy,
)


class PredictiveDistribution(BaseModel):
    """Predictive probability distribution and uncertainty descriptors."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_id: str = Field(description="Unique sample identifier")
    logits: list[float] = Field(description="Raw output logits [K]")
    probabilities: list[float] = Field(
        description="Softmax probability distribution over K classes"
    )
    predicted_class: int = Field(ge=0, description="Argmax predicted class index")
    true_class: int | None = Field(
        default=None, description="Ground truth class index if available"
    )
    max_probability: float = Field(
        ge=0.0, le=1.0, description="Maximum predicted class probability (confidence)"
    )
    entropy: float = Field(ge=0.0, description="Shannon entropy H(p) in nats")
    normalized_entropy: float = Field(
        ge=0.0, le=1.0, description="Normalized Shannon entropy H(p) / ln(K) in [0, 1]"
    )
    logit_margin: float = Field(
        ge=0.0, description="Margin between highest and second-highest logit"
    )
    probability_margin: float = Field(
        ge=0.0,
        le=1.0,
        description="Margin between highest and second-highest probability",
    )
    is_correct: bool | None = Field(
        default=None, description="Whether predicted class matches true class"
    )
    is_finite: bool = Field(
        default=True, description="Whether all logits and probabilities are finite"
    )

    @field_validator("probabilities")
    @classmethod
    def validate_probabilities(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValidationError("Probabilities list cannot be empty.")
        sum_p = sum(v)
        if abs(sum_p - 1.0) > 1e-4:
            raise ValidationError(f"Probabilities must sum to 1.0, got sum={sum_p}.")
        for val in v:
            if val < 0.0 or val > 1.0 + 1e-6:
                raise ValidationError(f"Probability out of [0, 1] range: {val}")
        return v

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class CalibrationSample(BaseModel):
    """Minimal sample contract for calibration and reliability curve analysis."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_id: str = Field(description="Sample identifier")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Model confidence (max softmax probability)"
    )
    is_correct: bool = Field(
        description="Whether the prediction is empirically correct"
    )
    predicted_class: int = Field(ge=0, description="Predicted class index")
    true_class: int = Field(ge=0, description="True ground-truth class index")
    probabilities: list[float] | None = Field(
        default=None, description="Optional full probability vector"
    )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class ReliabilityBin(BaseModel):
    """Summary of a single confidence bin for reliability diagrams."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    bin_index: int = Field(ge=0, description="0-indexed bin identifier")
    lower_bound: float = Field(
        ge=0.0, le=1.0, description="Lower confidence interval boundary"
    )
    upper_bound: float = Field(
        ge=0.0, le=1.0, description="Upper confidence interval boundary"
    )
    sample_count: int = Field(
        ge=0, description="Number of samples mapped into this bin"
    )
    mean_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Mean confidence of samples in this bin (0 if empty)",
    )
    empirical_accuracy: float = Field(
        ge=0.0,
        le=1.0,
        description="Empirical accuracy of samples in this bin (0 if empty)",
    )
    calibration_gap: float = Field(
        ge=0.0, le=1.0, description="Absolute calibration gap |accuracy - confidence|"
    )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class ConfidenceSubsetSummary(BaseModel):
    """Confidence and entropy statistics partitioned by correctness."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_count: int = Field(ge=0, description="Number of samples in subset")
    mean_max_probability: float = Field(
        ge=0.0, le=1.0, description="Mean max softmax probability"
    )
    median_max_probability: float = Field(
        ge=0.0, le=1.0, description="Median max softmax probability"
    )
    mean_entropy: float = Field(ge=0.0, description="Mean predictive entropy in nats")
    mean_normalized_entropy: float = Field(
        ge=0.0, le=1.0, description="Mean normalized predictive entropy"
    )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class ClassCalibrationSummary(BaseModel):
    """Class-conditional calibration breakdown."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    class_id: int = Field(ge=0, description="Class index")
    class_name: str = Field(description="Class name or label")
    sample_count: int = Field(ge=0, description="Number of samples in class")
    accuracy: float = Field(ge=0.0, le=1.0, description="Class-conditional accuracy")
    mean_confidence: float = Field(
        ge=0.0, le=1.0, description="Class-conditional mean confidence"
    )
    mean_entropy: float = Field(
        ge=0.0, description="Class-conditional mean predictive entropy"
    )
    ece: float | None = Field(
        default=None, description="Class-conditional ECE if sufficient samples"
    )
    warning: str | None = Field(
        default=None, description="Small sample size warning if count < 10"
    )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class CalibrationReport(BaseModel):
    """Comprehensive probability calibration report with reliability diagnostics."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_count: int = Field(ge=0, description="Total evaluated sample count")
    accuracy: float = Field(
        ge=0.0, le=1.0, description="Overall classification accuracy"
    )
    mean_confidence: float = Field(
        ge=0.0, le=1.0, description="Overall mean predictive confidence"
    )
    ece: float = Field(
        ge=0.0, le=1.0, description="Expected Calibration Error across reliability bins"
    )
    mce: float | None = Field(
        default=None, description="Maximum Calibration Error across non-empty bins"
    )
    brier_score: float = Field(
        ge=0.0, description="Multiclass Brier score: (1/N) sum ||p_n - y_n||^2"
    )
    nll: float = Field(
        ge=0.0, description="Mean negative log-likelihood (evaluation cross-entropy)"
    )
    mean_predictive_entropy: float = Field(
        ge=0.0, description="Mean predictive entropy across all samples (nats)"
    )
    mean_normalized_entropy: float = Field(
        ge=0.0, le=1.0, description="Mean normalized entropy across all samples"
    )
    binning_strategy: BinningStrategy = Field(
        default=BinningStrategy.EQUAL_WIDTH, description="Binning partition strategy"
    )
    bin_count: int = Field(ge=1, description="Number of confidence bins")
    reliability_bins: list[ReliabilityBin] = Field(
        description="Reliability diagram bin summaries"
    )
    error_subset_summary: ConfidenceSubsetSummary | None = Field(
        default=None, description="Confidence statistics on incorrect predictions"
    )
    correct_subset_summary: ConfidenceSubsetSummary | None = Field(
        default=None, description="Confidence statistics on correct predictions"
    )
    class_conditional_summaries: list[ClassCalibrationSummary] = Field(
        default_factory=list, description="Class-conditional calibration metrics"
    )
    warnings: list[str] = Field(default_factory=list, description="Diagnostic warnings")

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class TemperatureScalingResult(BaseModel):
    """Result of scalar temperature calibration on validation data."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    fitted_temperature: float = Field(
        gt=0.0, description="Optimal scalar temperature parameter T* > 0"
    )
    validation_nll_before: float = Field(
        ge=0.0, description="Validation NLL with T=1.0 (uncalibrated)"
    )
    validation_nll_after: float = Field(
        ge=0.0, description="Validation NLL with fitted T*"
    )
    ece_before: float = Field(
        ge=0.0, le=1.0, description="Validation ECE before scaling"
    )
    ece_after: float = Field(ge=0.0, le=1.0, description="Validation ECE after scaling")
    search_range: list[float] = Field(
        description="Bounds [min_T, max_T] used for grid search"
    )
    fitting_method: str = Field(
        default="deterministic_1d_grid_search", description="Optimization method"
    )
    iterations: int = Field(ge=1, description="Total candidate temperatures evaluated")
    warnings: list[str] = Field(
        default_factory=list, description="Optimization warnings"
    )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class OODSample(BaseModel):
    """Sample representation for out-of-distribution evaluation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_id: str = Field(description="Unique sample identifier")
    image: list[list[list[float]]] = Field(description="3D image tensor [C, H, W]")
    source_dataset_identity: str = Field(description="Originating dataset identity")
    category: OODCategory = Field(
        description="Taxonomy category (IN_DISTRIBUTION, OUT_OF_DISTRIBUTION, etc.)"
    )
    semantic_class: str | None = Field(
        default=None, description="Semantic class label if known"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary provenance metadata"
    )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class OODReferenceSet(BaseModel):
    """In-distribution reference representation cache for geometry-based OOD scoring."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_experiment: str = Field(description="Originating experiment ID")
    representation_layer: str = Field(
        description="Layer from which features were extracted"
    )
    sample_ids: list[str] = Field(description="Sample identifiers of reference set")
    labels: list[int] = Field(description="Integer class labels of reference samples")
    class_centroids: dict[str, list[float]] = Field(
        description="Class mean centroid vectors {class_id: [D]}"
    )
    intra_class_radii: dict[str, float] = Field(
        description="90th-percentile intra-class dispersion radius per class"
    )
    normalization_policy: str = Field(description="Vector normalization policy")
    distance_metric: str = Field(description="Primary distance metric")
    fingerprint: str = Field(description="SHA-256 fingerprint of the reference set")

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class OODScoreResult(BaseModel):
    """Individual sample OOD novelty scoring result."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_id: str = Field(description="Sample identifier")
    category: OODCategory = Field(description="True category of the sample")
    score_method: OODScoreMethod = Field(description="OOD score method")
    raw_score: float = Field(description="Raw metric score")
    normalized_ood_score: float = Field(
        description="Polarity-normalized score (higher = more OOD-like)"
    )
    score_direction: str = Field(
        default="higher_is_more_ood", description="Polarity convention"
    )
    predicted_class: int = Field(ge=0, description="Predicted class index")
    confidence: float = Field(ge=0.0, le=1.0, description="Max softmax probability")
    entropy: float = Field(ge=0.0, description="Predictive entropy")
    nearest_centroid_class: str | None = Field(
        default=None, description="Closest class centroid ID"
    )
    centroid_distance: float | None = Field(
        default=None, description="Distance to closest class centroid"
    )
    knn_distance: float | None = Field(
        default=None, description="Distance to k nearest in-distribution neighbors"
    )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class OODBinaryEvaluationSummary(BaseModel):
    """Binary ID vs OOD discrimination performance summary."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    score_method: OODScoreMethod = Field(description="Evaluated scoring method")
    auroc: float = Field(
        ge=0.0, le=1.0, description="Area Under the ROC curve (OOD as positive class)"
    )
    aupr: float | None = Field(
        default=None, description="Area Under the Precision-Recall curve"
    )
    threshold: float = Field(description="Selected decision threshold")
    threshold_policy: ThresholdPolicy = Field(description="Threshold selection policy")
    tpr_at_threshold: float = Field(
        ge=0.0,
        le=1.0,
        description="True Positive Rate (OOD recall) at decision threshold",
    )
    fpr_at_threshold: float = Field(
        ge=0.0, le=1.0, description="False Positive Rate (ID misclassified as OOD)"
    )
    detection_accuracy_at_threshold: float = Field(
        ge=0.0,
        le=1.0,
        description="Binary classification accuracy at decision threshold",
    )
    id_sample_count: int = Field(
        ge=1, description="Number of In-Distribution evaluation samples"
    )
    ood_sample_count: int = Field(
        ge=1, description="Number of Out-of-Distribution evaluation samples"
    )
    mean_id_score: float = Field(description="Mean normalized score on ID samples")
    mean_ood_score: float = Field(description="Mean normalized score on OOD samples")
    score_separation_gap: float = Field(
        description="Separation gap: (mean_ood_score - mean_id_score)"
    )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class CorruptionUncertaintyCurve(BaseModel):
    """Trajectory of confidence, entropy, and calibration across corruption severity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    corruption_type: str = Field(description="Corruption family identifier")
    severities: list[int] = Field(description="Evaluated severities [1..5]")
    accuracies: list[float] = Field(description="Accuracy per severity")
    mean_confidences: list[float] = Field(description="Mean confidence per severity")
    mean_entropies: list[float] = Field(
        description="Mean predictive entropy per severity"
    )
    eces: list[float] = Field(description="Expected calibration error per severity")
    mean_representation_drifts: list[float] = Field(
        description="Mean representation Euclidean drift per severity"
    )
    mean_ood_scores: list[float] = Field(
        description="Mean centroid OOD score per severity"
    )
    confidence_slope: float = Field(
        description="Linear regression slope of confidence vs severity"
    )
    entropy_slope: float = Field(
        description="Linear regression slope of entropy vs severity"
    )
    is_monotonic_entropy: bool = Field(
        description="Whether entropy strictly increases with severity"
    )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class PredictionFlipUncertainty(BaseModel):
    """Uncertainty and representation dynamics during prediction flips."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_id: str = Field(description="Sample identifier")
    corruption_type: str = Field(description="Corruption type")
    severity: int = Field(ge=1, le=5, description="Corruption severity")
    clean_prediction: int = Field(description="Clean input predicted class")
    corrupted_prediction: int = Field(description="Corrupted input predicted class")
    clean_confidence: float = Field(
        ge=0.0, le=1.0, description="Clean prediction confidence"
    )
    corrupted_confidence: float = Field(
        ge=0.0, le=1.0, description="Corrupted prediction confidence"
    )
    clean_entropy: float = Field(ge=0.0, description="Clean predictive entropy")
    corrupted_entropy: float = Field(ge=0.0, description="Corrupted predictive entropy")
    representation_drift: float = Field(
        ge=0.0, description="Euclidean representation displacement"
    )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class RepresentationConfidenceRelationship(BaseModel):
    """Empirical relationship between geometry and predictive confidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    centroid_distance_pearson_correlation: float | None = Field(
        default=None, description="Pearson r between centroid distance and confidence"
    )
    knn_distance_pearson_correlation: float | None = Field(
        default=None, description="Pearson r between kNN distance and confidence"
    )
    correct_mean_centroid_distance: float = Field(
        ge=0.0, description="Mean centroid distance of correct predictions"
    )
    incorrect_mean_centroid_distance: float = Field(
        ge=0.0, description="Mean centroid distance of incorrect predictions"
    )
    correct_mean_knn_distance: float = Field(
        ge=0.0, description="Mean kNN distance of correct predictions"
    )
    incorrect_mean_knn_distance: float = Field(
        ge=0.0, description="Mean kNN distance of incorrect predictions"
    )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class UncertaintyAnalysisReport(BaseModel):
    """Unified comprehensive uncertainty, calibration, and OOD analysis report."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str = Field(description="Evaluated model ID")
    architecture: str = Field(description="Visual backbone architecture family")
    source_objective: str = Field(description="Pretraining objective")
    dataset_fingerprint: str = Field(description="In-distribution dataset fingerprint")
    split: str = Field(description="Evaluation dataset split")
    representation_layer: str = Field(
        description="Extracted feature representation layer"
    )
    seed: int = Field(description="Random seed")
    calibration_report: CalibrationReport = Field(
        description="Uncalibrated model calibration report"
    )
    temperature_scaling: TemperatureScalingResult | None = Field(
        default=None, description="Validation temperature fitting result"
    )
    calibrated_report: CalibrationReport | None = Field(
        default=None, description="Calibrated test evaluation report"
    )
    ood_evaluations: dict[str, OODBinaryEvaluationSummary] = Field(
        default_factory=dict, description="OOD performance summaries keyed by method"
    )
    representation_relationship: RepresentationConfidenceRelationship | None = Field(
        default=None, description="Representation distance vs confidence relationship"
    )
    corruption_curves: list[CorruptionUncertaintyCurve] = Field(
        default_factory=list, description="Corruption uncertainty degradation curves"
    )
    prediction_flips: list[PredictionFlipUncertainty] = Field(
        default_factory=list, description="Detailed prediction flip records"
    )
    failure_counts: dict[str, int] = Field(
        default_factory=dict, description="Count of flagged failures by category"
    )
    warnings: list[str] = Field(
        default_factory=list, description="Scientific disclaimers and warnings"
    )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
