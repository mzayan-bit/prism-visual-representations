"""Comprehensive uncertainty and OOD analysis report compilation."""

from __future__ import annotations

from collections.abc import Sequence

from prism.uncertainty.contracts import (
    CalibrationReport,
    CorruptionUncertaintyCurve,
    OODBinaryEvaluationSummary,
    PredictionFlipUncertainty,
    RepresentationConfidenceRelationship,
    TemperatureScalingResult,
    UncertaintyAnalysisReport,
)
from prism.uncertainty.failures import UncertaintyFailureRecord


def compile_uncertainty_analysis_report(
    model_name: str,
    architecture: str,
    source_objective: str,
    dataset_name: str,
    split: str,
    representation_layer: str,
    seed: int,
    uncalibrated_report: CalibrationReport,
    ood_evaluations: dict[str, OODBinaryEvaluationSummary],
    representation_relationships: (RepresentationConfidenceRelationship | None) = None,
    temperature_scaling: TemperatureScalingResult | None = None,
    calibrated_report: CalibrationReport | None = None,
    corruption_curve: CorruptionUncertaintyCurve | None = None,
    prediction_flips: Sequence[PredictionFlipUncertainty] | None = None,
    failure_records: Sequence[UncertaintyFailureRecord] | None = None,
    warnings: list[str] | None = None,
) -> UncertaintyAnalysisReport:
    """Compile a complete UncertaintyAnalysisReport data structure.

    Parameters
    ----------
    model_name : str
        Model or checkpoint name.
    architecture : str
        Backbone architecture ('CNN', 'ResNet', 'ViT').
    source_objective : str
        Learning objective (e.g. 'supervised', 'simclr', 'reconstruction').
    dataset_name : str
        Name of evaluation dataset.
    split : str
        Evaluation split ('test', 'val').
    representation_layer : str
        Layer used for representation geometry.
    seed : int
        RNG seed.
    uncalibrated_report : CalibrationReport
        Standard uncalibrated probability calibration report.
    ood_evaluations : dict[str, OODBinaryEvaluationSummary]
        OOD detection summaries across scoring methods.
    representation_relationships : RepresentationConfidenceRelationship | None
        Geometry vs confidence metrics.
    temperature_scaling : TemperatureScalingResult | None
        Validation temperature scaling results.
    calibrated_report : CalibrationReport | None
        Test evaluation after temperature scaling.
    corruption_curve : CorruptionUncertaintyCurve | None
        Severity uncertainty trajectory.
    prediction_flips : Sequence[PredictionFlipUncertainty] | None
        Prediction flip events across corruptions.
    failure_records : Sequence[UncertaintyFailureRecord] | None
        Identified failure taxonomy cases.
    warnings : list[str] | None
        Additional warnings.

    Returns
    -------
    UncertaintyAnalysisReport
        Structured report contract.
    """
    all_warnings = list(uncalibrated_report.warnings)
    if temperature_scaling is not None:
        all_warnings.extend(temperature_scaling.warnings)
    if warnings:
        all_warnings.extend(warnings)

    failure_counts: dict[str, int] = {}
    if failure_records:
        for f in failure_records:
            k = f.failure_type.value
            failure_counts[k] = failure_counts.get(k, 0) + 1

    curves = [corruption_curve] if corruption_curve is not None else []
    flips = list(prediction_flips) if prediction_flips is not None else []

    return UncertaintyAnalysisReport(
        model_id=model_name,
        architecture=architecture,
        source_objective=source_objective,
        dataset_fingerprint=dataset_name,
        split=split,
        representation_layer=representation_layer,
        seed=seed,
        calibration_report=uncalibrated_report,
        temperature_scaling=temperature_scaling,
        calibrated_report=calibrated_report,
        ood_evaluations=ood_evaluations,
        representation_relationship=representation_relationships,
        corruption_curves=curves,
        prediction_flips=flips,
        failure_counts=failure_counts,
        warnings=sorted(set(all_warnings)),
    )
