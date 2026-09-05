import datasetJson from "./data/uncertaintyDataset.json";
import {
  CalibrationReportPayload,
  CorruptionUncertaintyCurvePayload,
  OODBinaryEvaluationSummaryPayload,
  RepresentationConfidenceRelationshipPayload,
  TemperatureScalingResultPayload,
  UncertaintyAnalysisReportPayload,
  UncertaintyArchitectureComparisonPayload,
  UncertaintyDatasetPayload,
  UncertaintyObjectiveComparisonPayload,
  UncertaintySampleItemPayload,
} from "./types";

const demoDataset: UncertaintyDatasetPayload =
  datasetJson as unknown as UncertaintyDatasetPayload;

export function getUncertaintyDataset(): UncertaintyDatasetPayload {
  return demoDataset;
}

export function getUncertaintyMetadata() {
  return demoDataset.meta;
}

export function getUncertaintyReport(): UncertaintyAnalysisReportPayload {
  return demoDataset.report;
}

export function getCalibrationReport(): CalibrationReportPayload {
  return demoDataset.report.calibration_report;
}

export function getCalibratedReport(): CalibrationReportPayload | null {
  return demoDataset.report.calibrated_report;
}

export function getTemperatureScalingResult(): TemperatureScalingResultPayload | null {
  return demoDataset.report.temperature_scaling;
}

export function getOODBinaryEvaluations(): Record<
  string,
  OODBinaryEvaluationSummaryPayload
> {
  return demoDataset.report.ood_evaluations;
}

export function getCorruptionUncertaintyCurves(): CorruptionUncertaintyCurvePayload[] {
  return demoDataset.report.corruption_curves;
}

export function getRepresentationRelationship(): RepresentationConfidenceRelationshipPayload | null {
  return demoDataset.report.representation_relationship;
}

export function getUncertaintySamples(): UncertaintySampleItemPayload[] {
  return demoDataset.samples;
}

export function getUncertaintyObjectiveComparisons(): UncertaintyObjectiveComparisonPayload[] {
  return demoDataset.objective_comparisons;
}

export function getUncertaintyArchitectureComparisons(): UncertaintyArchitectureComparisonPayload[] {
  return demoDataset.architecture_comparisons;
}
