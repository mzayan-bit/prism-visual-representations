import dataset from "./data/robustnessDataset.json";
import {
  CorruptionEvaluationSummary,
  CorruptionSeverityCurve,
  CorruptionType,
  CrossArchitectureRobustnessReport,
  RobustnessExperimentMeta,
  RobustnessExperimentReport,
  SampleRepresentationDrift,
} from "./types";

export function getRobustnessMetadata(): RobustnessExperimentMeta {
  return dataset.metadata as RobustnessExperimentMeta;
}

export function getCrossArchitectureRobustness(): CrossArchitectureRobustnessReport {
  return dataset.comparison as unknown as CrossArchitectureRobustnessReport;
}

export function getRobustnessReport(arch: string): RobustnessExperimentReport | null {
  const reports = dataset.reports as unknown as Record<
    string,
    RobustnessExperimentReport
  >;
  return reports[arch] || null;
}

export function getCorruptionEvaluation(
  arch: string,
  corruption: CorruptionType | string,
  severity: number
): CorruptionEvaluationSummary | null {
  const report = getRobustnessReport(arch);
  if (!report || !report.evaluations) return null;
  const key = `${corruption}::sev${severity}`;
  return report.evaluations[key] || null;
}

export function getCorruptionSeverityCurve(
  arch: string,
  corruption: CorruptionType | string
): CorruptionSeverityCurve | null {
  const report = getRobustnessReport(arch);
  if (!report || !report.severity_curves) return null;
  return report.severity_curves[corruption] || null;
}

export function getSampleDrifts(
  arch: string,
  corruption: CorruptionType | string,
  severity: number
): SampleRepresentationDrift[] {
  const report = getRobustnessReport(arch);
  if (!report || !report.sample_drifts) return [];
  const key = `${corruption}::sev${severity}`;
  return report.sample_drifts[key] || [];
}
