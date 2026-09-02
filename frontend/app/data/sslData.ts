import sslDatasetRaw from "./sslDataset.json";
import {
  SSLDemoPayload,
  SelfSupervisedLearningReportPayload,
  SupervisedVsSSLComparisonPayload,
  SSLLabelEfficiencyPointPayload,
  SSLGeometryPointPayload,
  SSLLayerProbePointPayload,
} from "../types";

export const sslDemoData: SSLDemoPayload = sslDatasetRaw as unknown as SSLDemoPayload;

export function getSSLReport(architecture: string): SelfSupervisedLearningReportPayload | null {
  return sslDemoData.reports[architecture.toLowerCase()] || null;
}

export function getSSLComparison(architecture: string): SupervisedVsSSLComparisonPayload | null {
  return sslDemoData.comparisons[architecture.toLowerCase()] || null;
}

export function getSSLLabelEfficiency(architecture: string): SSLLabelEfficiencyPointPayload[] {
  return sslDemoData.label_efficiency[architecture.toLowerCase()] || [];
}

export function getSSLGeometryPoints(architecture: string): SSLGeometryPointPayload[] {
  return sslDemoData.geometry_points[architecture.toLowerCase()] || [];
}

export function getSSLLayerProbes(architecture: string): SSLLayerProbePointPayload[] {
  return sslDemoData.layer_probes[architecture.toLowerCase()] || [];
}
