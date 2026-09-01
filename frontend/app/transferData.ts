import datasetJson from "./data/transferDataset.json";
import {
  LayerTransferProbePayload,
  SampleEfficiencySummaryPayload,
  SharedPCADriftPayload,
  TransferDemoPayload,
  TransferExperimentMetaPayload,
  TransferLearningReportPayload,
} from "./types";

const demoDataset: TransferDemoPayload = datasetJson as unknown as TransferDemoPayload;

export function getTransferDemoDataset(): TransferDemoPayload {
  return demoDataset;
}

export function getTransferMetadata(): TransferExperimentMetaPayload {
  return demoDataset.metadata;
}

export function getTransferReport(
  arch: string,
  strategy: string,
  budget: number = 1.0
): TransferLearningReportPayload | undefined {
  const key = `${arch.toLowerCase()}::${strategy.toLowerCase()}::${budget}`;
  return demoDataset.reports[key] || Object.values(demoDataset.reports).find(
    (r) =>
      r.architecture.toLowerCase() === arch.toLowerCase() &&
      r.strategy.toLowerCase() === strategy.toLowerCase()
  );
}

export function getTransferLayerProbes(
  arch: string
): LayerTransferProbePayload[] {
  return demoDataset.layer_probes[arch.toLowerCase()] || [];
}

export function getTransferDataEfficiency(
  arch: string
): SampleEfficiencySummaryPayload | undefined {
  return demoDataset.data_efficiency[arch.toLowerCase()];
}

export function getTransferSharedPCA(
  arch: string
): SharedPCADriftPayload | undefined {
  return demoDataset.shared_pca_drifts[arch.toLowerCase()];
}
