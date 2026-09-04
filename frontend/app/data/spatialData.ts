import {
  PretrainingObjectiveType,
  SpatialDataEfficiencyRecord,
  SpatialDatasetPayload,
  SpatialDetectionSamplePayload,
  SpatialLayerTransferabilityRecord,
  SpatialSegmentationSamplePayload,
  SpatialTaskType,
  SpatialTransferReportPayload,
} from "../types";
import rawSpatialDataset from "./spatialDataset.json";

const dataset = rawSpatialDataset as unknown as SpatialDatasetPayload;

export function getSpatialDataset(): SpatialDatasetPayload {
  return dataset;
}

export function getSpatialReports(
  architecture: string,
  taskType: SpatialTaskType
): SpatialTransferReportPayload[] {
  return dataset.reports.filter(
    (r) =>
      r.specification.task_type === taskType &&
      r.specification.source_experiment_id.includes(architecture.toLowerCase())
  );
}

export function getSpatialObjectiveComparison(
  architecture: string,
  taskType: SpatialTaskType
): Record<PretrainingObjectiveType, SpatialTransferReportPayload | null> {
  const reports = getSpatialReports(architecture, taskType);
  const result: Record<
    PretrainingObjectiveType,
    SpatialTransferReportPayload | null
  > = {
    supervised: null,
    simclr: null,
    reconstruction: null,
    scratch: null,
  };

  reports.forEach((r) => {
    const obj = r.specification.source_objective;
    if (obj in result) {
      result[obj] = r;
    }
  });

  return result;
}

export function getSpatialLayerTransferability(
  architecture: string
): SpatialLayerTransferabilityRecord[] {
  return dataset.layer_transferability[architecture.toLowerCase()] || [];
}

export function getSpatialDataEfficiency(
  architecture: string
): SpatialDataEfficiencyRecord[] {
  return dataset.data_efficiency[architecture.toLowerCase()] || [];
}

export function getSpatialDetectionSamples(): SpatialDetectionSamplePayload[] {
  return dataset.detection_samples || [];
}

export function getSpatialSegmentationSamples(): SpatialSegmentationSamplePayload[] {
  return dataset.segmentation_samples || [];
}
