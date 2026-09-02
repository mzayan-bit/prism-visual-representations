import reconstructionDatasetRaw from "./reconstructionDataset.json";
import {
  ReconstructionDatasetPayload,
  VisualTripletSamplePayload,
  MaskingRatioPointPayload,
  ThreeWayComparisonEntryPayload,
  ReconstructionLayerProbeEntryPayload,
  ReconstructionFailureCasePayload,
  ReconstructionDynamicsPayload,
  ReconstructionMetadataPayload,
} from "../types";

export const reconstructionDemoData: ReconstructionDatasetPayload =
  reconstructionDatasetRaw as unknown as ReconstructionDatasetPayload;

export function getReconstructionMetadata(): ReconstructionMetadataPayload {
  return reconstructionDemoData.metadata;
}

export function getVisualTriplets(
  method: string = "masked_patch_reconstruction"
): VisualTripletSamplePayload[] {
  if (method === "denoising_autoencoder") {
    return reconstructionDemoData.triplets_denoising;
  }
  return reconstructionDemoData.triplets_masked_patch;
}

export function getReconstructionDynamics(): ReconstructionDynamicsPayload {
  return reconstructionDemoData.dynamics;
}

export function getMaskingRatioStudy(): MaskingRatioPointPayload[] {
  return reconstructionDemoData.masking_ratio_study;
}

export function getThreeWayComparison(): ThreeWayComparisonEntryPayload[] {
  return reconstructionDemoData.three_way_comparison;
}

export function getReconstructionLayerProbes(): ReconstructionLayerProbeEntryPayload[] {
  return reconstructionDemoData.layer_probes;
}

export function getReconstructionFailureCases(): ReconstructionFailureCasePayload[] {
  return reconstructionDemoData.failure_cases;
}
