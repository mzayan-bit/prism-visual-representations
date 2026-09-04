import multimodalDataset from "./multimodalDataset.json";
import {
  MultimodalDatasetPayload,
  MultimodalSamplePayload,
  CrossModalRetrievalSummaryPayload,
  ZeroShotClassificationSummaryPayload,
  PromptSensitivityPayload,
  CentroidAlignmentPayload,
  MultimodalCollapseSummaryPayload,
  MultimodalObjectiveComparisonPayload,
  MultimodalArchitectureComparisonPayload,
  MultimodalCandidateFailurePayload,
} from "../types";

const dataset = multimodalDataset as unknown as MultimodalDatasetPayload;

export function getMultimodalDataset(): MultimodalDatasetPayload {
  return dataset;
}

export function getMultimodalMetadata() {
  return dataset.metadata;
}

export function getMultimodalSamples(): MultimodalSamplePayload[] {
  return dataset.samples || [];
}

export function getMultimodalSampleById(id: string): MultimodalSamplePayload | undefined {
  return dataset.samples?.find((s) => s.sample_id === id);
}

export function getCrossModalRetrievalSummary(): CrossModalRetrievalSummaryPayload {
  return dataset.retrieval_summary;
}

export function getZeroShotClassificationSummary(): ZeroShotClassificationSummaryPayload {
  return dataset.zero_shot_summary;
}

export function getPromptSensitivity(): PromptSensitivityPayload {
  return dataset.prompt_sensitivity;
}

export function getSharedGeometry() {
  return dataset.shared_geometry;
}

export function getMultimodalCentroidAlignments(): CentroidAlignmentPayload[] {
  return dataset.shared_geometry?.centroid_alignments || [];
}

export function getMultimodalCollapseSummary(): MultimodalCollapseSummaryPayload {
  return dataset.collapse_summary;
}

export function getMultimodalObjectiveComparisons(): MultimodalObjectiveComparisonPayload[] {
  return dataset.objective_comparisons || [];
}

export function getMultimodalArchitectureComparisons(): MultimodalArchitectureComparisonPayload[] {
  return dataset.architecture_comparisons || [];
}

export function getMultimodalRobustnessBenchmarks() {
  return dataset.robustness_benchmarks;
}

export function getMultimodalCandidateFailures(): MultimodalCandidateFailurePayload[] {
  return dataset.candidate_failures || [];
}
