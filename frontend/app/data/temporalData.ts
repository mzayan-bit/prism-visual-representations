import temporalDataset from "./temporalDataset.json";
import {
  TemporalDatasetPayload,
  TemporalObjectiveComparisonPayload,
  TemporalLayerProfileRecord,
  TemporalAggregatorComparisonPayload,
  TemporalRobustnessBenchmarkPayload,
  TemporalDataEfficiencyRecord,
  TemporalSequenceLengthRecord,
  TemporalCandidateFailurePayload,
  TemporalVideoSamplePayload,
} from "../types";

const dataset = temporalDataset as unknown as TemporalDatasetPayload;

export function getTemporalDataset(): TemporalDatasetPayload {
  return dataset;
}

export function getTemporalMetadata() {
  return dataset.metadata;
}

export function getTemporalSamples(): TemporalVideoSamplePayload[] {
  return dataset.samples || [];
}

export function getTemporalSampleById(id: string): TemporalVideoSamplePayload | undefined {
  return dataset.samples?.find((s) => s.video_id === id);
}

export function getTemporalObjectiveComparisons(): TemporalObjectiveComparisonPayload[] {
  return dataset.objective_comparisons || [];
}

export function getTemporalLayerProfiles(arch: string): TemporalLayerProfileRecord[] {
  return dataset.layer_profiles?.[arch] || [];
}

export function getTemporalAggregatorComparisons(): TemporalAggregatorComparisonPayload[] {
  return dataset.aggregator_comparisons || [];
}

export function getTemporalRobustnessBenchmarks(): TemporalRobustnessBenchmarkPayload[] {
  return dataset.robustness_benchmarks || [];
}

export function getTemporalDataEfficiencyCurves(): TemporalDataEfficiencyRecord[] {
  return dataset.data_efficiency_curves || [];
}

export function getTemporalSequenceLengthStudies(): TemporalSequenceLengthRecord[] {
  return dataset.sequence_length_studies || [];
}

export function getTemporalCandidateFailures(): TemporalCandidateFailurePayload[] {
  return dataset.candidate_failures || [];
}
