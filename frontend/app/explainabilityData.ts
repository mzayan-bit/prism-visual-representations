import datasetJson from "./data/explainabilityDataset.json";
import {
  ExplainabilityDemoPayload,
  ExplainabilityExperimentMeta,
  ExplainabilitySamplePayload,
} from "./types";

const demoDataset: ExplainabilityDemoPayload = datasetJson as unknown as ExplainabilityDemoPayload;

export function getExplainabilityDemoDataset(): ExplainabilityDemoPayload {
  return demoDataset;
}

export function getExplainabilityMetadata(): ExplainabilityExperimentMeta {
  return demoDataset.metadata;
}

export function getAllExplainabilitySamples(): ExplainabilitySamplePayload[] {
  return demoDataset.samples;
}

export function getExplainabilitySample(
  sampleId: string
): ExplainabilitySamplePayload | undefined {
  return demoDataset.samples.find((s) => s.sample_id === sampleId);
}
