import dataset from "./data/observatoryDataset.json";
import {
  CrossArchitectureGeometryReport,
  LayerGeometryProfile,
  RepresentationGeometryReport,
} from "./types";

export interface ObservatoryMetadata {
  experiment_id: string;
  name: string;
  architectures: string[];
  layers: Record<string, string[]>;
  data_budgets: number[];
  num_classes: number;
  class_names: string[];
}

export function getObservatoryMetadata(): ObservatoryMetadata {
  return dataset.metadata as ObservatoryMetadata;
}

export function getCrossArchitectureComparison(): CrossArchitectureGeometryReport {
  return dataset.comparison as unknown as CrossArchitectureGeometryReport;
}

export function getLayerGeometryProfile(
  arch: string
): LayerGeometryProfile | null {
  const profiles = dataset.layer_profiles as unknown as Record<
    string,
    LayerGeometryProfile
  >;
  return profiles[arch] || null;
}

export function getRepresentationGeometryReport(
  arch: string,
  layer: string
): RepresentationGeometryReport | null {
  const profile = getLayerGeometryProfile(arch);
  if (!profile || !profile.detailed_reports) {
    return null;
  }
  return profile.detailed_reports[layer] || null;
}
