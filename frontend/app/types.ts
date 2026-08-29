export type DistanceMetric =
  | "euclidean"
  | "squared_euclidean"
  | "cosine_similarity"
  | "cosine_distance";

export type SpatialTransformation = "none" | "global_average_pool" | "flatten";
export type NormalizationPolicy = "none" | "l2_normalize" | "standardize";

export interface VectorNormSummary {
  mean_norm: number;
  std_norm: number;
  min_norm: number;
  max_norm: number;
}

export interface ClassCentroidSummary {
  class_id: string;
  class_name: string;
  sample_count: number;
  centroid: number[];
  centroid_norm: number;
  intra_class_mean_distance: number;
  intra_class_std_distance: number;
  intra_class_max_distance: number;
  intra_class_radius_90: number;
  nearest_competing_class: string | null;
  distance_to_nearest_competing_centroid: number | null;
}

export interface CentroidGeometryReport {
  class_order: string[];
  class_centroids: Record<string, ClassCentroidSummary>;
  centroid_distance_matrix: number[][];
  mean_intra_class_distance: number;
  mean_inter_class_centroid_distance: number;
  min_inter_class_centroid_distance: number;
  separation_to_compactness_ratio: number;
}

export interface NearestNeighborEntry {
  rank: number;
  neighbor_sample_id: string;
  neighbor_label: string | number;
  distance: number;
  same_class: boolean;
}

export interface SampleNeighborhood {
  query_sample_id: string;
  query_label: string | number;
  neighbors: NearestNeighborEntry[];
  same_class_fraction: number;
  distance_to_own_centroid: number | null;
  nearest_competing_centroid_distance: number | null;
}

export interface CandidateFailureCase {
  sample_id: string;
  label: string | number;
  failure_kind: string;
  description: string;
  metric_value: number;
}

export interface NeighborhoodGeometrySummary {
  k: number;
  metric: DistanceMetric;
  mean_label_consistency: number;
  median_label_consistency: number;
  per_class_label_consistency: Record<string, number>;
  candidate_failures: CandidateFailureCase[];
  sample_neighborhoods: Record<string, SampleNeighborhood>;
}

export interface ProjectionResult {
  method: string;
  original_dim: number;
  projected_dim: number;
  num_samples: number;
  sample_ids: string[];
  labels: (string | number)[];
  coordinates: number[][];
  explained_variance: number[];
  explained_variance_ratio: number[];
  cumulative_explained_variance: number[];
  mean_vector?: number[];
}

export interface RepresentationGeometryReport {
  experiment_id: string;
  model_id: string;
  layer_name: string;
  num_samples: number;
  feature_dim: number;
  num_classes: number;
  class_names: string[];
  source_split: string;
  spatial_transformation: SpatialTransformation;
  normalization_policy: NormalizationPolicy;
  distance_metric: DistanceMetric;
  vector_norms: VectorNormSummary;
  centroid_geometry: CentroidGeometryReport;
  neighborhood_geometry: NeighborhoodGeometrySummary;
  pca_projection: ProjectionResult;
  candidate_failures: CandidateFailureCase[];
  warnings: string[];
  metadata?: Record<string, unknown>;
}

export interface LayerGeometryPoint {
  layer_name: string;
  depth_index: number;
  feature_dim: number;
  original_shape: number[] | null;
  mean_intra_class_distance: number;
  mean_inter_class_centroid_distance: number;
  separation_to_compactness_ratio: number;
  mean_label_consistency: number;
  pca_first_two_variance_ratio: number;
}

export interface LayerGeometryProfile {
  experiment_id: string;
  model_id: string;
  architecture: string;
  distance_metric: DistanceMetric;
  normalization_policy: NormalizationPolicy;
  layer_points: LayerGeometryPoint[];
  compactness_trend: number[];
  separation_trend: number[];
  consistency_trend: number[];
  ratio_trend: number[];
  detailed_reports?: Record<string, RepresentationGeometryReport>;
}

export interface ArchitectureGeometrySummary {
  architecture: string;
  model_family: string;
  model_id: string;
  layer_name: string;
  feature_dim: number;
  mean_vector_norm: number;
  intra_class_compactness: number;
  inter_class_separation: number;
  separation_to_compactness_ratio: number;
  neighbor_label_consistency: number;
  pca_first_two_variance_ratio: number;
  total_parameters: number | null;
}

export interface CrossArchitectureGeometryReport {
  comparison_id: string;
  name: string;
  dataset_fingerprint: string;
  data_budget: number;
  distance_metric: DistanceMetric;
  normalization_policy: NormalizationPolicy;
  architectures: Record<string, ArchitectureGeometrySummary>;
  coordinate_space_note: string;
}
