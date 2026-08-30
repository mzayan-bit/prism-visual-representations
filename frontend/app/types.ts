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

// ----------------------------------------------------------------------------
// Phase 15 — Robustness & Distribution Shift Laboratory Types
// ----------------------------------------------------------------------------

export type CorruptionType =
  | "gaussian_noise"
  | "blur"
  | "brightness"
  | "contrast"
  | "occlusion"
  | "resolution_degradation";

export interface SampleRepresentationDrift {
  sample_id: string;
  label: string | number;
  clean_prediction: number;
  corrupted_prediction: number;
  prediction_changed: boolean;
  clean_correct: boolean;
  corrupted_correct: boolean;
  clean_loss: number;
  corrupted_loss: number;
  euclidean_drift: number;
  cosine_similarity: number;
  cosine_distance: number;
  relative_norm_change: number;
}

export interface RepresentationDriftSummary {
  num_samples: number;
  mean_euclidean_drift: number;
  std_euclidean_drift: number;
  median_euclidean_drift: number;
  max_euclidean_drift: number;
  mean_cosine_similarity: number;
  mean_cosine_distance: number;
  mean_relative_norm_change: number;
  per_class_mean_drift: Record<string, number>;
  per_class_mean_cosine_similarity: Record<string, number>;
  drift_by_prediction_outcome: Record<string, number>;
  top_drift_sample_ids: string[];
}

export interface SharedPCAProjectionResult {
  original_dim: number;
  projected_dim: number;
  num_samples: number;
  sample_ids: string[];
  labels: (string | number)[];
  clean_coordinates: number[][];
  corrupted_coordinates: number[][];
  displacement_vectors: number[][];
  displacement_magnitudes: number[];
  explained_variance_ratio: number[];
  cumulative_explained_variance: number[];
  basis_note: string;
}

export interface ClassCentroidDriftSummary {
  class_label: string;
  clean_centroid_norm: number;
  corrupted_centroid_norm: number;
  centroid_displacement: number;
  cosine_similarity: number;
  clean_intra_compactness: number;
  corrupted_intra_compactness: number;
  compactness_delta: number;
  clean_competing_separation: number;
  corrupted_competing_separation: number;
  competing_separation_delta: number;
}

export interface NeighborhoodDriftSummary {
  k: number;
  mean_neighbor_overlap_ratio: number;
  clean_mean_label_consistency: number;
  corrupted_mean_label_consistency: number;
  label_consistency_delta: number;
  nearest_neighbor_label_flip_fraction: number;
}

export interface GeometryDriftReport {
  clean_centroid_report: CentroidGeometryReport;
  corrupted_centroid_report: CentroidGeometryReport;
  mean_centroid_displacement: number;
  class_centroid_drifts: Record<string, ClassCentroidDriftSummary>;
  neighborhood_drift: NeighborhoodDriftSummary;
  shared_pca: SharedPCAProjectionResult;
  separation_to_compactness_ratio_delta: number;
}

export interface LayerAttentionDrift {
  layer_name: string;
  clean_entropy: number;
  corrupted_entropy: number;
  entropy_delta: number;
  clean_diagonal_mass: number;
  corrupted_diagonal_mass: number;
  diagonal_mass_delta: number;
}

export interface AttentionDriftSummary {
  model_id: string;
  num_layers: number;
  clean_overall_mean_entropy: number;
  corrupted_overall_mean_entropy: number;
  overall_entropy_delta: number;
  clean_overall_diagonal_mass: number;
  corrupted_overall_diagonal_mass: number;
  overall_diagonal_mass_delta: number;
  layer_drifts: LayerAttentionDrift[];
}

export interface RobustnessFailureRecord {
  sample_id: string;
  category: string;
  description: string;
  severity: number;
  metrics: Record<string, number>;
}

export interface CorruptionEvaluationSummary {
  corruption_type: CorruptionType;
  severity: number;
  num_samples: number;
  clean_accuracy: number;
  corrupted_accuracy: number;
  absolute_accuracy_drop: number;
  relative_accuracy_drop: number;
  clean_loss: number;
  corrupted_loss: number;
  loss_increase: number;
  predictions_changed_count: number;
  prediction_consistency_fraction: number;
  representation_drift: RepresentationDriftSummary;
  geometry_drift: GeometryDriftReport;
  attention_drift: AttentionDriftSummary | null;
  failure_counts_by_category: Record<string, number>;
}

export interface CorruptionSeverityCurve {
  corruption_type: CorruptionType;
  severities: number[];
  clean_accuracy: number;
  accuracy_trajectory: number[];
  loss_trajectory: number[];
  representation_drift_trajectory: number[];
  neighbor_consistency_trajectory: number[];
  centroid_displacement_trajectory: number[];
  total_accuracy_drop: number;
  mean_accuracy: number;
  area_under_curve: number;
}

export interface RobustnessExperimentReport {
  experiment_id: string;
  model_id: string;
  model_family: string;
  dataset_id: string;
  eval_split: string;
  layer_name: string;
  num_samples: number;
  clean_accuracy: number;
  clean_loss: number;
  evaluations: Record<string, CorruptionEvaluationSummary>;
  severity_curves: Record<string, CorruptionSeverityCurve>;
  sample_drifts: Record<string, SampleRepresentationDrift[]>;
  flagged_failures: RobustnessFailureRecord[];
  warnings: string[];
}

export interface ArchitectureRobustnessSummary {
  architecture: string;
  model_family: string;
  model_id: string;
  clean_accuracy: number;
  mean_corrupted_accuracy: number;
  mean_accuracy_drop: number;
  mean_representation_drift: number;
  mean_neighbor_overlap: number;
  mean_centroid_displacement: number;
  total_parameters: number | null;
}

export interface CrossArchitectureRobustnessReport {
  comparison_id: string;
  name: string;
  dataset_id: string;
  architectures: Record<string, ArchitectureRobustnessSummary>;
  detailed_reports: Record<string, RobustnessExperimentReport>;
  coordinate_space_note: string;
}

export interface RobustnessExperimentMeta {
  experiment_id: string;
  name: string;
  architectures: string[];
  corruption_types: string[];
  severities: number[];
  layers: Record<string, string[]>;
  data_budgets: number[];
  num_classes: number;
  class_names: string[];
}

export interface RobustnessDatasetPayload {
  metadata: RobustnessExperimentMeta;
  reports: Record<string, RobustnessExperimentReport>;
  comparison: CrossArchitectureRobustnessReport;
}
