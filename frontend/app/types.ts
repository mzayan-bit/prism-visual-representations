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

// -----------------------------------------------------------------------------
// Phase 16: Explainability & Visual Attribution Types
// -----------------------------------------------------------------------------

export type AttributionMethod =
  | "input_gradient"
  | "gradient_x_input"
  | "occlusion_sensitivity"
  | "grad_cam"
  | "vit_attention";

export type TargetClassMode =
  | "predicted_class"
  | "true_class"
  | "explicit_class";

export type ChannelReductionPolicy =
  | "abs_max"
  | "abs_mean"
  | "l2_channel_norm";

export type AttributionNormalizationPolicy =
  | "none"
  | "min_max_absolute"
  | "abs_sum_normalize"
  | "signed_min_max";

export type ViTAttentionHeadPolicy = "mean_heads" | "specific_head";
export type OcclusionFillPolicy = "zero" | "image_mean";

export interface AttributionSpecification {
  method: AttributionMethod;
  target_mode: TargetClassMode;
  explicit_target_class?: number | null;
  layer_name?: string | null;
  channel_reduction: ChannelReductionPolicy;
  normalization: AttributionNormalizationPolicy;
  occlusion_window_size: [number, number];
  occlusion_stride: [number, number];
  occlusion_fill: OcclusionFillPolicy;
  occlusion_max_windows: number;
  vit_head_policy: ViTAttentionHeadPolicy;
  vit_head_index?: number | null;
  vit_layer_index: number;
  seed?: number | null;
  version: string;
}

export interface AttributionStatistics {
  min_value: number;
  max_value: number;
  mean_value: number;
  std_value: number;
  total_absolute_mass: number;
  top_10_percent_mass_fraction: number;
  top_25_percent_mass_fraction: number;
  spatial_entropy: number;
  concentration_score: number;
  center_of_mass_row: number;
  center_of_mass_col: number;
  is_finite: boolean;
}

export interface AttributionResult {
  sample_id: string;
  model_id: string;
  architecture: string;
  method: AttributionMethod;
  specification: AttributionSpecification;
  target_class: number;
  predicted_class: number;
  true_class: number | null;
  target_score: number | null;
  predicted_score: number | null;
  source_image_shape: number[];
  attribution_shape: number[];
  raw_attribution_map: number[][];
  normalized_attribution_map: number[][];
  statistics: AttributionStatistics;
  positive_mass: number;
  negative_mass: number;
  absolute_mass: number;
  method_metadata: Record<string, unknown>;
  warnings: string[];
}

export interface MethodAgreementResult {
  method_a: AttributionMethod;
  method_b: AttributionMethod;
  cosine_similarity: number;
  top_10_percent_overlap: number;
  top_25_percent_overlap: number;
  center_of_mass_displacement: number;
  concentration_difference: number;
}

export interface AttributionComparisonReport {
  sample_id: string;
  model_id: string;
  architecture: string;
  target_class: number;
  predicted_class: number;
  true_class: number | null;
  results: Record<string, AttributionResult>;
  pairwise_agreements: MethodAgreementResult[];
  cosine_similarity_matrix: Record<string, Record<string, number | null>>;
  top_10_overlap_matrix: Record<string, Record<string, number | null>>;
  mean_cross_method_agreement: number;
  warnings: string[];
}

export interface AttributionDriftSummary {
  sample_id: string;
  model_id: string;
  architecture: string;
  method: AttributionMethod;
  corruption_type: string;
  corruption_severity: number;
  clean_target_class: number;
  corrupted_target_class: number;
  clean_predicted_class: number;
  corrupted_predicted_class: number;
  prediction_preserved: boolean;
  clean_score: number | null;
  corrupted_score: number | null;
  attribution_cosine_similarity: number;
  top_10_percent_mask_overlap: number;
  top_25_percent_mask_overlap: number;
  center_of_mass_displacement: number;
  concentration_delta: number;
  representation_drift_distance: number | null;
  warnings: string[];
}

export type ExplanationFailureCategory =
  | "low_attribution_signal"
  | "method_disagreement"
  | "attribution_shift_under_corruption"
  | "prediction_flip_with_stable_attribution"
  | "large_attribution_shift_with_stable_prediction"
  | "diffuse_attribution"
  | "localized_single_region";

export interface ExplanationFailureFlag {
  category: ExplanationFailureCategory;
  severity: "low" | "medium" | "high" | "critical";
  description: string;
  metrics: Record<string, unknown>;
}

export interface ExplainabilityExperimentMeta {
  experiment_id: string;
  name: string;
  architectures: string[];
  supported_methods: Record<string, string[]>;
  layers: Record<string, string[]>;
  num_classes: number;
  class_names: string[];
  sample_ids: string[];
}

export interface ExplainabilitySamplePayload {
  sample_id: string;
  true_class: number;
  class_name: string;
  image_tensor: number[][][];
  corrupted_image_tensor: number[][][] | null;
  corruption_name: string | null;
  corruption_severity: number | null;
  predictions: Record<string, {
    predicted_class: number;
    predicted_name: string;
    score: number;
    confidence: number;
    probabilities: number[];
  }>;
  corrupted_predictions: Record<string, {
    predicted_class: number;
    predicted_name: string;
    score: number;
    confidence: number;
    probabilities: number[];
  }>;
  attributions: Record<string, Record<string, AttributionResult>>;
  corrupted_attributions: Record<string, Record<string, AttributionResult>>;
  comparison_reports: Record<string, AttributionComparisonReport>;
  drift_summaries: Record<string, Record<string, AttributionDriftSummary>>;
  failure_flags: Record<string, ExplanationFailureFlag[]>;
}

export interface ExplainabilityDemoPayload {
  metadata: ExplainabilityExperimentMeta;
  samples: ExplainabilitySamplePayload[];
}

export type TransferStrategyType =
  | "scratch_baseline"
  | "linear_probe"
  | "partial_fine_tune"
  | "full_fine_tune";

export interface ParameterFreezePlanPayload {
  frozen_parameters: string[];
  trainable_parameters: string[];
  logical_stages: Record<string, string[]>;
  total_tensors: number;
  frozen_tensors: number;
  trainable_tensors: number;
  total_scalar_elements: number;
  frozen_scalar_elements: number;
  trainable_scalar_elements: number;
  trainable_fraction: number;
}

export interface LayerTransferProbePayload {
  layer_name: string;
  representation_dim: number;
  target_num_classes: number;
  target_dataset_id: string;
  target_data_budget: number;
  probe_parameters_count: number;
  train_accuracy: number;
  val_accuracy: number;
  test_accuracy: number | null;
  train_loss: number;
  val_loss: number;
  epochs_trained: number;
  best_epoch: number;
  duration_seconds: number;
}

export interface TransferRepresentationDriftPayload {
  source_model_id: string;
  architecture: string;
  layer: string;
  transfer_strategy: string;
  num_samples: number;
  mean_euclidean_drift: number;
  median_euclidean_drift: number;
  max_euclidean_drift: number;
  mean_cosine_similarity: number;
  mean_relative_norm_change: number;
  is_frozen_backbone: boolean;
}

export interface DataEfficiencyPointPayload {
  data_budget: number;
  sample_count: number;
  strategy: TransferStrategyType;
  val_accuracy: number;
  test_accuracy: number | null;
  train_loss: number;
  val_loss: number;
  epochs_trained: number;
  best_epoch: number;
}

export interface SampleEfficiencySummaryPayload {
  architecture: string;
  target_dataset_id: string;
  points: DataEfficiencyPointPayload[];
  normalized_auc: number;
}

export interface TransferStrategyComparisonPayload {
  scratch_accuracy: number;
  linear_probe_accuracy: number;
  partial_fine_tune_accuracy: number;
  full_fine_tune_accuracy: number;
  linear_probe_gain: number;
  partial_fine_tune_gain: number;
  full_fine_tune_gain: number;
}

export interface TransferLearningReportPayload {
  transfer_id: string;
  source_model_id: string;
  target_model_id: string;
  architecture: string;
  strategy: TransferStrategyType;
  train_loss: number;
  val_loss: number;
  train_accuracy: number;
  val_accuracy: number;
  test_accuracy: number | null;
  epochs_trained: number;
  best_epoch: number;
  freeze_plan: ParameterFreezePlanPayload;
  scratch_comparison: TransferStrategyComparisonPayload | null;
  layer_probes: LayerTransferProbePayload[];
  representation_drift: TransferRepresentationDriftPayload | null;
  warnings: string[];
  duration_seconds: number;
}

export interface TransferExperimentMetaPayload {
  experiment_id: string;
  source_models: string[];
  architectures: string[];
  target_tasks: string[];
  target_budgets: number[];
  strategies: string[];
  source_classes: string[];
  target_classes: string[];
}

export interface SharedPCADriftPayload {
  pre_coordinates: number[][];
  post_coordinates: number[][];
  displacement_vectors: number[][];
  explained_variance_ratio: number[];
  mean_displacement: number;
}

export interface TransferDemoPayload {
  metadata: TransferExperimentMetaPayload;
  reports: Record<string, TransferLearningReportPayload>;
  layer_probes: Record<string, LayerTransferProbePayload[]>;
  data_efficiency: Record<string, SampleEfficiencySummaryPayload>;
  shared_pca_drifts: Record<string, SharedPCADriftPayload>;
}

