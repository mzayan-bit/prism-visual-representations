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

// ---------------------------------------------------------------------------
// Phase 18: Self-Supervised Learning (SimCLR) Types
// ---------------------------------------------------------------------------

export interface SSLMetadataPayload {
  experiment_id: string;
  title: string;
  description: string;
  method: string;
  architectures: string[];
  temperatures: number[];
  dataset_id: string;
  created_at_utc: string;
}

export interface RepresentationCollapseSummaryPayload {
  total_dimensions: number;
  mean_feature_std: number;
  near_zero_variance_dimensions: number;
  near_zero_variance_fraction: number;
  distinct_sample_cosine_spread: number;
  mean_positive_alignment_distance: number;
  is_collapsed: boolean;
  warnings: string[];
}

export interface SelfSupervisedLearningReportPayload {
  ssl_id: string;
  encoder_family: string;
  architecture: string;
  dataset_id: string;
  total_encoder_parameters: number;
  projection_head_parameters: number;
  epochs: number;
  temperature: number;
  loss_trajectory: number[];
  positive_similarity_trajectory: number[];
  negative_similarity_trajectory: number[];
  similarity_gap_trajectory: number[];
  learning_rate_trajectory: number[];
  collapse_summary: RepresentationCollapseSummaryPayload;
  linear_probe_accuracy: number | null;
  supervised_probe_accuracy: number | null;
  scratch_accuracy: number | null;
  transfer_gain_vs_scratch: number | null;
  warnings: string[];
}

export interface SupervisedVsSSLComparisonPayload {
  architecture: string;
  dataset_id: string;
  supervised_accuracy: number;
  ssl_accuracy: number;
  scratch_accuracy: number;
  supervised_feature_std: number;
  ssl_feature_std: number;
  accuracy_gap_ssl_vs_supervised: number;
  accuracy_gain_ssl_vs_scratch: number;
}

export interface SSLLabelEfficiencyPointPayload {
  budget_fraction: number;
  budget_percent_label: string;
  ssl_accuracy: number;
  supervised_accuracy: number;
  scratch_accuracy: number;
}

export interface SSLGeometryPointPayload {
  sample_id: string;
  pca_x: number;
  pca_y: number;
  class_label: number;
  class_name: string;
  is_positive_view: boolean;
}

export interface SSLLayerProbePointPayload {
  layer_id: string;
  layer_depth_index: number;
  representation_dim: number;
  ssl_accuracy: number;
  supervised_accuracy: number;
}

export interface SSLDemoPayload {
  metadata: SSLMetadataPayload;
  reports: Record<string, SelfSupervisedLearningReportPayload>;
  comparisons: Record<string, SupervisedVsSSLComparisonPayload>;
  label_efficiency: Record<string, SSLLabelEfficiencyPointPayload[]>;
  geometry_points: Record<string, SSLGeometryPointPayload[]>;
  layer_probes: Record<string, SSLLayerProbePointPayload[]>;
}

// ==========================================
// PHASE 19: RECONSTRUCTION LEARNING TYPES
// ==========================================

export interface VisualTripletSamplePayload {
  sample_id: string;
  class_name: string;
  method: string;
  original_image: number[][][];
  corrupted_or_masked_image: number[][][];
  reconstructed_image: number[][][];
  error_map: number[][];
  masked_patch_indices: number[];
  sample_mse: number;
  failure_category: string | null;
}

export interface MaskingRatioPointPayload {
  mask_ratio: number;
  mask_ratio_percent: string;
  reconstruction_mse: number;
  linear_probe_accuracy: number;
  latent_std: number;
}

export interface ThreeWayComparisonEntryPayload {
  architecture: string;
  supervised_accuracy: number;
  simclr_accuracy: number;
  reconstruction_accuracy: number;
  supervised_latent_std: number;
  simclr_latent_std: number;
  reconstruction_latent_std: number;
  supervised_compactness: number;
  simclr_compactness: number;
  reconstruction_compactness: number;
  supervised_separation: number;
  simclr_separation: number;
  reconstruction_separation: number;
}

export interface ReconstructionLayerProbeEntryPayload {
  layer_id: string;
  depth_index: number;
  supervised_accuracy: number;
  simclr_accuracy: number;
  reconstruction_accuracy: number;
}

export interface ReconstructionFailureCasePayload {
  sample_id: string;
  category: string;
  reconstruction_mse: number;
  description: string;
  patch_index: number | null;
}

export interface ReconstructionDynamicsPayload {
  epochs: number[];
  total_loss: number[];
  masked_mse: number[];
  latent_std: number[];
  learning_rate: number[];
}

export interface ReconstructionMetadataPayload {
  experiment_id: string;
  title: string;
  description: string;
  methods: string[];
  architectures: string[];
  mask_ratios: number[];
  dataset_id: string;
  created_at_utc: string;
}

export interface ReconstructionDatasetPayload {
  metadata: ReconstructionMetadataPayload;
  triplets_masked_patch: VisualTripletSamplePayload[];
  triplets_denoising: VisualTripletSamplePayload[];
  dynamics: ReconstructionDynamicsPayload;
  masking_ratio_study: MaskingRatioPointPayload[];
  three_way_comparison: ThreeWayComparisonEntryPayload[];
  layer_probes: ReconstructionLayerProbeEntryPayload[];
  failure_cases: ReconstructionFailureCasePayload[];
}

// ==========================================
// Phase 20: Spatial Transfer Types
// ==========================================

export type SpatialTaskType = "object_detection" | "semantic_segmentation";
export type PretrainingObjectiveType = "supervised" | "simclr" | "reconstruction" | "scratch";
export type SpatialTransferStrategyType =
  | "frozen_spatial_probe"
  | "partial_fine_tune"
  | "full_fine_tune";

export interface SpatialDetectionAnnotationPayload {
  class_id: number;
  class_name?: string;
  box: [number, number, number, number];
}

export interface SpatialDetectionPredictionBoxPayload {
  class_id: number;
  confidence: number;
  box: [number, number, number, number];
}

export interface SpatialDetectionSamplePayload {
  sample_id: string;
  image: number[][][];
  image_shape: [number, number, number];
  ground_truth_boxes: SpatialDetectionAnnotationPayload[];
  predicted_boxes: SpatialDetectionPredictionBoxPayload[];
}

export interface SpatialSegmentationSamplePayload {
  sample_id: string;
  image: number[][][];
  ground_truth_mask: number[][];
  predicted_mask: number[][];
  num_classes: number;
}

export interface SpatialDetectionMetricsPayload {
  total_samples: number;
  total_targets: number;
  total_predictions: number;
  matched_objects: number;
  mean_iou: number;
  precision: number;
  recall: number;
  class_accuracy: number;
  mean_localization_error: number;
  iou_threshold: number;
}

export interface SpatialSegmentationMetricsPayload {
  num_classes: number;
  total_pixels: number;
  pixel_accuracy: number;
  mean_iou: number;
  per_class_iou: Record<number, number>;
  per_class_dice: Record<number, number>;
  confusion_matrix: number[][];
}

export interface SpatialTransferReportPayload {
  report_id: string;
  specification: {
    specification_id: string;
    source_objective: PretrainingObjectiveType;
    source_experiment_id: string;
    task_type: SpatialTaskType;
    spatial_layer: string;
    transfer_strategy: SpatialTransferStrategyType;
    num_classes: number;
    learning_rate: number;
    epochs: number;
    batch_size: number;
    data_budget_fraction: number;
  };
  total_parameters: number;
  frozen_parameters: number;
  trainable_parameters: number;
  head_parameters: number;
  trainable_fraction: number;
  feature_shape: [number, number, number];
  feature_resolution: string;
  training_loss_trajectory: number[];
  epochs_completed: number;
  detection_metrics?: SpatialDetectionMetricsPayload | null;
  segmentation_metrics?: SpatialSegmentationMetricsPayload | null;
  spatial_representation_drift_cosine: number;
  spatial_representation_drift_rmse: number;
  warnings: string[];
}

export interface SpatialLayerTransferabilityRecord {
  layer: string;
  depth_index: number;
  feature_resolution: string;
  detection_mean_iou: number;
  segmentation_mean_iou: number;
  feature_channels: number;
}

export interface SpatialDataEfficiencyRecord {
  budget_fraction: number;
  supervised_iou: number;
  simclr_iou: number;
  reconstruction_iou: number;
  scratch_iou: number;
}

export interface SpatialDatasetPayload {
  meta: {
    generated_by: string;
    version: string;
    architectures: string[];
    objectives: string[];
    tasks: string[];
  };
  reports: SpatialTransferReportPayload[];
  layer_transferability: Record<string, SpatialLayerTransferabilityRecord[]>;
  data_efficiency: Record<string, SpatialDataEfficiencyRecord[]>;
  detection_samples: SpatialDetectionSamplePayload[];
  segmentation_samples: SpatialSegmentationSamplePayload[];
}

// ---------------------------------------------------------------------------
// Phase 21: Video & Temporal Representation Learning Types
// ---------------------------------------------------------------------------

export type TemporalTaskType =
  | "video_classification"
  | "temporal_representation_analysis"
  | "frame_classification";

export type TemporalAggregationType =
  | "simple_rnn"
  | "learned_temporal_pooling"
  | "mean_pool"
  | "max_pool"
  | "last_frame";

export type TemporalTransferStrategyType =
  | "frozen_frame_encoder"
  | "partial_fine_tune"
  | "full_fine_tune"
  | "frame_independent";

export type TemporalCorruptionType =
  | "frame_drop"
  | "frame_duplication"
  | "frame_shuffle"
  | "temporal_subsampling"
  | "spatial_composite";

export interface TemporalTrajectoryPayload {
  start_pos: [number, number];
  end_pos: [number, number];
  per_frame_positions: [number, number][];
  direction: string;
  velocity_magnitude: number;
  is_stationary: boolean;
}

export interface TemporalPCATrajectoryPayload {
  timestep: number;
  pca_1: number;
  pca_2: number;
}

export interface TemporalTimelineMetricPayload {
  timestep: number;
  representation_norm: number;
  adjacent_drift: number;
  adjacent_cosine_similarity: number;
  motion_displacement: number;
}

export interface TemporalVideoSamplePayload {
  video_id: string;
  frame_tensors: number[][][][]; // T x C x H x W
  frame_ids: string[];
  frame_indices: number[];
  label: number;
  frame_count: number;
  frame_shape: [number, number, number];
  motion_trajectory: TemporalTrajectoryPayload | null;
  dataset_fingerprint: string;
  split: string;
  metadata: Record<string, unknown>;
  pca_trajectory: TemporalPCATrajectoryPayload[];
  timeline_metrics: TemporalTimelineMetricPayload[];
  hidden_norms: number[];
  attention_weights: number[];
}

export interface TemporalObjectiveComparisonPayload {
  objective: string;
  label: string;
  frozen_accuracy: number;
  finetune_accuracy: number;
  temporal_consistency: number;
  sequence_drift: number;
  trainable_fraction: number;
  description: string;
}

export interface TemporalLayerProfileRecord {
  layer_name: string;
  depth_fraction: number;
  feature_dim: number;
  accuracy: number;
  consistency: number;
}

export interface TemporalAggregatorComparisonPayload {
  aggregator: TemporalAggregationType;
  label: string;
  accuracy: number;
  order_sensitive: boolean;
  temporal_params: number;
  notes: string;
}

export interface TemporalRobustnessBenchmarkPayload {
  corruption_type: TemporalCorruptionType;
  label: string;
  clean_accuracy: number;
  perturbed_accuracy: number;
  accuracy_delta: number;
  representation_drift: number;
  description: string;
}

export interface TemporalDataEfficiencyRecord {
  budget_fraction: number;
  samples: number;
  reconstruction: number;
  supervised: number;
  simclr: number;
  scratch: number;
}

export interface TemporalSequenceLengthRecord {
  num_frames: number;
  accuracy: number;
  temporal_consistency: number;
  mean_drift: number;
}

export interface TemporalCandidateFailurePayload {
  failure_type: string;
  sample_id: string;
  direction: string;
  description: string;
  severity: "low" | "medium" | "high";
}

export interface TemporalDatasetPayload {
  metadata: {
    phase: number;
    title: string;
    dataset_fingerprint: string;
    num_classes: number;
    class_names: string[];
    architectures: string[];
    pretraining_objectives: string[];
    aggregators: TemporalAggregationType[];
    transfer_strategies: TemporalTransferStrategyType[];
  };
  samples: TemporalVideoSamplePayload[];
  objective_comparisons: TemporalObjectiveComparisonPayload[];
  layer_profiles: Record<string, TemporalLayerProfileRecord[]>;
  aggregator_comparisons: TemporalAggregatorComparisonPayload[];
  robustness_benchmarks: TemporalRobustnessBenchmarkPayload[];
  data_efficiency_curves: TemporalDataEfficiencyRecord[];
  sequence_length_studies: TemporalSequenceLengthRecord[];
  candidate_failures: TemporalCandidateFailurePayload[];
}

export interface TokenizedTextPayload {
  original_text: string;
  token_strings: string[];
  token_ids: number[];
  sequence_length: number;
  attention_mask: number[];
}

export interface RetrievalCandidatePayload {
  sample_id: string;
  similarity: number;
}

export interface MultimodalSamplePayload {
  sample_id: string;
  text: string;
  captions: string[];
  class_label: number | null;
  class_name: string | null;
  split: string;
  pair_identity: string;
  image: number[][][];
  tokenized: TokenizedTextPayload;
  paired_cosine: number;
  paired_distance: number;
  image_pca: number[];
  text_pca: number[];
  i2t_rank: number;
  t2i_rank: number;
  top_text_candidates: RetrievalCandidatePayload[];
  top_image_candidates: RetrievalCandidatePayload[];
}

export interface CrossModalRetrievalSummaryPayload {
  image_to_text_r1: number;
  image_to_text_r3: number;
  image_to_text_r5: number;
  image_to_text_mrr: number;
  text_to_image_r1: number;
  text_to_image_r3: number;
  text_to_image_r5: number;
  text_to_image_mrr: number;
  sample_count: number;
  candidate_count: number;
}

export interface ZeroShotClassificationSummaryPayload {
  prompt_template: string;
  class_count: number;
  accuracy: number;
  per_class_accuracy: Record<string, number>;
  confusion_matrix: number[][];
  class_names: string[];
  top_3_accuracy: number | null;
}

export interface PromptSensitivityPayload {
  templates: string[];
  results: Record<string, ZeroShotClassificationSummaryPayload>;
  pairwise_agreements: Record<string, number>;
}

export interface CentroidAlignmentPayload {
  class_name: string;
  image_centroid: number[];
  text_centroid: number[];
  euclidean_distance: number;
  cosine_similarity: number;
}

export interface MultimodalCollapseSummaryPayload {
  visual_dim_variance: number;
  visual_feature_std: number;
  visual_pairwise_similarity: number;
  text_dim_variance: number;
  text_feature_std: number;
  text_pairwise_similarity: number;
  matched_similarity: number;
  unmatched_similarity: number;
  similarity_gap: number;
  is_collapsed: boolean;
}

export interface MultimodalObjectiveComparisonPayload {
  objective: string;
  linear_probe_accuracy: number | null;
  zero_shot_accuracy: number | null;
  image_to_text_r1: number | null;
  text_to_image_r1: number | null;
  effective_dimensionality: number;
  robustness_retention: number;
  label_supervision: string;
}

export interface MultimodalArchitectureComparisonPayload {
  architecture: string;
  i2t_r1: number;
  t2i_r1: number;
  zero_shot_acc: number;
  mean_paired_cosine: number;
  probe_acc: number;
}

export interface MultimodalCandidateFailurePayload {
  failure_type: string;
  sample_id: string;
  description: string;
  severity: "low" | "medium" | "high";
  paired_cosine?: number;
  matched_rank?: number;
  confidence_delta?: number;
  cosine_drop?: number;
}

export interface MultimodalDatasetPayload {
  metadata: {
    phase: number;
    title: string;
    dataset_fingerprint: string;
    num_classes: number;
    class_names: string[];
    prompt_templates: string[];
    architectures: string[];
    pretraining_objectives: string[];
  };
  samples: MultimodalSamplePayload[];
  training_history: Array<Record<string, number>>;
  retrieval_summary: CrossModalRetrievalSummaryPayload;
  zero_shot_summary: ZeroShotClassificationSummaryPayload;
  prompt_sensitivity: PromptSensitivityPayload;
  shared_geometry: {
    explained_variance_ratio: number[];
    mean_paired_distance: number;
    mean_paired_cosine: number;
    centroid_alignments: CentroidAlignmentPayload[];
  };
  collapse_summary: MultimodalCollapseSummaryPayload;
  robustness_benchmarks: {
    corruptions: string[];
    results: Record<
      string,
      {
        corruption: string;
        severity: number;
        mean_paired_cosine: number;
        cosine_drop: number;
        mean_visual_drift: number;
        mean_alignment_drift: number;
        image_to_text_r1: number;
        image_to_text_r3: number;
        image_to_text_mrr: number;
        zero_shot_accuracy: number | null;
      }
    >;
  };
  objective_comparisons?: MultimodalObjectiveComparisonPayload[];
  architecture_comparisons?: MultimodalArchitectureComparisonPayload[];
  candidate_failures?: MultimodalCandidateFailurePayload[];
}

// ==========================================
// Phase 23: Uncertainty, Calibration & OOD Types
// ==========================================

export interface ReliabilityBinPayload {
  bin_index: number;
  lower_bound: number;
  upper_bound: number;
  sample_count: number;
  mean_confidence: number;
  empirical_accuracy: number;
  calibration_gap: number;
}

export interface ConfidenceSubsetSummaryPayload {
  sample_count: number;
  mean_max_probability: number;
  median_max_probability: number;
  mean_entropy: number;
  mean_normalized_entropy: number;
}

export interface ClassCalibrationSummaryPayload {
  class_id: number;
  class_name: string;
  sample_count: number;
  accuracy: number;
  mean_confidence: number;
  mean_entropy: number;
  ece: number | null;
  warning: string | null;
}

export interface CalibrationReportPayload {
  sample_count: number;
  accuracy: number;
  mean_confidence: number;
  ece: number;
  mce: number | null;
  brier_score: number;
  nll: number;
  negative_log_likelihood?: number;
  mean_predictive_entropy: number;
  mean_normalized_entropy: number;
  bin_count: number;
  binning_strategy: string;
  reliability_bins: ReliabilityBinPayload[];
  correct_predictions_summary?: ConfidenceSubsetSummaryPayload;
  incorrect_predictions_summary?: ConfidenceSubsetSummaryPayload;
  error_subset_summary?: ConfidenceSubsetSummaryPayload;
  correct_subset_summary?: ConfidenceSubsetSummaryPayload;
  class_summaries?: ClassCalibrationSummaryPayload[];
  warnings?: string[];
}

export interface TemperatureScalingResultPayload {
  fitted_temperature: number;
  validation_nll_before: number;
  validation_nll_after: number;
  ece_before: number;
  ece_after: number;
  search_range: [number, number];
  fitting_method: string;
  iterations: number;
  warnings: string[];
}

export interface OODBinaryEvaluationSummaryPayload {
  score_method: string;
  auroc: number;
  aupr: number | null;
  threshold: number;
  threshold_policy: string;
  tpr_at_threshold: number;
  fpr_at_threshold: number;
  detection_accuracy_at_threshold: number;
  id_sample_count: number;
  ood_sample_count: number;
  mean_id_score: number;
  mean_ood_score: number;
  score_separation_gap: number;
}

export interface CorruptionUncertaintyCurvePayload {
  corruption_type: string;
  severities: number[];
  accuracies: number[];
  mean_confidences: number[];
  mean_entropies: number[];
  eces: number[];
  mean_representation_drifts: number[];
  mean_ood_scores: number[];
  confidence_slope: number;
  entropy_slope: number;
  is_monotonic_entropy: boolean;
}

export interface PredictionFlipUncertaintyPayload {
  sample_id: string;
  corruption_type: string;
  severity: number;
  clean_prediction: number;
  corrupted_prediction: number;
  clean_confidence: number;
  corrupted_confidence: number;
  clean_entropy: number;
  corrupted_entropy: number;
  representation_drift: number;
}

export interface RepresentationConfidenceRelationshipPayload {
  centroid_distance_pearson_correlation: number | null;
  knn_distance_pearson_correlation: number | null;
  correct_mean_centroid_distance: number;
  incorrect_mean_centroid_distance: number;
  correct_mean_knn_distance: number;
  incorrect_mean_knn_distance: number;
}

export interface UncertaintyAnalysisReportPayload {
  model_id: string;
  architecture: string;
  source_objective: string;
  dataset_fingerprint: string;
  split: string;
  representation_layer: string;
  seed: number;
  calibration_report: CalibrationReportPayload;
  temperature_scaling: TemperatureScalingResultPayload | null;
  calibrated_report: CalibrationReportPayload | null;
  ood_evaluations: Record<string, OODBinaryEvaluationSummaryPayload>;
  representation_relationship: RepresentationConfidenceRelationshipPayload | null;
  corruption_curves: CorruptionUncertaintyCurvePayload[];
  prediction_flips: PredictionFlipUncertaintyPayload[];
  failure_counts: Record<string, number>;
  warnings: string[];
}

export interface UncertaintySampleItemPayload {
  sample_id: string;
  category: string;
  predicted_class: number;
  true_class: number | null;
  is_correct: boolean;
  confidence: number;
  entropy: number;
  nearest_centroid_class: string;
  centroid_distance: number;
  knn_distance: number;
  msp_score: number;
  is_ood_detected: boolean;
}

export interface UncertaintyObjectiveComparisonPayload {
  objective: string;
  architecture: string;
  accuracy: number;
  ece: number;
  brier_score: number;
  nll: number;
  mean_entropy: number;
  ood_msp_auroc: number;
  ood_centroid_auroc: number;
  ood_knn_auroc: number;
  temperature: number;
}

export interface UncertaintyArchitectureComparisonPayload {
  architecture: string;
  accuracy: number;
  ece: number;
  brier_score: number;
  nll: number;
  mean_entropy: number;
  ood_msp_auroc: number;
  ood_centroid_auroc: number;
  ood_knn_auroc: number;
}

export interface UncertaintyDatasetPayload {
  meta: {
    phase: number;
    title: string;
    model_id: string;
    architecture: string;
    source_objective: string;
    dataset_fingerprint: string;
    split: string;
    representation_layer: string;
    seed: number;
    num_classes: number;
    class_names: string[];
    architectures: string[];
    pretraining_objectives: string[];
    calibration_modes: string[];
    ood_score_methods: string[];
    corruptions: string[];
  };
  report: UncertaintyAnalysisReportPayload;
  reference_set: Record<string, unknown>;
  samples: UncertaintySampleItemPayload[];
  objective_comparisons: UncertaintyObjectiveComparisonPayload[];
  architecture_comparisons: UncertaintyArchitectureComparisonPayload[];
  ood_spec: Record<string, unknown>;
}


