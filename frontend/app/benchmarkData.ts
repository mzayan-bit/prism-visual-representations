import benchmarkDataJson from "./data/benchmarkDataset.json";

export interface BenchmarkCampaignPayload {
  campaign_id: string;
  title: string;
  description: string;
  research_questions: Array<{
    question_id: string;
    natural_language_question: string;
    independent_variables: string[];
    independent_values: string[];
    dependent_metrics: string[];
    controlled_factors: Record<string, unknown>;
    limitations: string[];
  }>;
  architectures: string[];
  objectives: string[];
  datasets: string[];
  tasks: string[];
  seeds: number[];
  budgets: number[];
  fingerprint: string;
}

export interface CampaignCoverageSummaryPayload {
  campaign_id: string;
  planned_experiments_count: number;
  completed_experiments_count: number;
  partial_experiments_count: number;
  failed_experiments_count: number;
  missing_experiments_count: number;
  not_applicable_count: number;
  completion_fraction: number;
  evaluated_seeds_count: number;
  warnings: string[];
}

export interface CoverageMatrixPayload {
  matrix_id: string;
  row_factor: string;
  column_factor: string;
  row_values: string[];
  column_values: string[];
  grid: Record<string, Record<string, Record<string, number>>>;
  warnings: string[];
}

export interface BenchmarkTablePayload {
  table_id: string;
  title: string;
  research_question_id: string | null;
  row_factor: string;
  column_factor: string;
  metric_id: string;
  unit: string;
  metric_direction: string;
  rows: Array<Record<string, unknown>>;
  footnotes: string[];
  control_status: string;
  warnings: string[];
}

export interface RepresentationProfilePayload {
  profile_id: string;
  architecture: string;
  objective: string;
  semantic_performance: number | null;
  geometry: number | null;
  label_efficiency: number | null;
  transferability: number | null;
  robustness: number | null;
  spatial_transfer: number | null;
  temporal_transfer: number | null;
  calibration: number | null;
  ood_separation: number | null;
  multimodal_alignment: number | null;
  metadata: Record<string, unknown>;
}

export interface ParetoAnalysisPayload {
  analysis_id: string;
  metric_ids: string[];
  candidate_experiment_ids: string[];
  non_dominated_experiment_ids: string[];
  dominated_relationships: Record<string, string[]>;
  exclusions: string[];
  missing_metric_warnings: string[];
}

export interface TradeoffPointPayload {
  experiment_id: string;
  factors: Record<string, unknown>;
  x_metric: string;
  x_value: number;
  y_metric: string;
  y_value: number;
  note: string;
}

export interface ResearchFindingPayload {
  finding_id: string;
  research_question_id: string;
  statement: string;
  supporting_result_ids: string[];
  comparison_audit: {
    comparison_id: string;
    factor_a: Record<string, unknown>;
    factor_b: Record<string, unknown>;
    varied_factors: string[];
    controlled_factors: string[];
    status: string;
    is_strictly_controlled: boolean;
    confounding_warnings: string[];
  } | null;
  effect_size_delta: number | null;
  scope: Record<string, unknown>;
  caveats: string[];
  evidence_strength: string;
}

export interface EvidenceGapPayload {
  gap_id: string;
  research_question_id: string;
  missing_factor_combination: Record<string, unknown>;
  missing_metric_id: string | null;
  missing_seed_count: number;
  rationale: string;
}

export interface MissingExperimentPlanPayload {
  plan_id: string;
  campaign_id: string;
  missing_experiments: Array<Record<string, unknown>>;
  estimated_work_units: number;
  warnings: string[];
}

export interface BenchmarkResultCellPayload {
  result_id: string;
  experiment_id: string;
  experiment_fingerprint: string;
  metric_id: string;
  value: number | null;
  status: string;
  seed: number | null;
  source_report_type: string;
  source_run_id: string;
  factors: Record<string, unknown>;
  warnings: string[];
  provenance: Record<string, unknown>;
}

export interface BenchmarkDatasetPayload {
  campaign: BenchmarkCampaignPayload;
  coverage_summary: CampaignCoverageSummaryPayload;
  coverage_matrix: CoverageMatrixPayload;
  benchmark_tables: BenchmarkTablePayload[];
  profiles: RepresentationProfilePayload[];
  pareto_analysis: ParetoAnalysisPayload;
  tradeoff_analysis: TradeoffPointPayload[];
  findings: ResearchFindingPayload[];
  evidence_gaps: EvidenceGapPayload[];
  missing_plan: MissingExperimentPlanPayload;
  report_summary: {
    report_id: string;
    title: string;
    executive_summary: string;
    methodology_summary: string;
    manifest: Record<string, unknown>;
  };
  architecture_synthesis: Record<string, Record<string, { mean?: number; std?: number | null }>>;
  objective_synthesis: Record<string, Record<string, { mean?: number; std?: number | null }>>;
  all_cells: BenchmarkResultCellPayload[];
}

const rawData = benchmarkDataJson as unknown as BenchmarkDatasetPayload;

export function getBenchmarkDataset(): BenchmarkDatasetPayload {
  return rawData;
}

export function getBenchmarkCampaign(): BenchmarkCampaignPayload {
  return rawData.campaign;
}

export function getCoverageSummary(): CampaignCoverageSummaryPayload {
  return rawData.coverage_summary;
}

export function getCoverageMatrix(): CoverageMatrixPayload {
  return rawData.coverage_matrix;
}

export function getBenchmarkTables(): BenchmarkTablePayload[] {
  return rawData.benchmark_tables || [];
}

export function getRepresentationProfiles(): RepresentationProfilePayload[] {
  return rawData.profiles || [];
}

export function getParetoAnalysis(): ParetoAnalysisPayload {
  return rawData.pareto_analysis;
}

export function getTradeoffAnalysis(): TradeoffPointPayload[] {
  return rawData.tradeoff_analysis || [];
}

export function getResearchFindings(): ResearchFindingPayload[] {
  return rawData.findings || [];
}

export function getEvidenceGaps(): EvidenceGapPayload[] {
  return rawData.evidence_gaps || [];
}

export function getMissingExperimentPlan(): MissingExperimentPlanPayload {
  return rawData.missing_plan;
}

export function getAllBenchmarkCells(): BenchmarkResultCellPayload[] {
  return rawData.all_cells || [];
}
