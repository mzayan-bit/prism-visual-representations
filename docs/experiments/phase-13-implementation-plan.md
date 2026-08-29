# Phase 13 implementation plan

## Research objective

Make PRISM able to execute and audit controlled CNN, residual CNN, and Vision
Transformer experiments without adding another model implementation.

## Existing infrastructure to reuse

- `ExperimentDefinition`, `ExperimentRun`, and lifecycle validation
- `ExperimentExecutionHarness` and deterministic `PreparedExecution`
- `DataPreparer`, controlled manifests, partitions, and nested subsets
- `TrainingEngine` and `EvaluationEngine`
- `TrainingResult`, `EvaluationReport`, and `MetricRecord`
- `FeatureDistributionSummary`, `ModelGradientFlowSummary`, and
  `TransformerAttentionProfile`
- existing model `get_parameters()` and `extract_representations()` contracts

## Orchestration gaps

1. There is no suite identity or suite lifecycle around multiple definitions.
2. Existing pairwise comparison contracts do not audit actual typed definition
   fields for undeclared differences.
3. Parameter counts, run-level curves, convergence, representations, attention,
   and partial failures are not aggregated into one report.
4. Data-budget and repeated-seed studies have no planning/aggregation contracts.

## Scientific invariants

- Strict comparisons reject undeclared differences.
- Architecture-appropriate differences are explicit and serialized.
- Dataset, partition, subset, preprocessing, seed, optimizer, scheduler,
  evaluation, and model metadata are compared from typed fields.
- Parameter counts come from actual trainable parameter mappings.
- Missing metrics are represented as unavailable; no result is fabricated.
- CNN/ResNet attention is not applicable, not approximated.
- Reports describe measurements and do not universalize conclusions.

## Execution and report flow

`ArchitectureComparisonSuite` → factor audit → `ExperimentExecutionHarness` →
`DataPreparer` → `TrainingEngine` → `EvaluationEngine` → structured run result →
metric/curve/convergence/gradient/representation/attention summaries →
serializable comparison report.

## Testing strategy

Unit tests will cover factor detection, strict validation, exact parameter counts,
serialization, metric extraction, convergence, aggregation, failure isolation,
and planning. A tiny CPU-only synthetic CNN/ResNet/ViT smoke test will validate
the complete workflow without downloads or network access.

## Explicit non-goals

No new neural architecture, large benchmark, hyperparameter search, distributed
execution, checkpointing, pretrained model support, representation geometry,
attention rollout, or Observatory dashboard is part of Phase 13.
