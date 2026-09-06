# PRISM Benchmark Orchestration & Research Synthesis Guide

## Overview
Phase 24 integrates all prior PRISM visual representation learning paradigms (Supervised, Contrastive/SimCLR, Masked Autoencoding/MAE, Multimodal CLIP, Geometric, Information Theoretic, Dynamic Routing, Calibration, and Uncertainty) into a unified, controlled benchmark orchestration, evidence synthesis, reporting, and experiment-gap analysis layer.

The central research question answered by this subsystem is:
> **"What conclusions about visual representation learning are genuinely supported by the total body of experiments already implemented in PRISM?"**

---

## Core Principles & Research Contracts

1. **Strict No-Fake-Result Policy**:
   Every benchmark measurement cell explicitly carries a typed status:
   - `OBSERVED`: Ground-truth empirical measurement from a verified experiment run.
   - `AGGREGATED`: Multi-seed sample mean and variance computed over $N \ge 2$ seeds.
   - `MISSING`: Planned experimental factor combination that has not yet been executed.
   - `FAILED`: Experiment crashed or was interrupted during execution.
   - `NOT_APPLICABLE`: The metric is mathematically or conceptually inapplicable to the paradigm (e.g., zero-shot text retrieval for a unimodal supervised ResNet).
   Missing results are **never** encoded as `0.0`, default placeholders, or interpolated approximations.

2. **Controlled Factor Auditing**:
   Every pairwise comparison and matrix row/column comparison executes a formal `ComparisonControlAudit`. Comparisons are classified into:
   - `STRICTLY_CONTROLLED`: Exactly one independent variable is varied while all shared factors (dataset split, optimization budget, seed, resolution) are held strictly equal.
   - `PARTIALLY_CONTROLLED`: Only random seed varies or declared non-interfering hyperparameter adjustments exist.
   - `DESCRIPTIVE_ONLY`: Multiple factors vary simultaneously; findings are strictly qualified with scientific caveats.
   - `INVALID_COMPARISON`: Incompatible datasets, loss divergences, or uncontrolled multi-variable confounding.

3. **Grounded Findings & Evidence Strength**:
   Findings are generated using deterministic templates conditioned on sample counts and control audits:
   - `SUPPORTED_BY_REPEATED_RUNS` ($N \ge 3$ seeds with statistical significance and strict control).
   - `SUPPORTED_BY_SINGLE_RUN` (Single-seed observation with strict control; includes caveat).
   - `DESCRIPTIVE_ONLY` (Confounded or exploratory observation).
   - `INSUFFICIENT_EVIDENCE` (Evidence gaps detected or planned runs pending).

4. **Multi-Objective Tradeoffs & Pareto Frontiers**:
   No single scalar aggregate metric collapses performance, robustness, calibration, efficiency, and alignment. Multi-dimensional representation profiles (spanning 10 orthogonal axes) and non-dominated Pareto frontiers preserve the true trade-offs between representation properties.

---

## Architecture & Module Structure

```
backend/src/prism/benchmarking/
├── __init__.py           # Package namespace exports
├── enums.py              # Canonical Enums (ResultStatus, EvidenceStrength, MetricCategory, etc.)
├── contracts.py          # Strict Pydantic contracts for cells, matrices, profiles, findings, reports
├── registry.py           # FactorRegistry & MetricRegistry (25 canonical metrics with methodological notes)
├── adapters.py           # Pure functional adapters transforming all 11 prior PRISM report schemas
├── store.py              # In-memory BenchmarkResultStore with provenance indexing and conflict detection
├── aggregation.py        # Multi-seed sample mean, variance, min/max, median aggregation
├── comparisons.py        # Control audit engine and direction-aware pairwise comparison
├── matrices.py           # 2D BenchmarkMatrix and structured BenchmarkTable builders
├── synthesis.py          # Cross-architecture, cross-objective, multi-dimensional profiles, Pareto fronts
├── coverage.py           # CoverageMatrix, CampaignCoverageSummary, EvidenceGap detection, MissingExperimentPlan
├── findings.py           # Grounded ResearchFinding generation with strength ratings and caveats
├── reporting.py          # PRISMResearchReport compilation, figures, and reproducibility manifest
├── runner.py             # BenchmarkCampaignRunner with dry-run, selective execution, and failure tracking
├── service.py            # BenchmarkService for API queries and frontend data export
└── export.py             # Deterministic JSON, Markdown, and CSV serializers
```

---

## Metric Registry & Canonical Evaluation Axes

PRISM orchestrates evaluations across 10 canonical representation axes:

1. **Semantic Performance**: Classification Accuracy, Top-k Accuracy, Macro F1.
2. **Linear Probe Quality**: Linear probe top-1 accuracy on frozen visual representations.
3. **Sample Efficiency**: Low-data regime accuracy (10% and 1% label budgets).
4. **Out-of-Distribution Robustness**: Corruption accuracy drop across Gaussian noise, blur, and contrast perturbations.
5. **Calibration & Uncertainty**: Expected Calibration Error (ECE), Temperature-Scaled ECE, Brier Score, OOD AUROC.
6. **Feature Geometry & Intrinsic Dimensionality**: Participation ratio, explained variance ratio (PCA 90%), effective rank.
7. **Representation Similarity & Invariance**: Centered Kernel Alignment (CKA), Canonical Correlation Analysis (SVCCA), Augmentation invariance score.
8. **Attribution Consensus**: Saliency and Grad-CAM intersection-over-union across architecture stems.
9. **Efficiency & Computational Cost**: Active parameters count, forward inference latency (ms), FLOPs proxy.
10. **Multimodal Alignment**: Image-to-Text Recall@1, Text-to-Image Recall@1, Mean Reciprocal Rank (MRR), Zero-shot Top-1 Accuracy, Embedding Collapse gap.

---

## CLI & Programmatic Usage

### Executing a Benchmark Campaign
```python
from prism.benchmarking import (
    BenchmarkCampaign,
    BenchmarkCampaignRunner,
    BenchmarkResultStore,
    canonical_metric_registry,
)

campaign = BenchmarkCampaign(
    campaign_id="prism_full_cross_paradigm_v1",
    title="PRISM Full Cross-Paradigm Benchmark",
    description="Cross-architecture and cross-objective evaluation across CIFAR-10 and Synthetic shapes.",
    architectures=["resnet", "vit", "cnn", "mlp"],
    objectives=["supervised", "simclr", "reconstruction", "vision_language"],
    datasets=["cifar10"],
    tasks=["classification"],
    seeds=[42, 100, 2024],
    budgets=[1.0, 0.1],
)

runner = BenchmarkCampaignRunner(campaign=campaign)
report = runner.run_campaign(dry_run=False)
```

### Exporting Reports and Manifests
```python
from prism.benchmarking.export import (
    export_report_to_json,
    export_report_to_markdown,
    export_table_to_csv,
)

# Deterministic JSON Export
json_str = export_report_to_json(report)

# Formatted Research Markdown Report
md_str = export_report_to_markdown(report)

# CSV Export for Table Data
for tbl in report.tables:
    csv_str = export_table_to_csv(tbl)
```
