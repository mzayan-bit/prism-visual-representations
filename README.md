# PRISM — Probing the Evolution of Visual Representations

PRISM is an open-source, research-oriented computer vision platform engineered to systematically study how visual representations are acquired, structured, generalized, and transferred across different learning paradigms.

```
       ┌────────────────────────────────────────────────────────┐
       │                         PRISM                          │
       │    Probing the Evolution of Visual Representations     │
       └────────────────────────────────────────────────────────┘
                                    │
    ┌───────────────────────────────┼───────────────────────────────┐
    ▼                               ▼                               ▼
Linear & Shallow           Deep Convolutions &              Transformers &
Baselines                  Residual Learning                Self-Supervision
(Softmax, Linear)          (MLP, CNN, ResNet, Schedulers)   (ViT, Self-Attn, SSL)
```

---

## Project Status

> [!NOTE]
> **Active Development**: PRISM has completed **Phase 24: Cross-Paradigm Benchmark Orchestration & Research Synthesis**.
> The platform integrates all prior PRISM visual representation learning paradigms into a unified, controlled benchmark orchestration, evidence synthesis, reporting, and experiment-gap analysis layer:
> - **Unified Benchmark Schema & Registries** (`MetricDefinition`, `FactorRegistry`, `MetricRegistry` covering 25 canonical metrics across 10 evaluation axes with methodological safeguards).
> - **Pure Functional Report Adapters** (Bridging all 11 prior PRISM report schemas into standard `BenchmarkResultCell` instances).
> - **Multi-Seed Statistical Aggregation & Rigorous Provenance** (`aggregate_repeated_seeds` computing sample mean, variance, min/max, median, and strict $N=1$ warning attachment; provenance conflict detection).
> - **Controlled Comparison & Audit Engine** (`audit_comparison_control` enforcing strict factor matching; `compute_pairwise_comparison` with direction-aware effect sizes).
> - **Matrix, Table & Coverage Generators** (`build_benchmark_matrix`, `build_benchmark_table`, `build_coverage_matrix`, `compute_campaign_coverage_summary`, `detect_evidence_gaps`, `generate_missing_experiment_plan`).
> - **Cross-Paradigm Synthesis & Pareto Frontiers** (`synthesize_cross_architecture`, `synthesize_cross_objective`, `extract_representation_profile` across 10 orthogonal axes, `compute_pareto_front`, `extract_tradeoff_pairs`).
> - **Grounded Research Findings & Formal Reports** (`generate_research_findings` with evidence strength ratings and caveats; `compile_prism_research_report` with reproducibility manifest, figure specifications, and JSON/Markdown/CSV export).
> - **Interactive Research Benchmark Observatory UI**: Coverage matrix, benchmark tables, architecture synthesis, objective synthesis, multi-dimensional profile radar, Pareto frontier explorer, tradeoff scatter, grounded findings panel, evidence gap planner, provenance drawer, custom report builder, and report preview modal.

---

## Milestone Progress & Roadmap

| Phase | Focus Area | Status |
| --- | --- | --- |
| **Phase 1** | Repository Foundation, Architecture Scaffolding & CI | :white_check_mark: Completed |
| **Phase 2** | Research Core: Domain Contracts, Lifecycle, Fingerprinting & Serialization | :white_check_mark: Completed |
| **Phase 3** | Reproducibility Runtime Infrastructure, Seeding & Execution Harness | :white_check_mark: Completed |
| **Phase 4** | Controlled Dataset Abstractions & Fixed Partition Manifests | :white_check_mark: Completed |
| **Phase 5** | Executable Dataset Pipeline & Deterministic Data Loading | :white_check_mark: Completed |
| **Phase 6** | Linear Classifiers & Pixel Baselines | :white_check_mark: Completed |
| **Phase 7** | Deep Learning Baselines (MLPs, Optimization & Regularization) | :white_check_mark: Completed |
| **Phase 8** | Convolutional Architectures (CNNs & Spatial Representations) | :white_check_mark: Completed |
| **Phase 9** | Normalization, Stable Optimization & Feature Distribution Tracking | :white_check_mark: Completed |
| **Phase 10** | Residual Learning, Skip Connections, Plain-vs-ResNet & Gradient Flow | :white_check_mark: Completed |
| **Phase 11** | Learning Rate Scheduling & Reproducible Optimization Control | :white_check_mark: Completed |
| **Phase 12** | Vision Transformers (ViT) & Attention Geometry | :white_check_mark: Completed |
| **Phase 13** | Controlled CNN / ResNet / Vision Transformer Architecture Experiments | :white_check_mark: Completed |
| **Phase 14** | Representation Geometry Observatory & Manifold Analysis | :white_check_mark: Completed |
| **Phase 15** | Robustness Under Corruptions & Distribution Shifts | :white_check_mark: Completed |
| **Phase 16** | Explainability & Visual Attribution Laboratory | :white_check_mark: Completed |
| **Phase 17** | Transfer Learning & Representation Reuse Laboratory | :white_check_mark: Completed |
| **Phase 18** | Self-Supervised Learning & Contrastive Pretraining | :white_check_mark: Completed |
| **Phase 19** | Generative / Reconstruction-Based Representation Learning | :white_check_mark: Completed |
| **Phase 20** | Detection & Segmentation Representation Transfer | :white_check_mark: Completed |
| **Phase 21** | Video & Temporal Representation Learning | :white_check_mark: Completed |
| **Phase 22** | Multimodal Vision-Language Representation Alignment | :white_check_mark: Completed |
| **Phase 23** | Uncertainty, Calibration & Out-of-Distribution Representation Analysis | :white_check_mark: Completed |
| **Phase 24** | Cross-Paradigm Representation Benchmark & Diagnostic Synthesis | :white_check_mark: Completed |

---

## Repository Architecture

PRISM is organized as a modular monorepo cleanly separating Python research components from configuration contracts, documentation, and the future web interface:

```
prism-visual-representations/
├── backend/                   # Python research engine and package
│   ├── src/
│   │   └── prism/             # Core library package
│   │       ├── api/           # API serving layer and precomputed benchmarks
│   │       ├── artifacts/     # Artifact contracts and references
│   │       ├── core/          # Base enums, identifiers, errors, metadata
│   │       ├── data/          # Samples, universes, materialization, ordering, batching
│   │       ├── experiments/   # Definitions, runs, harness, seeding, comparisons
│   │       ├── models/        # Linear, MLP, CNN, ResNet, residual blocks, patches, attention, norm
│   │       ├── training/      # Training engine, losses, SGD, schedulers, gradient flow
│   │       ├── evaluation/    # Evaluation engine, metrics, and structured reports
│   │       ├── representations/# Representation descriptors, feature batches, summaries
│   │       ├── transfer/      # Transfer specifications, freeze plans, linear probes, retention
│   │       ├── spatial/       # Detection and segmentation spatial transfer
│   │       ├── temporal/      # Video and temporal representation learning
│   │       ├── multimodal/    # Dual-encoder vision-language alignment, tokenizer, symmetric loss
│   │       ├── uncertainty/   # Probabilities, calibration, temperature scaling, OOD scores, corruptions
│   │       ├── benchmarking/  # Cross-paradigm orchestration, synthesis, reports, coverage, registry
│   │       ├── robustness/    # Corruptions, distribution shifts, OOD tests
│   │       ├── explainability/# Saliency, attention rollout, Grad-CAM
│   │       └── utils/         # Seeding, hashing, structured logging
│   └── tests/                 # Backend unit, smoke, and integration test suites
│
├── frontend/                  # Next.js / TypeScript research observatory
│   ├── app/                   # App Router pages and layout
│   └── ...
│
├── configs/                   # Declarative YAML configurations
├── docs/                      # Technical and methodology documentation
├── experiments/               # Research artifacts and analyses
├── artifacts/                 # Generated run outputs (checkpoints, metrics, figures)
└── tests/                     # Top-level smoke, unit, and integration test suites
```

---

## Getting Started

### Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended)
- Node.js 18+ & pnpm (for frontend observatory)

### Installation & Quality Validation

```bash
# Sync environment dependencies
uv sync

# Run static quality checks, typing, and test suites
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest

# Alternatively, run via top-level Makefile
make check
```
