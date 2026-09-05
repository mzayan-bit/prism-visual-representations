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
> **Active Development**: PRISM has completed **Phase 23: Uncertainty, Calibration & Out-of-Distribution Representation Analysis**.
> The platform provides a rigorous foundation for probing predictive confidence, probability calibration, and out-of-distribution (OOD) representation geometry in visual models:
> - **Uncertainty & Calibration Foundations** (`compute_stable_softmax`, `compute_predictive_entropy`, `compute_normalized_entropy`, `compute_logit_margin`, `compute_probability_margin`, `compute_reliability_bins`, `compute_expected_calibration_error`, `compute_maximum_calibration_error`, `compute_brier_score`, `compute_negative_log_likelihood`).
> - **Post-Hoc Scalar Temperature Scaling** (`apply_temperature_scaling`, `fit_temperature_scaling` via deterministic coarse-fine grid search on validation NLL with invariant test classification accuracy).
> - **OOD Scoring & Reference Geometry** (`compute_max_softmax_ood_score`, `compute_entropy_ood_score`, `compute_class_centroid_ood_score`, `compute_knn_ood_score`, `compute_energy_ood_score`, `build_ood_reference_set`, `compute_class_centroids`, `compute_intra_class_radii`).
> - **OOD Discrimination Evaluation** (`compute_auroc` via exact Mann-Whitney U rank-sum integration, `compute_aupr` trapezoidal integration, `select_ood_threshold` with fixed/quantile/target-TPR policies).
> - **Corruption Uncertainty Trajectories** (`evaluate_corruption_uncertainty`, `CorruptionUncertaintyCurve`, `PredictionFlipUncertainty` tracking confidence decay, entropy dynamics, and representation drift).
> - **Uncertainty Failure Taxonomy** (`detect_uncertainty_failures` flagging `HIGH_CONFIDENCE_ERROR`, `LOW_CONFIDENCE_CORRECT`, `HIGH_CONFIDENCE_OOD`, `CALIBRATION_OUTLIER`, `CORRUPTION_OVERCONFIDENCE`, `OOD_NEAR_KNOWN_STRUCTURE`, `ID_REPRESENTATION_OUTLIER`, `NON_MONOTONIC_UNCERTAINTY`).
> - **Interactive Uncertainty Laboratory UI**: Reliability diagram with equal-width binning, confidence histogram by correctness, temperature scaling before/after panel, OOD score distribution inspector, AUROC curve, representation novelty scatter, OOD sample explorer, corruption uncertainty curves, cross-objective comparison, and failure taxonomy explorer.

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
| **Phase 24** | Cross-Paradigm Representation Benchmark & Diagnostic Synthesis | :hourglass_flowing_sand: Planned |

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
