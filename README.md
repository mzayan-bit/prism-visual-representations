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
> **Active Development**: PRISM has completed **Phase 17: Transfer Learning & Representation Reuse Laboratory**.
> The platform introduces a rigorous empirical framework for evaluating how visual representations learned on source domains transfer to downstream target tasks across Convolutional Neural Networks (CNNs), Residual Networks (ResNets), and Vision Transformers (ViTs):
> - **Source Model State & Snapshot Integrity** (`ModelStateSnapshot`, `create_model_state_snapshot`, `restore_model_from_snapshot`) with tensor checksumming and shape validation.
> - **Strict Parameter Freezing** (`ParameterFreezePlan`, `create_freeze_plan`) integrated directly into `SGDOptimizer` excluding frozen parameters from velocity updates and weight decay.
> - **Modular Classifier Head Replacement** (`replace_classifier_head`) dynamically adapting output dimensions across Linear, MLP, CNN, ResNet, and ViT architectures.
> - **Four Core Transfer Strategies**: Scratch Baseline (`SCRATCH_BASELINE`), Linear Probe (`LINEAR_PROBE`), Partial Fine-Tuning (`PARTIAL_FINE_TUNE`), and Full Fine-Tuning (`FULL_FINE_TUNE`).
> - **Layer Transferability Probes** (`LayerTransferProbeResult`, `probe_all_layers_transferability`) quantifying target classification utility at arbitrary intermediate layers.
> - **Representation Retention & Drift** (`TransferRepresentationDriftSummary`, `compute_transfer_shared_pca`) measuring Euclidean drift, cosine similarity, norm delta, and shared PCA coordinate displacement.
> - **Label-Efficiency Transfer Scaling Curves** quantifying target sample efficiency across nested data budgets (10% → 100%) and normalized Area Under Curve (AUC).
> - **Interactive Next.js PRISM Transfer Laboratory** featuring parameter freeze maps, strategy comparison matrices, label-efficiency scaling charts, depth probe bars, and shared PCA drift vector fields.

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
| **Phase 18** | Self-Supervised Learning & Contrastive Pretraining | :hourglass_flowing_sand: Planned |

---

## Research Paradigm Progression

```
Pixels and Linear Models
       ↓
Neural Networks & Backpropagation (MLPs)
       ↓
Optimization, Schedulers & Regularization
       ↓
Convolutional Architectures & Spatial Feature Maps (CNNs)
       ↓
Batch Normalization & Representation Distribution Stability
       ↓
Residual Learning, Explicit Skip Connections & Gradient Flow
       ↓
Learning Rate Scheduling & Optimization Control
       ↓
Vision Transformers & Attention Geometry
       ↓
Self-Supervised Learning (Contrastive / Masked)
       ↓
Representation Geometry & Manifold Analysis (CKA, Spectra, Probing)
       ↓
Robustness under Corruptions & Distribution Shifts
       ↓
Explainability across CNNs and Transformers
       ↓
Temporal & Video Understanding
       ↓
Generative-Data Experiments
       ↓
Dense Vision Transfer (Detection & Segmentation)
```

---

## Repository Architecture

PRISM is organized as a modular monorepo cleanly separating Python research components from configuration contracts, documentation, and the future web interface:

```
prism-visual-representations/
├── backend/                   # Python research engine and package
│   ├── src/
│   │   └── prism/             # Core library package
│   │       ├── api/           # Future API serving layer
│   │       ├── artifacts/     # Artifact contracts and references
│   │       ├── core/          # Base enums, identifiers, errors, metadata
│   │       ├── data/          # Samples, universes, materialization, ordering, batching
│   │       ├── experiments/   # Definitions, runs, harness, seeding, comparisons
│   │       ├── models/        # Linear, MLP, CNN, ResNet, residual blocks, patches, attention, norm
│   │       ├── training/      # Training engine, losses, SGD, schedulers, gradient flow
│   │       ├── evaluation/    # Evaluation engine, metrics, and structured reports
│   │       ├── representations/# Representation descriptors, feature batches, summaries
│   │       ├── robustness/    # Corruptions, distribution shifts, OOD tests
│   │       ├── explainability/# Saliency, attention rollout, Grad-CAM
│   │       ├── visualization/ # Projections (UMAP/t-SNE), figure generation
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
