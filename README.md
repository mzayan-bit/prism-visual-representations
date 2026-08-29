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
> **Active Development**: PRISM has completed **Phase 14: Representation Geometry Observatory**.
> The platform introduces a complete mathematical and visual framework for representation geometry analysis: deterministic Jacobi Principal Component Analysis (PCA) with consistent sign orientation, class centroid summaries, intra-class compactness ($\bar{d}_{\text{intra}}$), inter-class centroid separation ($\bar{d}_{\text{inter}}$), separation-to-compactness ratios, exact $k$-nearest neighbor consistency, candidate failure case discovery, layer-wise geometric evolution profiling across network depth, cross-architecture geometry benchmarks (CNN vs ResNet vs ViT) on matched data budgets, and an interactive Next.js PRISM Observatory research dashboard.

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
| **Phase 15** | Robustness Under Corruptions & Distribution Shifts | :hourglass_flowing_sand: Planned |
| **Phase 16** | Comparative Explainability (Attributions, Rollout, Grad-CAM) | :hourglass_flowing_sand: Planned |
| **Phase 17** | Downstream Dense Transfer (Detection & Segmentation) | :hourglass_flowing_sand: Planned |

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
