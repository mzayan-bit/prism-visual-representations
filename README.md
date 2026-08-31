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
> **Active Development**: PRISM has completed **Phase 16: Explainability & Visual Attribution Laboratory**.
> The platform introduces a complete mathematical and visual framework for producing, analyzing, comparing, and visualizing model-attribution signals across Convolutional Neural Networks (CNNs), Residual CNNs (ResNets), and Vision Transformers (ViTs):
> - **Input Gradient Saliency** ($\partial S_c / \partial x$) & **Gradient $\times$ Input** ($(\partial S_c / \partial x) \odot x$) with channel reduction policies (`ABS_MAX`, `ABS_MEAN`, `L2_CHANNEL_NORM`).
> - **Sliding-Window Occlusion Sensitivity** with configurable window size, stride, zero/mean fill policies, and safeguard thresholds.
> - **Grad-CAM** for CNNs and ResNets with spatial activation and gradient capture, channel-importance alpha weights ($\alpha_k^c$), ReLU rectification, and deterministic bilinear 2D upsampling.
> - **Vision Transformer CLS-to-Patch Attention Attribution** mapping multi-head encoder attention weights to 2D patch grids.
> - **Cross-Method Agreement Analysis** with spatial cosine similarity, top-$p$% Jaccard overlap masks, and center-of-mass displacement metrics.
> - **Attribution Drift Analysis** measuring spatial evidence stability under controlled input corruptions paired with representation drift distance.
> - **Explanation Failure Taxonomy** systematically flagging low attribution mass, method disagreement, corruption shifts, and prediction flips with stable attributions.
> - **Interactive Next.js PRISM Explainability Laboratory** with multi-mode navigation, continuous overlay sliders, colormaps, agreement matrices, and failure diagnostics.

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
