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
> **Active Development**: PRISM has completed **Phase 21: Video & Temporal Representation Learning**.
> The platform extends beyond static-image representation analysis into controlled short-video and temporal representation research, investigating how visual representations change when information is aggregated across time:
> - **Video & Temporal Contracts** (`VideoSample`, `VideoBatch`, `MotionTrajectory`, `FrameMetadata`) preserving frame identity $[T \times C \times H \times W]$ and trajectory metadata.
> - **Deterministic Synthetic Video Dataset** (`SyntheticVideoGenerator`) producing horizontal, vertical, stationary, and static-sequence controls without external media files.
> - **Temporal Frame Encoder Adapter** (`TemporalFrameEncoder`) sharing 100% of image encoder weights across timesteps $[N \cdot T, C, H, W] \to [N, T, D]$ for CNNs, ResNets, and ViTs.
> - **Lightweight Aggregators & Vanilla RNN** (`MeanTemporalPooling`, `MaxTemporalPooling`, `LastFramePooling`, `LearnedTemporalPooling` with softmax attention, and `SimpleRNN` $h_t = \tanh(W_x x_t + W_h h_{t-1} + b)$ with exact analytical BPTT).
> - **Temporal Consistency & Motion Sensitivity** (adjacent Euclidean drift, cosine similarity, max temporal jump, temporal drift curve $d(h_0, h_t)$, and correlation with ground-truth velocity).
> - **Temporal Robustness & Corruptions** (`FRAME_DROP`, `FRAME_DUPLICATION`, `FRAME_SHUFFLE`, `TEMPORAL_SUBSAMPLING`, `SPATIAL_COMPOSITE`).
> - **Cross-Objective Pretraining Transfer** (Supervised, SimCLR, Reconstruction, Scratch across Frozen, Partial, and Full fine-tuning).
> - **Interactive Temporal Laboratory UI**: Frame strip, representation timeline, aggregation view / learned attention weights, shared PCA trajectory plot, robustness perturbation explorer, layer transferability panel, and failure explorer.

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
| **Phase 22** | Multi-Modal & Vision-Language Alignment | :hourglass_flowing_sand: Planned |

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
│   │       ├── ssl/           # SimCLR contrastive pretraining, projection heads, collapse diagnostics
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
