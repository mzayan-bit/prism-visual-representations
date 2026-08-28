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
Baselines                  Normalization                    Self-Supervision
(Softmax, Linear)          (MLP, Conv2D, BatchNorm, CNN)   (ViT, Self-Attn, SSL)
```

---

## Project Status

> [!NOTE]
> **Active Development**: PRISM has completed **Phase 9: Normalization, Stable CNN Optimization, Feature Distribution Tracking, and Controlled Normalization Comparisons**.
> The platform supports vector and spatial batch normalization layers (`BatchNorm1D`, `BatchNorm2D`), strict train/eval mode semantics with exponential moving average running statistics tracking, clean separation between trainable affine parameters ($\gamma, \beta$) and non-trainable state (`running_mean`, `running_var`), normalization-augmented CNN and MLP models, statistical feature distribution summaries (`FeatureDistributionSummary`, `compute_distribution_summary`), representation stability comparison utilities (`compare_distribution_summaries`), and auditable controlled comparisons (`create_normalization_comparison`).
>
> Vision Transformers (ViT), self-attention geometry, and self-supervised learning paradigms will be introduced in subsequent planned phases.

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
| **Phase 10** | Vision Transformers (ViT) & Attention Geometry | :hourglass_flowing_sand: Planned |
| **Phase 11** | Self-Supervised Learning & Representation Analysis (CKA, Probes) | :hourglass_flowing_sand: Planned |
| **Phase 12** | Robustness Under Corruptions & Distribution Shifts | :hourglass_flowing_sand: Planned |
| **Phase 13** | Comparative Explainability (Attributions, Rollout, Grad-CAM) | :hourglass_flowing_sand: Planned |
| **Phase 14** | Downstream Dense Transfer (Detection & Segmentation) | :hourglass_flowing_sand: Planned |

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
│   │       ├── models/        # Linear, MLP, CNN models, conv2d, pooling, normalization
│   │       ├── training/      # Training engine, losses, SGD, LR schedulers, results
│   │       ├── evaluation/    # Evaluation engine, metrics, and structured reports
│   │       ├── representations/# Representation descriptors, feature batches, summaries
│   │       ├── robustness/    # Corruptions, distribution shifts, OOD tests
│   │       ├── explainability/# Saliency, attention rollout, Grad-CAM
│   │       ├── visualization/ # Projections (UMAP/t-SNE), figure generation
│   │       └── utils/         # Seeding, hashing, structured logging
│   └── tests/                 # Backend unit, smoke, and integration test suites
│
├── frontend/                  # Next.js / TypeScript research observatory
├── configs/                   # Declarative YAML configurations
├── experiments/               # Research artifacts and analyses
├── docs/                      # Technical and methodology documentation
├── data/                      # Local data stores (git-ignored raw data)
├── artifacts/                 # Generated run outputs (checkpoints, metrics, figures)
└── tests/                     # Top-level smoke, unit, and integration test suites
```

---

## Quickstart & Verification

Ensure you have [uv](https://docs.astral.sh/uv/) and [Node.js](https://nodejs.org/) installed:

```bash
# Clone the repository
git clone https://github.com/mzayan-bit/prism-visual-representations.git
cd prism-visual-representations

# Run the complete test suite
make test

# Run backend and frontend linting and type checks
make check
```

To run only the end-to-end smoke test suite:
```bash
uv run pytest tests/smoke/
```

---

## License

PRISM is released under the [MIT License](LICENSE).
