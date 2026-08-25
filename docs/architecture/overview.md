# PRISM Architecture Overview

## Monorepo Layout & System Design

PRISM is architected as a modular research monorepo structured around clean domain boundaries.

```
prism-visual-representations/
├── backend/                   # Python research engine and package
│   ├── src/
│   │   └── prism/             # Core library package
│   │       ├── api/           # Future API serving layer
│   │       ├── core/          # Base contracts, configuration schemas, primitives
│   │       ├── data/          # Dataset loaders, fingerprints, split manifests
│   │       ├── experiments/   # Experiment harness, runner, provenance tracking
│   │       ├── models/        # Vision backbones, probe heads, model registry
│   │       ├── training/      # Deterministic training loops and optimization
│   │       ├── evaluation/    # Metric evaluation, calibration, benchmarks
│   │       ├── representations/# CKA, linear probing, singular value spectra
│   │       ├── robustness/    # Corruptions, distribution shifts, OOD tests
│   │       ├── explainability/# Saliency, attention rollout, Grad-CAM
│   │       ├── visualization/ # Projections (UMAP/t-SNE), figure generation
│   │       └── utils/         # Seeding, hashing, structured logging
│   └── tests/                 # Backend unit and module test suites
│
├── frontend/                  # Next.js / TypeScript research observatory
│   ├── app/                   # App Router pages and layout
│   └── ...
│
├── configs/                   # Declarative YAML configurations
│   ├── base/                  # Runtime and environment defaults
│   ├── datasets/              # Dataset and preprocessing configs
│   ├── experiments/           # End-to-end experiment definitions
│   ├── models/                # Architecture configurations
│   └── training/              # Optimization schedules and budgets
│
├── experiments/               # Research artifacts and analyses
│   ├── notebooks/             # Exploratory Jupyter notebooks
│   └── reports/               # Synthesized research findings
│
├── docs/                      # Technical and methodology documentation
│   ├── architecture/          # Architecture blueprints
│   ├── methodology/           # Research contracts and fairness standards
│   ├── experiments/           # Campaign notes
│   └── development/           # Setup and developer guides
│
├── data/                      # Local data stores (git-ignored raw data)
├── artifacts/                 # Generated run outputs (checkpoints, metrics, figures)
└── tests/                     # Top-level smoke, unit, and integration test suites
```

---

## Domain Subsystems

### 1. `prism.core`
Defines base interfaces and Pydantic configuration schemas that enforce strict validation across all modules.

### 2. `prism.data`
Manages deterministic dataset pipelines. Enforces cryptographic dataset fingerprints to guarantee test set sanctity and identical preprocessing across comparative baselines.

### 3. `prism.models`
Provides unified model wrappers for diverse visual representation learners. Models implement a standardized interface exposing both final task outputs and intermediate activation representations.

### 4. `prism.training`
Encapsulates reproducible training loops, strict multi-device seed management, and optimizer state lifecycle management.

### 5. `prism.representations`
Hosts mathematical probing algorithms including:
- Centered Kernel Alignment (CKA) for layer-to-layer representation similarity.
- Linear and non-linear diagnostic probes.
- Singular value spectrum analysis and intrinsic dimension estimators.

### 6. `prism.robustness`
Evaluates representation stability under synthetic corruptions (noise, blur, weather), natural distribution shifts, and adversarial perturbations.

### 7. `prism.explainability`
Provides comparative visual attribution (Grad-CAM, attention rollout, integrated gradients) to analyze spatial inductive biases across convolutional and attention-based architectures.

### 8. `prism.visualization`
Generates structured figure outputs, publication-ready vector charts, and 2D/3D embedding coordinates (UMAP/t-SNE/PCA) for ingestion by the frontend research observatory.

### 9. `prism.api` & `frontend`
Reserved programmatic server layer and Next.js research observatory providing interactive exploration of representation geometries, metrics, and explainability maps.
