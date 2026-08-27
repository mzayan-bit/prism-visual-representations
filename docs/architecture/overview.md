# PRISM Architecture Overview

## Monorepo Layout & System Design

PRISM is architected as a modular research monorepo structured around clean domain boundaries.

```
prism-visual-representations/
├── backend/                   # Python research engine and package
│   ├── src/
│   │   └── prism/             # Core library package
│   │       ├── api/           # Future API serving layer
│   │       ├── artifacts/     # Artifact contracts and references
│   │       ├── core/          # Base enums, identifiers, errors, metadata
│   │       ├── data/          # Samples, universes, materialization, ordering, batching
│   │       ├── experiments/   # Experiment definitions, runs, harness, seeding, context
│   │       ├── models/        # Linear classifiers, initializations, vision model specs
│   │       ├── training/      # Training engine, losses, SGD optimizer, results
│   │       ├── evaluation/    # Evaluation engine, metrics, and structured reports
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

## Trainable Baseline & Training Engine Architecture

Phase 6 introduces the first end-to-end learning loop: a CS231N-style Linear Softmax Classifier optimized via SGD on deterministic batches.

```
ExperimentDefinition
        │
        ▼
Runtime Preparation (ExperimentExecutionHarness.prepare)
        │
        ▼
Materialized Dataset & Deterministic Batches (DataPreparer)
        │
        ▼
LinearSoftmaxClassifier (scores = xW + b)
        │
        ├── 1. Flatten Input: [B, C, H, W] -> [B, D]
        ├── 2. Forward Pass: Z = XW + b -> Raw Logits [B, num_classes]
        │
        ▼
SoftmaxCrossEntropyLoss (Numerically Stabilized)
        │
        ├── 3. Loss = - (1/B) * sum(log(P_yi)) + (1/2) * lambda * ||W||^2
        ├── 4. Analytic Gradient: dZ = (P - 1(y==c)) / B
        │
        ▼
Backward Pass & Optimization (SGDOptimizer)
        │
        ├── 5. Parameter Gradients: dW = X^T @ dZ + lambda * W, db = sum(dZ)
        ├── 6. Parameter Update: W <- W - lr * (mu * v_W + dW)
        │
        ▼
Evaluation Engine & Metrics Logging
        │
        ├── 7. MetricRecords logged into ExperimentRun
        ├── 8. EvaluationReport on Test Partition
        │
        ▼
TrainingResult & Completed Run Lifecycle
```

### 1. `LinearSoftmaxClassifier` (`prism.models.linear`)
- Multiclass linear model computing $Z = XW + b$.
- Flattens multidimensional image inputs into $[B, D]$ feature vectors.
- Initializes parameters deterministically via `initialize_linear_parameters`.

### 2. `SoftmaxCrossEntropyLoss` (`prism.training.loss`)
- Numerically stabilized cross-entropy loss consuming raw logits and target class indices.
- Computes exact analytic gradients $dZ$ for backpropagation without double-softmaxing.

### 3. `SGDOptimizer` (`prism.training.optimizers`)
- Stochastic Gradient Descent optimizer supporting momentum ($\mu$) and L2 weight decay ($\lambda$).
- Factory function `create_optimizer` binding from declarative `OptimizerSpecification`.

### 4. `TrainingEngine` & `EvaluationEngine` (`prism.training.engine`, `prism.evaluation.engine`)
- Manages complete `ExperimentRun` lifecycle (`PLANNED` -> `RUNNING` -> `COMPLETED` / `FAILED`).
- Enforces epoch-aware batch ordering per training epoch.
- Evaluates test and validation splits without gradient computation or weight modification.
- Returns immutable `TrainingResult` summary.

---

## Domain Subsystems

### `prism.core`
Defines system-wide primitives (`enums`, `identifiers`, `errors`, `metadata`).

### `prism.data`
Manifests, sample records, canonical universes, partition generators, nested subsets, dataset materialization, deterministic ordering, and batch loading.

### `prism.models`
Declarative model specifications (`ModelSpecification`), base vision model contract (`BaseVisionModel`), deterministic parameter initializations, and linear classifier baselines.

### `prism.training`
Training configurations (`TrainingConfiguration`), numerical losses (`SoftmaxCrossEntropyLoss`), optimizers (`SGDOptimizer`), training engine (`TrainingEngine`), and execution results (`TrainingResult`).

### `prism.evaluation`
Standardized evaluation protocols (`EvaluationConfiguration`, `MetricSpecification`), evaluation engine (`EvaluationEngine`), and structured reports (`EvaluationReport`).

### `prism.artifacts`
Artifact tracking contracts (`ArtifactReference`) storing logical keys, storage URIs, checksums, and generating run IDs.
