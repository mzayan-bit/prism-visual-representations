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
│   │       ├── experiments/   # Definitions, runs, harness, seeding, comparisons
│   │       ├── models/        # Linear models, MLPs, activations, specifications
│   │       ├── training/      # Training engine, losses, SGD, LR schedulers, results
│   │       ├── evaluation/    # Evaluation engine, metrics, and structured reports
│   │       ├── representations/# Representation descriptors, feature batches
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

## Trainable Baseline & Deep Learning Architecture

PRISM supports both linear classifiers and deep non-linear Multi-Layer Perceptrons (MLPs) with explicit regularization, deterministic dropout, and learning-rate scheduling:

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
Model Execution (LinearSoftmaxClassifier / MultiLayerPerceptron)
        │
        ├── 1. Flatten Input: [B, C, H, W] -> [B, D]
        ├── 2. Hidden Layers: H_l+1 = Activation(H_l @ W_l + b_l)
        ├── 3. Deterministic Dropout (Training Mode Only): H_l+1 * Mask / (1 - p)
        ├── 4. Representation Extraction: "input_flat", "hidden_l", "final_hidden"
        └── 5. Output Layer: Logits Z = H_final @ W_out + b_out
        │
        ▼
SoftmaxCrossEntropyLoss (Numerically Stabilized)
        │
        ├── 6. Loss = - (1/B) * sum(log(P_yi))
        └── 7. Analytic Gradient: dZ = (P - 1(y==c)) / B
        │
        ▼
Backward Pass & Optimization (SGDOptimizer + Scheduler)
        │
        ├── 8. Backpropagation across all hidden layers & activations
        ├── 9. Scheduler Steps LR: lr(epoch) via Step or Cosine Annealing
        └── 10. Optimizer Step: W_l <- W_l - lr * (mu * v_W + dW_l + lambda * W_l)
        │
        ▼
Evaluation Engine & Metrics Logging
        │
        ├── 11. MetricRecords logged into ExperimentRun (loss, acc, lr)
        └── 12. EvaluationReport on Test Partition (Eval mode, dropout disabled)
        │
        ▼
TrainingResult & Completed Run Lifecycle
```

### 1. `MultiLayerPerceptron` (`prism.models.mlp`)
- Configurable hidden layers (e.g. `[128]`, `[512, 256]`, `[1024, 512, 256]`).
- Supports non-linear activation functions (`ReLU`, `GELU`).
- Supports deterministic inverted dropout ($0.0 \le p < 1.0$) with explicit per-step RNG seeding.
- Full analytical backpropagation propagating gradients through output, activations, dropout masks, and hidden layers.

### 2. `BaseActivation` & Analytical Derivatives (`prism.models.activations`)
- `ReLUActivation`: $f(x) = \max(0, x)$, $f'(x) = 1(x > 0)$.
- `GELUActivation`: Smooth Gaussian Error Linear Unit with exact analytic derivative.

### 3. Parameter Initializations (`prism.models.initialization`)
- `initialize_linear_parameters`: Xavier Gaussian initialization with zero biases.
- `initialize_mlp_parameters`: He/Kaiming normal initialization for ReLU hidden layers and Xavier initialization for output layers.

### 4. Learning Rate Schedulers (`prism.training.schedulers`)
- `ConstantLRScheduler`: Maintains constant learning rate across training.
- `StepLRScheduler`: Decays learning rate by factor $\gamma$ every `step_size` epochs.
- `CosineAnnealingLRScheduler`: Cosine decay from `base_lr` to `min_lr` over total epochs.
- Linear warmup support across schedulers.

### 5. Representation Extraction (`prism.representations.contracts`)
- Models expose `extract_representations(inputs, layer="final_hidden")`.
- Supports standardized layer handles: `"input_flat"`, `"hidden_0"`, `"hidden_1"`, ..., `"final_hidden"`, `"logits"`.
- `RepresentationDescriptor` and `RepresentationBatch` metadata containers.

### 6. Controlled Comparisons (`prism.experiments.comparisons`)
- `ControlledComparison` schema explicitly binding baseline and candidate experiments, isolated varied factors, invariant fixed factors, and deterministic fingerprints.

---

## Domain Subsystems

### `prism.core`
Defines system-wide primitives (`enums`, `identifiers`, `errors`, `metadata`).

### `prism.data`
Manifests, sample records, canonical universes, partition generators, nested subsets, dataset materialization, deterministic ordering, and batch loading.

### `prism.models`
Declarative model specifications (`ModelSpecification`), base vision model contract (`BaseVisionModel`), linear classifiers, deep MLPs, activations, and parameter initializations.

### `prism.training`
Training configurations (`TrainingConfiguration`), numerical losses (`SoftmaxCrossEntropyLoss`), optimizers (`SGDOptimizer`), learning rate schedulers (`BaseLRScheduler`), training engine (`TrainingEngine`), and execution results (`TrainingResult`).

### `prism.evaluation`
Standardized evaluation protocols (`EvaluationConfiguration`, `MetricSpecification`), evaluation engine (`EvaluationEngine`), and structured reports (`EvaluationReport`).

### `prism.representations`
Representation extraction metadata contracts (`RepresentationDescriptor`, `RepresentationBatch`) and representation analysis tools.

### `prism.experiments`
Declarative experiment definitions (`ExperimentDefinition`), run lifecycle tracking (`ExperimentRun`), runtime harness (`ExperimentExecutionHarness`), and controlled comparison contracts (`ControlledComparison`).

### `prism.artifacts`
Artifact tracking contracts (`ArtifactReference`) storing logical keys, storage URIs, checksums, and generating run IDs.
