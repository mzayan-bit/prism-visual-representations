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

PRISM supports linear classifiers, deep MLPs, and Convolutional Neural Networks (CNNs) with explicit spatial representations, deterministic pooling, batch normalization, regularization, and learning-rate scheduling:

```
ExperimentDefinition (Linear / MLP / CNN Specification + Normalization)
        │
        ▼
Runtime Preparation (ExperimentExecutionHarness.prepare)
        │
        ▼
Materialized Dataset & Deterministic Batches (DataPreparer)
        │
        ▼
Model Execution (LinearSoftmaxClassifier / MultiLayerPerceptron / ConvolutionalNeuralNetwork)
        │
        ├── 1. Spatial Processing: [N, C, H, W] preserved through Conv2D blocks
        ├── 2. Multi-Channel Convolutions: Y_conv = Conv2D(X, W, b, stride, pad)
        ├── 3. Spatial Batch Normalization: Y_norm = BatchNorm2D(Y_conv, gamma, beta)
        │      • Training mode: computes channel-wise batch mean & var; updates running stats
        │      • Evaluation mode: uses accumulated running mean & var without updates
        ├── 4. Spatial Non-Linearities: A = Activation(Y_norm)
        ├── 5. Spatial Max Pooling: Y_pool = MaxPool2D(A, kernel, stride)
        ├── 6. Feature Maps: "conv_0_pre_norm", "conv_0_post_norm", "conv_0", "final_spatial"
        ├── 7. Flatten to Vector: [N, D_spatial] where D_spatial = C_last * H_last * W_last
        └── 8. Classifier Head & Raw Logits: Z = H @ W_cls + b_cls [N, num_classes]
        │
        ▼
SoftmaxCrossEntropyLoss (Numerically Stabilized)
        │
        ├── 9. Loss = - (1/B) * sum(log(P_yi))
        └── 10. Analytic Gradient: dZ = (P - 1(y==c)) / B
        │
        ▼
Backward Pass & Optimization (SGDOptimizer + Scheduler)
        │
        ├── 11. Classifier backprop -> dH_spatial [N, D_spatial]
        ├── 12. Spatial unflatten -> dY_final_spatial [N, C_last, H_last, W_last]
        ├── 13. Pool argmax routing -> Activation derivative -> BatchNorm2D gradients (dX, dgamma, dbeta) -> Conv2D
        ├── 14. Scheduler Steps LR: lr(epoch) via Step or Cosine Annealing
        └── 15. Optimizer Step: updates trainable parameters (W, b, gamma, beta) only
        │
        ▼
Evaluation Engine & Metrics Logging
        │
        ├── 16. MetricRecords logged into ExperimentRun (loss, acc, lr)
        └── 17. EvaluationReport on Test Partition (Eval mode, dropout disabled, running stats used)
        │
        ▼
TrainingResult & Completed Run Lifecycle
```

### 1. `BatchNorm1D` & `BatchNorm2D` (`prism.models.normalization`)
- `BatchNorm1D`: Normalizes vector representations $[N, D]$ feature-wise.
- `BatchNorm2D`: Normalizes convolutional representations $[N, C, H, W]$ channel-wise across all $N \times H \times W$ spatial and batch positions.
- **Train vs Eval Mode Semantics**:
  - In training mode: computes current batch mean $\mu_B$ and variance $\sigma_B^2$, normalizes batch, updates non-trainable running statistics ($\text{running\_mean}$, $\text{running\_var}$) via exponential moving average with configurable `momentum`.
  - In evaluation mode: strictly freezes running statistics and normalizes inputs using stored running mean and variance.
- **Trainable Parameters vs State**:
  - Trainable parameters: affine scale $\gamma$ (init $1.0$) and shift $\beta$ (init $0.0$) updated by `SGDOptimizer`.
  - Non-trainable state: `running_mean` and `running_var` discovered via `get_state()` and never updated by optimizers.

### 2. `Conv2D` (`prism.models.convolution`)
- Multi-channel 2D convolution layer operating on $[N, C_{\text{in}}, H_{\text{in}}, W_{\text{in}}]$ tensors.
- Supports configurable kernel sizes, strides, zero-padding, and optional bias.
- Full analytical backpropagation computing exact gradients w.r.t weights ($dW$), bias ($db$), and inputs ($dX$).

### 3. `MaxPool2D` & `AvgPool2D` (`prism.models.pooling`)
- `MaxPool2D`: Spatial maximum downsampling with exact argmax index tracking and gradient routing.
- `AvgPool2D`: Spatial average downsampling with uniform gradient distribution.

### 4. `ConvolutionalNeuralNetwork` (`prism.models.cnn`)
- Composable convolutional baseline stacking $\text{Conv2D} \to [\text{BatchNorm2D}] \to \text{Activation} \to [\text{MaxPool2D}]$ blocks.
- Configurable normalization via `normalization: "batch_norm"`, `norm_eps`, `norm_momentum`, `norm_affine`.
- Exposes intermediate spatial feature maps (`"conv_0_pre_norm"`, `"conv_0_post_norm"`, `"final_spatial"`) and final representations.

### 5. `MultiLayerPerceptron` (`prism.models.mlp`)
- Deep MLP supporting $\text{Linear} \to [\text{BatchNorm1D}] \to \text{Activation} \to [\text{Dropout}]$.
- Exposes intermediate feature stages (`"hidden_0_pre_norm"`, `"hidden_0_post_norm"`, `"hidden_0"`, `"final_hidden"`).

### 6. Feature Distribution Summaries (`prism.representations.summaries`)
- `FeatureDistributionSummary`: Structured statistical contract recording mean, variance, standard deviation, minimum, maximum, zero fraction, finiteness status, sample count, tensor shape, and channel-wise statistics for 4D spatial tensors.
- `compute_distribution_summary`: Computes exact statistics over arbitrary representation tensors without mutating model state.
- `compare_distribution_summaries`: Measures stability shifts (mean shift, std shift, range delta, zero fraction delta) across layers or training regimes.

### 7. Controlled Comparisons (`prism.experiments.comparisons`)
- `ControlledComparison` schema and `create_normalization_comparison` helper isolating normalization factors (`normalization`, `norm_eps`, `norm_momentum`, `norm_affine`) while holding dataset, model width/depth, and optimization budgets invariant.

---

## Domain Subsystems

### `prism.core`
Defines system-wide primitives (`enums`, `identifiers`, `errors`, `metadata`).

### `prism.data`
Manifests, sample records, canonical universes, partition generators, nested subsets, dataset materialization, deterministic ordering, and batch loading.

### `prism.models`
Model specifications (`ModelSpecification`), base vision model contract (`BaseVisionModel`), linear classifiers, MLPs, CNNs, Conv2D, pooling, batch normalization (`BatchNorm1D`, `BatchNorm2D`), and parameter initializations.

### `prism.training`
Training configurations (`TrainingConfiguration`), numerical losses (`SoftmaxCrossEntropyLoss`), optimizers (`SGDOptimizer`), learning rate schedulers (`BaseLRScheduler`), training engine (`TrainingEngine`), and execution results (`TrainingResult`).

### `prism.evaluation`
Standardized evaluation protocols (`EvaluationConfiguration`, `MetricSpecification`), evaluation engine (`EvaluationEngine`), and structured reports (`EvaluationReport`).

### `prism.representations`
Representation descriptors (`RepresentationDescriptor`, `RepresentationBatch`), feature distribution summaries (`FeatureDistributionSummary`), and stability comparison utilities.

### `prism.experiments`
Declarative experiment definitions (`ExperimentDefinition`), run lifecycle tracking (`ExperimentRun`), runtime harness (`ExperimentExecutionHarness`), and controlled comparison contracts (`ControlledComparison`).

### `prism.artifacts`
Artifact tracking contracts (`ArtifactReference`) storing logical keys, storage URIs, checksums, and generating run IDs.
