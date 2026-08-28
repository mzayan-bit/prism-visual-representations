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
│   │       ├── models/        # Linear, MLP, CNN, ResNet, residual blocks, conv2d, pooling, norm
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

PRISM supports linear classifiers, deep MLPs, Convolutional Neural Networks (CNNs), and Deep Residual Neural Networks (ResNets) with explicit skip connections, dual-branch gradient propagation, deterministic pooling, batch normalization, regularization, and learning-rate scheduling:

```
ExperimentDefinition (Linear / MLP / CNN / ResNet Specification)
        │
        ▼
Runtime Preparation (ExperimentExecutionHarness.prepare)
        │
        ▼
Materialized Dataset & Deterministic Batches (DataPreparer)
        │
        ▼
Model Execution (ResidualNeuralNetwork / ConvolutionalNeuralNetwork)
        │
        ├── 1. Stem Processing: Conv2D -> BatchNorm2D -> Activation [N, C_stem, H, W]
        ├── 2. Residual Stages: Stacks of composable ResidualBlock instances
        │      • Main Path F(x): Conv2D -> BatchNorm2D -> Act -> Conv2D -> BatchNorm2D
        │      • Shortcut Path S(x): Identity (S(x)=x) or Projection (Conv2D 1x1 + BatchNorm2D)
        │      • Explicit Residual Addition: Z = F(x) + S(x) (strict shape validation)
        │      • Post-Addition Non-Linearity: Y = Activation(Z)
        ├── 3. Feature Maps: "stem", "stage_s_block_b_residual", "stage_s_block_b_shortcut", "stage_s_block_b_post_add"
        ├── 4. Flatten to Vector: [N, D_spatial] where D_spatial = C_last * H_last * W_last
        └── 5. Classifier Head & Raw Logits: Z = H @ W_cls + b_cls [N, num_classes]
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
        ├── 8. Classifier backprop -> dH_spatial [N, D_spatial]
        ├── 9. Spatial unflatten -> dY_final_spatial [N, C_last, H_last, W_last]
        ├── 10. Residual Addition Backprop: routes upstream dZ to BOTH branches (dF = dZ, dS = dZ)
        │       • Summed Input Gradient: dX = dX_main + dX_shortcut
        │       • Accumulates parameter gradients in conv1, conv2, norm1, norm2, and proj_conv
        ├── 11. Scheduler Steps LR: lr(epoch) via Step or Cosine Annealing
        └── 12. Optimizer Step: updates trainable parameters (W, b, gamma, beta) only
        │
        ▼
Gradient Flow Tracking & Evaluation Engine
        │
        ├── 13. ParameterGradientSummary & ModelGradientFlowSummary across depth
        ├── 14. MetricRecords logged into ExperimentRun (loss, acc, lr, grad_norm)
        └── 15. EvaluationReport on Test Partition (Eval mode, dropout disabled, running stats used)
        │
        ▼
TrainingResult & Completed Run Lifecycle
```

### 1. Residual Learning & Skip Connections (`prism.models.residual`)
- `ResidualAdd`: Explicit tensor addition node validating compatible dimensions $[N, C, H, W]$. Upstream gradient routes identically to both branches ($dA = dZ, dB = dZ$).
- `IdentityShortcut`: Parameter-free skip connection for shape-preserving residual blocks ($S(x) = x$).
- `ProjectionShortcut`: $1\times 1$ convolution with optional `BatchNorm2D` for spatial downsampling and channel expansion ($S(x) = \text{Norm}(\text{Conv2D}(x, 1\times 1, \text{stride}=s))$).
- `ResidualBlock`: Composable 2-convolution residual block ($y = \text{act}(F(x) + S(x))$) with exact analytical backpropagation through both branches.

### 2. `ResidualNeuralNetwork` (`prism.models.resnet`)
- Multi-stage residual vision model composed of stem, configurable stages and blocks per stage, spatial flattening/pooling, and classifier heads.
- Exposes intermediate representations: `"stem"`, `"stage_{s}_block_{b}_residual"`, `"stage_{s}_block_{b}_shortcut"`, `"stage_{s}_block_{b}_post_add"`, `"stage_{s}_block_{b}"`, `"final_spatial"`, `"final_hidden"`, `"logits"`.

### 3. Gradient Flow Research Infrastructure (`prism.training.gradient_flow`)
- `ParameterGradientSummary`: Captures layer identifier, logical stage, relative depth index, L2 norm, mean, std dev, min, max, zero fraction, finiteness status, and tensor dimensions.
- `ModelGradientFlowSummary`: Structured container tracking depth-ordered parameter gradient summaries and global gradient norms.
- `compute_gradient_flow_summary`: Non-mutating gradient extractor scanning model parameters and gradients.
- `compare_gradient_flow_summaries`: Measures global norm ratios, norm deltas, early-to-late gradient ratios, and layer-by-layer gradient shifts.

### 4. `BatchNorm1D` & `BatchNorm2D` (`prism.models.normalization`)
- `BatchNorm1D`: Normalizes vector representations $[N, D]$ feature-wise.
- `BatchNorm2D`: Normalizes convolutional representations $[N, C, H, W]$ channel-wise across all $N \times H \times W$ spatial and batch positions.
- **Train vs Eval Mode Semantics**:
  - In training mode: computes current batch mean $\mu_B$ and variance $\sigma_B^2$, normalizes batch, updates non-trainable running statistics ($\text{running\_mean}$, $\text{running\_var}$) via exponential moving average with configurable `momentum`.
  - In evaluation mode: strictly freezes running statistics and normalizes inputs using stored running mean and variance.
- **Trainable Parameters vs State**:
  - Trainable parameters: affine scale $\gamma$ (init $1.0$) and shift $\beta$ (init $0.0$) updated by `SGDOptimizer`.
  - Non-trainable state: `running_mean` and `running_var` discovered via `get_state()` and never updated by optimizers.

### 5. `Conv2D` & Pooling (`prism.models.convolution`, `prism.models.pooling`)
- `Conv2D`: Multi-channel 2D convolution layer operating on $[N, C_{\text{in}}, H_{\text{in}}, W_{\text{in}}]$ tensors with full analytical backpropagation.
- `MaxPool2D` & `AvgPool2D`: Spatial pooling with exact argmax index tracking or uniform gradient routing.

### 6. Controlled Comparisons (`prism.experiments.comparisons`)
- `ControlledComparison` schema, `create_normalization_comparison`, and `create_residual_comparison` helpers isolating architectural factors (`model_family`, `architecture`, `has_skip_connections`, `shortcut_type`) while holding dataset, model width/depth, and optimization budgets invariant.

---

## Domain Subsystems

### `prism.core`
Defines system-wide primitives (`enums`, `identifiers`, `errors`, `metadata`).

### `prism.data`
Manifests, sample records, canonical universes, partition generators, nested subsets, dataset materialization, deterministic ordering, and batch loading.

### `prism.models`
Model specifications (`ModelSpecification`), base vision model contract (`BaseVisionModel`), linear classifiers, MLPs, CNNs, ResNets, Conv2D, pooling, batch normalization (`BatchNorm1D`, `BatchNorm2D`), residual blocks (`ResidualBlock`), and parameter initializations.

### `prism.training`
Training configurations (`TrainingConfiguration`), numerical losses (`SoftmaxCrossEntropyLoss`), optimizers (`SGDOptimizer`), learning rate schedulers (`BaseLRScheduler`), gradient flow tracking (`ModelGradientFlowSummary`), training engine (`TrainingEngine`), and execution results (`TrainingResult`).

### `prism.evaluation`
Standardized evaluation protocols (`EvaluationConfiguration`, `MetricSpecification`), evaluation engine (`EvaluationEngine`), and structured reports (`EvaluationReport`).

### `prism.representations`
Representation descriptors (`RepresentationDescriptor`, `RepresentationBatch`), feature distribution summaries (`FeatureDistributionSummary`), and stability comparison utilities.

### `prism.experiments`
Declarative experiment definitions (`ExperimentDefinition`), run lifecycle tracking (`ExperimentRun`), runtime harness (`ExperimentExecutionHarness`), and controlled comparison contracts (`ControlledComparison`).

### `prism.artifacts`
Artifact tracking contracts (`ArtifactReference`) storing logical keys, storage URIs, checksums, and generating run IDs.
