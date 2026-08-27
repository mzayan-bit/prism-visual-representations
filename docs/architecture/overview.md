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
│   │       ├── models/        # Linear, MLP, CNN models, conv2d, pooling, spatial utils
│   │       ├── training/      # Training engine, losses, SGD, LR schedulers, results
│   │       ├── evaluation/    # Evaluation engine, metrics, and structured reports
│   │       ├── representations/# Representation descriptors, feature batches, spatial maps
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

PRISM supports linear classifiers, deep MLPs, and Convolutional Neural Networks (CNNs) with explicit spatial representations, deterministic pooling, regularization, and learning-rate scheduling:

```
ExperimentDefinition (Linear / MLP / CNN Specification)
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
        ├── 3. Spatial Non-Linearities: A = Activation(Y_conv)
        ├── 4. Spatial Max Pooling: Y_pool = MaxPool2D(A, kernel, stride)
        ├── 5. Spatial Feature Maps: "conv_0", "pool_0", ..., "final_spatial" [N, C_last, H_last, W_last]
        ├── 6. Flatten to Vector: [N, D_spatial] where D_spatial = C_last * H_last * W_last
        └── 7. Classifier Head & Raw Logits: Z = H @ W_cls + b_cls [N, num_classes]
        │
        ▼
SoftmaxCrossEntropyLoss (Numerically Stabilized)
        │
        ├── 8. Loss = - (1/B) * sum(log(P_yi))
        └── 9. Analytic Gradient: dZ = (P - 1(y==c)) / B
        │
        ▼
Backward Pass & Optimization (SGDOptimizer + Scheduler)
        │
        ├── 10. Classifier backprop -> dH_spatial [N, D_spatial]
        ├── 11. Spatial unflatten -> dY_final_spatial [N, C_last, H_last, W_last]
        ├── 12. Pool argmax routing -> Activation derivative -> Conv2D weight/input gradients
        ├── 13. Scheduler Steps LR: lr(epoch) via Step or Cosine Annealing
        └── 14. Optimizer Step: W_l <- W_l - lr * (mu * v_W_l + dW_l + lambda * W_l)
        │
        ▼
Evaluation Engine & Metrics Logging
        │
        ├── 15. MetricRecords logged into ExperimentRun (loss, acc, lr)
        └── 16. EvaluationReport on Test Partition (Eval mode, dropout disabled)
        │
        ▼
TrainingResult & Completed Run Lifecycle
```

### 1. `Conv2D` (`prism.models.convolution`)
- Multi-channel 2D convolution layer operating on $[N, C_{\text{in}}, H_{\text{in}}, W_{\text{in}}]$ tensors.
- Supports configurable kernel sizes, strides, zero-padding, and optional bias.
- Full analytical backpropagation computing exact gradients w.r.t weights ($dW$), bias ($db$), and inputs ($dX$).

### 2. `MaxPool2D` & `AvgPool2D` (`prism.models.pooling`)
- `MaxPool2D`: Spatial maximum downsampling with exact argmax index tracking and gradient routing.
- `AvgPool2D`: Spatial average downsampling with uniform gradient distribution.

### 3. `ConvolutionalNeuralNetwork` (`prism.models.cnn`)
- Composable convolutional baseline stacking Conv2D $\to$ Activation $\to$ MaxPool blocks.
- Automatically calculates spatial dimension reduction and derives classifier input size.
- Tracks cumulative receptive field size and effective stride jump across stages.
- Exposes intermediate spatial feature maps (`"final_spatial"`) and final vector representations (`"final_hidden"`).

### 4. Parameter Initializations (`prism.models.initialization`)
- `initialize_linear_parameters`: Xavier Gaussian initialization.
- `initialize_mlp_parameters`: He/Kaiming normal for ReLU hidden layers and Xavier for output.
- `initialize_conv2d_parameters`: He/Kaiming normal scaled by receptive field fan-in ($C_{\text{in}} \times K_h \times K_w$).

### 5. Representation Extraction & Metadata (`prism.representations.contracts`)
- `RepresentationDescriptor` and `RepresentationBatch` supporting both 1D vector embeddings and 3D/4D spatial feature maps.
- Tracks `representation_kind` (`"vector"` vs `"spatial"`), `spatial_shape` $(C, H, W)$, and `receptive_field`.

### 6. Controlled Comparisons (`prism.experiments.comparisons`)
- `ControlledComparison` schema explicitly binding baseline and candidate experiments, isolated varied factors (Linear vs MLP vs CNN), invariant fixed factors, and deterministic SHA-256 fingerprints.

---

## Domain Subsystems

### `prism.core`
Defines system-wide primitives (`enums`, `identifiers`, `errors`, `metadata`).

### `prism.data`
Manifests, sample records, canonical universes, partition generators, nested subsets, dataset materialization, deterministic ordering, and batch loading.

### `prism.models`
Model specifications (`ModelSpecification`), base model interface (`BaseVisionModel`), linear classifiers, MLPs, CNNs, Conv2D, pooling, and parameter initializations.

### `prism.training`
Training configurations (`TrainingConfiguration`), numerical losses (`SoftmaxCrossEntropyLoss`), optimizers (`SGDOptimizer`), learning rate schedulers (`BaseLRScheduler`), training engine (`TrainingEngine`), and execution results (`TrainingResult`).

### `prism.evaluation`
Standardized evaluation protocols (`EvaluationConfiguration`, `MetricSpecification`), evaluation engine (`EvaluationEngine`), and structured reports (`EvaluationReport`).

### `prism.representations`
Representation extraction metadata contracts (`RepresentationDescriptor`, `RepresentationBatch`) supporting vector and spatial representations.

### `prism.experiments`
Declarative experiment definitions (`ExperimentDefinition`), run lifecycle tracking (`ExperimentRun`), runtime harness (`ExperimentExecutionHarness`), and controlled comparison contracts (`ControlledComparison`).

### `prism.artifacts`
Artifact tracking contracts (`ArtifactReference`) storing logical keys, storage URIs, checksums, and generating run IDs.
