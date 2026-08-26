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
│   │       ├── data/          # Dataset manifests, splits, preprocessing policies
│   │       ├── experiments/   # Experiment definitions, runs, harness, seeding, context
│   │       ├── models/        # Vision model specifications and registries
│   │       ├── training/      # Training configurations and optimizer policies
│   │       ├── evaluation/    # Evaluation configurations and structured reports
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

## Reproducibility Runtime & Experiment Harness

The PRISM Reproducibility Runtime bridges immutable experiment definitions and physical execution by preparing, probing, and auditing the runtime environment before any ML workload starts.

```
ExperimentDefinition (Immutable Scientific Intent)
        │
        ├── Validate Task & Model Compatibility
        │
        ▼
ExperimentExecutionHarness.prepare(experiment)
        │
        ├── 1. Compute SHA-256 Configuration Fingerprint
        ├── 2. Inspect Git Provenance (commit, branch, dirty tracking)
        ├── 3. Probe Hardware & Compute Backends (CPU, CUDA, Apple Silicon MPS)
        ├── 4. Capture Environment Snapshot (Python version, OS, allowlisted packages)
        ├── 5. Initialize Multi-Backend RNG (Python, NumPy, PyTorch CPU/CUDA/MPS)
        └── 6. Bind Provenance & Environment to ExperimentRun (PLANNED)
        │
        ▼
PreparedExecution / RuntimeContext
(Immutable audit trail ready for future training engines)
```

### 1. `ExperimentExecutionHarness` (`prism.experiments.harness`)
Validates an `ExperimentDefinition`, inspects host capabilities, applies seeding, binds metadata to an `ExperimentRun`, and outputs an immutable `PreparedExecution` runtime context. The harness stops before workload execution (no training loops, dataloaders, or tensor allocations).

### 2. `PreparedExecution` / `RuntimeContext` (`prism.experiments.context`)
An immutable (`frozen=True`) execution snapshot linking:
- Experiment ID, Run ID, and SHA-256 Configuration Fingerprint.
- Multi-backend RNG seed initialization report (`SeedInitializationResult`).
- Structured host environment metadata (`EnvironmentMetadata`).
- Discovered hardware acceleration capabilities (`HardwareMetadata`).
- Source code version control state (`CodeRevisionMetadata`).
- Transparent reproducibility capability facts (`get_reproducibility_report()`).

### 3. Multi-Backend Deterministic Seeding (`prism.experiments.seeding`)
Centrally manages random state across:
- **Python standard library**: `random.seed(seed)` and `os.environ["PYTHONHASHSEED"]`.
- **NumPy**: `numpy.random.seed(seed)` when NumPy is installed.
- **PyTorch**: `torch.manual_seed(seed)`, `torch.cuda.manual_seed_all(seed)`, cuDNN determinism flags, and `torch.use_deterministic_algorithms(True)`.
- **Graceful Hardware Fallback**: Transparently records limitations on CPU-only, CUDA, and Apple Silicon MPS platforms without crashing.

### 4. Git Provenance Inspector (`prism.experiments.provenance`)
Discovers commit SHA, active branch, remote repository URL, and working tree cleanliness (`-uno` to inspect tracked modifications only, avoiding arbitrary file indexing).

---

## Domain Subsystems

### `prism.core`
Defines system-wide primitives:
- **`enums`**: `TaskType`, `RunStatus`, `ModelFamily`, `InitializationStrategy`, `ArtifactType`, `MetricDirection`, `PrecisionMode`, `DevicePreference`, `SplitName`.
- **`identifiers`**: Centralized generation and validation of alphanumeric prefixed IDs.
- **`errors`**: Domain exception hierarchy rooted at `PrismError` (`ConfigurationError`, `ValidationError`, `LifecycleError`, `InvalidTransitionError`, `SerializationError`, `FingerprintError`, `ReproducibilityError`, `RuntimeInitializationError`, `ProvenanceError`).
- **`metadata`**: Provenance schemas (`CreationMetadata`, `CodeRevisionMetadata`, `EnvironmentMetadata`, `HardwareMetadata`).

### `prism.data`
Declarative dataset manifest models (`DatasetManifest`, `SplitSpecification`, `PreprocessingPolicy`, `AugmentationPolicy`) describing dataset dimensions and splits without instantiating tensors in memory.

### `prism.models`
Framework-neutral model descriptions (`ModelSpecification`) capturing architectures, parameter configurations, and probe attachments across CNNs, Transformers, and Self-Supervised backbones.

### `prism.training`
Validated optimization configurations (`TrainingConfiguration`, `OptimizerSpecification`, `SchedulerSpecification`, `GradientClipping`, `EarlyStoppingPolicy`).

### `prism.evaluation`
Standardized evaluation protocols (`EvaluationConfiguration`, `MetricSpecification`, `EvaluationReport`).

### `prism.artifacts`
Artifact tracking contracts (`ArtifactReference`) storing logical keys, storage URIs, checksums, and generating run IDs.
