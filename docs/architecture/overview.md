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
│   │       ├── experiments/   # Experiment definitions, runs, lifecycle, metrics, hashing
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

## Research Core Domain Contracts

The PRISM Research Core establishes a strongly-typed, immutable, framework-neutral domain layer that enforces reproducibility, task compatibility, and provenance tracking across all visual learning paradigms.

### 1. `ExperimentDefinition` (`prism.experiments.definitions`)
An immutable (`frozen=True`) specification representing the scientific intent of an experiment prior to execution:
- **Identifier & Task**: Unique experiment ID (e.g. `exp-cifar10-resnet18`), task paradigm (`TaskType`), hypothesis, and tags.
- **Dataset Contract**: Declares a `DatasetManifest` with preprocessing, augmentation, and split partitions.
- **Model Contract**: Declares a `ModelSpecification` with architectural family, initialization, and input dimensions.
- **Training Contract**: Declares a `TrainingConfiguration` with epochs, batch sizes, optimizer, scheduler, and precision.
- **Evaluation Contract**: Declares an `EvaluationConfiguration` with target splits, metrics, and thresholds.
- **Reproducibility Settings**: Declares master seeds, determinism flags, and audit requirements.
- **Semantic Fingerprinting**: Provides `compute_fingerprint()` which computes a deterministic SHA-256 hash of all semantic inputs.

### 2. `ExperimentRun` (`prism.experiments.runs`)
Represents an individual physical execution attempt of an `ExperimentDefinition`:
- **Run Identity**: Unique run ID (e.g. `run-a1b2c3d4e5f6`) linked back to parent `experiment_id`.
- **Lifecycle State Machine**: Enforces strict valid state transitions:
  - `PLANNED` → `QUEUED` / `RUNNING` / `CANCELLED`
  - `QUEUED` → `RUNNING` / `CANCELLED`
  - `RUNNING` → `COMPLETED` / `FAILED` / `CANCELLED`
  - Terminal states (`COMPLETED`, `FAILED`, `CANCELLED`) cannot be re-executed.
- **Provenance Snapshot**: Records configuration fingerprint, runtime environment, code revision, and failure telemetry.
- **Telemetry Log**: Collects scalar `MetricRecord` entries and registers output `ArtifactReference` handles.

### 3. `EvaluationReport` (`prism.evaluation.reports`)
An immutable evaluation summary linking:
- Run and experiment identifiers.
- Complete evaluation configuration.
- Detailed scalar metric records across splits.
- Generated artifact references (e.g. confusion matrices, UMAP projections).
- High-level summary metrics.

---

## Domain Subsystems

### `prism.core`
Defines system-wide primitives:
- **`enums`**: `TaskType`, `RunStatus`, `ModelFamily`, `InitializationStrategy`, `ArtifactType`, `MetricDirection`, `PrecisionMode`, `DevicePreference`, `SplitName`.
- **`identifiers`**: Centralized generation and validation of alphanumeric prefixed IDs.
- **`errors`**: Domain exception hierarchy (`PrismError`, `ConfigurationError`, `ValidationError`, `InvalidTransitionError`, `SerializationError`, `FingerprintError`).
- **`metadata`**: Provenance schemas (`CreationMetadata`, `CodeRevisionMetadata`, `EnvironmentMetadata`).

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
