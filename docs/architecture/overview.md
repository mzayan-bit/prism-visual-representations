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

## Executable Dataset Pipeline & Deterministic Batching

Phase 5 introduces the executable dataset layer that turns controlled sample identities into machine-learning-ready examples, deterministic batches, and auditable runtime context.

```
ExperimentDefinition
        │
        ▼
Controlled Dataset References (Canonical Manifest, Partition Manifest, Subset Manifest)
        │
        ▼
Dataset Materializer (Exact Sample Resolution & Integrity Validation)
        │
        ▼
Executable Preprocessing (Deterministic resize, crop, and normalization transforms)
        │
        ▼
MaterializedDataset (In-memory indexed dataset preserving sample IDs)
        │
        ▼
Deterministic Ordering (Sequential / Fixed Shuffle / Epoch-Aware Shuffle)
        │ ── SHA-256 Ordering Fingerprint
        ▼
Deterministic Batch Loader (Batches preserving payload, labels, and sample_id traceability)
        │
        ▼
DataRuntimeContext (Auditable metadata context bound to execution)
```

### 1. `MaterializedSample` & `MaterializedDataset` (`prism.data.materialized`)
- **`MaterializedSample`**: Immutable runtime payload containing data (array/tensor), target label, source coordinates, and unique `sample_id`.
- **`MaterializedDataset`**: In-memory indexed dataset providing deterministic random access, slicing, and transform pipeline binding without mutating original manifests.

### 2. `DatasetMaterializer` (`prism.data.materializer`)
- Resolves requested sample IDs against provider adapters (`SyntheticVisionAdapter`, `CIFAR10Adapter`, `CIFAR100Adapter`).
- Enforces strict validation: source index, source split, and target labels must match manifest identities.

### 3. Deterministic Data Ordering (`prism.data.ordering`)
- **`OrderingStrategy`**:
  - `SEQUENTIAL`: Canonical manifest order (ideal for validation and test evaluation).
  - `FIXED_SHUFFLE`: Deterministic shuffle using a fixed base seed.
  - `EPOCH_AWARE_SHUFFLE`: Deterministic shuffle using combined `(seed, epoch)` arithmetic.
- **`compute_ordering_fingerprint(...)`**: Deterministic SHA-256 digest capturing exact sample sequence, strategy, seed, and epoch.

### 4. Batch Traceability (`prism.data.batching`)
- **`MaterializedBatch`**: Preserves batch index, batch size, inputs, targets, and the exact list of `sample_ids` contained in every batch.
- **`DeterministicBatchLoader`**: Pure Python / CPU-safe batch iterator with explicit `drop_last` behavior.

### 5. `DataPreparer` & `DataRuntimeContext` (`prism.data.preparer`, `prism.data.context`)
- Bridges `PreparedExecution` with physical data materialization.
- Produces immutable `DataRuntimeContext` storing canonical fingerprint, partition fingerprint, subset fingerprint, ordering fingerprint, batch size, and adapter metadata.

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

---

## Domain Subsystems

### `prism.core`
Defines system-wide primitives (`enums`, `identifiers`, `errors`, `metadata`).

### `prism.data`
Manifests, sample records, canonical universes, partition generators, nested subsets, dataset materialization, deterministic ordering, and batch loading.

### `prism.models`
Framework-neutral model descriptions (`ModelSpecification`) across CNNs, Transformers, and Self-Supervised backbones.

### `prism.training`
Validated optimization configurations (`TrainingConfiguration`, `OptimizerSpecification`, `SchedulerSpecification`, `GradientClipping`).

### `prism.evaluation`
Standardized evaluation protocols (`EvaluationConfiguration`, `MetricSpecification`, `EvaluationReport`).

### `prism.artifacts`
Artifact tracking contracts (`ArtifactReference`) storing logical keys, storage URIs, checksums, and generating run IDs.
