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
│   │       ├── data/          # Samples, universes, partitions, subsets, adapters
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

## Controlled Dataset Architecture & Data Identity

PRISM separates declarative dataset descriptions from exact sample universes, fixed partitions, and low-data subsets to guarantee scientifically fair comparisons across learning paradigms.

```
1. Dataset Source (e.g. CIFAR-10 / CIFAR-100)
        │
        ▼
2. CanonicalSampleManifest (Ordered universe of all SampleRecords)
        │ ── SHA-256 Digest
        ▼
3. PartitionManifest (Deterministic split assignment: Train / Val / Isolated Test)
        │ ── SHA-256 Digest
        ▼
4. SubsetManifest (Strictly nested data budgets: 1% ⊆ 5% ⊆ 10% ⊆ 25% ⊆ 50% ⊆ 100%)
        │ ── SHA-256 Digest
        ▼
5. ControlledDataReference ──► Bound to ExperimentDefinition
```

### 1. `SampleRecord` & `CanonicalSampleManifest` (`prism.data.samples`)
- **Stable Identity**: Sample identity is constructed from canonical coordinates (e.g. `cifar10/train/000042`) rather than runtime memory addresses.
- **Canonical Sample Universe**: Immutable manifest capturing every available sample in a deterministic order with verified counts and category labels.

### 2. `PartitionManifest` & Partition Generator (`prism.data.partitions`)
- **Fixed Partitions**: Mutually exclusive mapping of canonical samples into named splits (`train`, `val`, `test`).
- **Deterministic Stratification**: Stratified splitting using a local RNG (`random.Random(seed)`) without touching global random state.
- **Official Test Split Isolation**: Official test sets are kept strictly isolated and untouched.

### 3. `SubsetManifest` & Nested Subset Generator (`prism.data.subsets`)
- **Strict Mathematical Nesting**: Guarantees $S_{1\%} \subseteq S_{5\%} \subseteq S_{10\%} \subseteq S_{25\%} \subseteq S_{50\%} \subseteq S_{100\%} = \text{TrainSplit}$.
- **Data-Efficiency Integrity**: Avoids independent random sampling across data budgets, ensuring that low-data regimes are genuine nested subsets of higher budgets.

### 4. Benchmark Dataset Adapters (`prism.data.adapters`)
- **`CIFAR10Adapter` & `CIFAR100Adapter`**: Standardized adapters producing canonical sample universes (60k samples), benchmark partitions (45k train, 5k val, 10k isolated test), and nested subsets.
- **Optional Dependency Isolation**: Core data manifest and fingerprint generation operates with zero external dependencies (no PyTorch/torchvision required). Raw loading provides explicit, guarded calls.

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
Controlled dataset manifests, sample records, canonical universes, partition generators, nested subsets, and benchmark adapters.

### `prism.models`
Framework-neutral model descriptions (`ModelSpecification`) across CNNs, Transformers, and Self-Supervised backbones.

### `prism.training`
Validated optimization configurations (`TrainingConfiguration`, `OptimizerSpecification`, `SchedulerSpecification`, `GradientClipping`).

### `prism.evaluation`
Standardized evaluation protocols (`EvaluationConfiguration`, `MetricSpecification`, `EvaluationReport`).

### `prism.artifacts`
Artifact tracking contracts (`ArtifactReference`) storing logical keys, storage URIs, checksums, and generating run IDs.
