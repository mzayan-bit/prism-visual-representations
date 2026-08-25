# PRISM

### Probing the Evolution of Visual Representations

PRISM is an open-source, research-oriented computer vision platform designed to systematically investigate how different learning paradigms acquire, structure, and transfer visual representations.

> **Project Philosophy**  
> *One visual problem. Multiple learning paradigms. A deeper understanding of how machines learn to see.*

---

## Central Research Question

> **“How do different learning paradigms learn visual representations, generalize with limited supervision, fail under distribution shifts, and transfer to downstream vision tasks?”**

Rather than serving as another generic computer vision dashboard, model zoo, or pipeline orchestrator, PRISM is built around **fair, reproducible, and controlled scientific experiments**. Every model comparison is conducted under strictly matched dataset splits, identical preprocessing pipelines, and explicit compute budget allocations.

---

## Project Status

> [!NOTE]
> **Active Development**: PRISM is currently in the **Foundation Phase**. The core package boundaries, configuration registry, research contracts, testing harnesses, and monorepo scaffolding are initialized. Domain models, training loops, dataset manifests, and interactive analysis dashboards will be introduced in subsequent planned phases.

---

## High-Level Research Roadmap

The PRISM research campaign will systematically progress across learning paradigms and analytical dimensions:

```
Pixels and Linear Models
       ↓
Neural Networks & Backpropagation
       ↓
Optimization & Regularization
       ↓
Convolutional Architectures (CNNs)
       ↓
Vision Transformers & Attention
       ↓
Self-Supervised Learning (Contrastive / Masked)
       ↓
Representation Geometry & Manifold Analysis (CKA, Spectra, Probing)
       ↓
Robustness under Corruptions & Distribution Shifts
       ↓
Explainability across CNNs and Transformers
       ↓
Temporal & Video Understanding
       ↓
Generative-Data Experiments
       ↓
Dense Vision Transfer (Detection & Segmentation)
```

---

## Repository Architecture

PRISM is organized as a modular monorepo cleanly separating Python research components from configuration contracts, documentation, and the future web interface:

```
prism-visual-representations/
├── backend/                   # Python research engine and library package
│   ├── src/
│   │   └── prism/             # Core library
│   │       ├── api/           # Future API serving layer
│   │       ├── core/          # Base domain contracts, configuration schemas
│   │       ├── data/          # Dataset loaders, fingerprints, split manifests
│   │       ├── experiments/   # Experiment runner, provenance tracking
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
│   ├── experiments/           # End-to-end experiment recipes
│   ├── models/                # Architecture specifications
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

## Getting Started

### Prerequisites
- Python `>= 3.10`
- [`uv`](https://astral.sh/uv) (recommended) or standard Python `venv`
- Node.js `>= 18.0.0` (for frontend observatory)

### Backend Setup
```bash
# Clone the repository
git clone https://github.com/mzayan-bit/prism-visual-representations.git
cd prism-visual-representations

# Create a virtual environment using uv
uv venv --python 3.11 .venv
source .venv/bin/activate

# Install PRISM in editable mode with development dependencies
uv pip install -e ".[dev]"
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
The development landing page will be available at `http://localhost:3000`.

---

## Development & Testing Commands

The project includes a `Makefile` providing standard development workflows:

| Command | Action |
| --- | --- |
| `make install` | Install backend package in editable mode with dev dependencies |
| `make test` | Run pytest test suite across smoke, unit, and integration tests |
| `make lint` | Run Ruff linter checks |
| `make format` | Format code using Ruff |
| `make typecheck` | Run static type checking with Mypy |
| `make check` | Run all validation checks (lint, typecheck, test) |

To run specific test categories directly via `pytest`:
```bash
# Run smoke tests
pytest -m smoke

# Run unit tests
pytest -m unit
```

---

## Documentation Links

- [Architecture Overview](docs/architecture/overview.md) — Detailed monorepo design and subsystem specifications.
- [Research Contract](docs/methodology/research-contract.md) — Core scientific principles, reproducibility standards, and data policies.
- [Getting Started Guide](docs/development/getting-started.md) — Comprehensive installation and development instructions.
- [Repository Conventions](docs/development/repository-conventions.md) — Coding conventions, typing rules, and Git standards.
- [Contributing Guidelines](CONTRIBUTING.md) — How to contribute to PRISM.

---

## License

This project is licensed under the [MIT License](LICENSE).
