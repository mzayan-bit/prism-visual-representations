# PRISM Demonstration Campaign Guide

This guide documents the official PRISM demonstration campaign: **PRISM Representation Showcase** (`prism_representation_showcase`).

---

## Overview

The PRISM Demonstration Campaign provides a complete, deterministic, evidence-grounded dataset designed to showcase all 11 PRISM laboratories and the top-level Benchmark Observatory without requiring hours of compute.

- **Campaign Identifier**: `prism_representation_showcase`
- **Total Observed Cells**: 810 canonical benchmark metric evaluations
- **Artifacts Location**: `artifacts/demo/`
- **Runtime**: < 3 seconds to generate deterministically

---

## Campaign Matrix Design

The demonstration campaign spans a balanced grid across architectures, pretraining objectives, evaluation tasks, and random seeds:

| Experimental Factor | Evaluated Values | Count |
| :--- | :--- | :--- |
| **Model Architectures** | Convolutional Neural Network (`cnn`), Deep ResNet (`resnet`), Vision Transformer (`vit`) | 3 |
| **Pretraining Objectives** | Supervised Cross-Entropy (`supervised`), Contrastive SSL (`simclr`), Masked Reconstruction (`reconstruction`), Vision-Language Dual-Encoder (`multimodal`), Unpretrained Baseline (`scratch`) | 5 |
| **Random Seeds** | `42`, `100`, `2024` | 3 |
| **Evaluation Domains** | 10 canonical research axes (Classification, Geometry, Robustness, Explainability, Transfer, Spatial, Temporal, Multimodal, Calibration, OOD) | 10 |

---

## Generated Artifacts

Executing the demo generation workflow produces five core artifacts in `artifacts/demo/`:

1. **`prism_demo_campaign.json`**: Complete serialized `BenchmarkCampaign` containing all 810 `BenchmarkResultCell` instances with full metadata, sample counts, and cryptographic provenance fingerprints.
2. **`prism_demo_report.json`**: Serialized `ResearchReport` containing cross-architecture syntheses, cross-objective syntheses, 10-dimensional representation profiles, Pareto frontiers, grounded findings, and evidence gap summaries.
3. **`prism_demo_report.md`**: Formal publication-ready Markdown research report with executive summary, benchmark tables, Pareto tradeoff analysis, and reproducibility manifest.
4. **`benchmark_matrix.csv`**: Structured CSV matrix tabulating every model-objective pair across all canonical PRISM metrics.
5. **`benchmark_table.csv`**: Raw flat CSV record table for external statistical analysis in R, Pandas, or Excel.

---

## Generating & Validating Demo Artifacts

### Quick Generation via Make
```bash
# Generate official demo artifacts into artifacts/demo/
make demo
```

### Direct Script Usage
The demo generation utility (`scripts/generate_demo.py`) provides fine-grained control:

```bash
# Generate demo artifacts with default seed (42)
python scripts/generate_demo.py

# Verify existing demo artifacts without modifying them
python scripts/generate_demo.py --check

# Dry-run generation to verify pipeline without writing files
python scripts/generate_demo.py --dry-run

# Output artifacts to a custom directory with a custom seed
python scripts/generate_demo.py --output-dir /path/to/artifacts --seed 1234
```

---

## Exploring the Demo in the Research Observatory

To explore the generated demonstration campaign visually:

1. Launch the Next.js development server:
   ```bash
   make dev
   ```
2. Open `http://localhost:3000` in a modern browser.
3. Explore the 5 research domains via the top navigation bar:
   - **Synthesis**: Benchmark Observatory with live matrix explorer, Pareto tradeoff scatter, multi-dimensional radar profiles, evidence gap planner, and report generator.
   - **Representation**: Geometry Observatory (PCA manifolds, $k$-NN consistency), Robustness Laboratory (paired corruption drift vectors), and Explainability Laboratory (saliency and Grad-CAM maps).
   - **Learning Paradigms**: Transfer Learning (probe separability), Self-Supervised Learning (SimCLR collapse metrics), and Reconstruction (masked patch recovery).
   - **Downstream Probes**: Spatial Transfer (bounding box IoU, segmentation mIoU), Temporal Laboratory (video consistency, motion correlation), and Multimodal Laboratory (bidirectional text/image retrieval).
   - **Reliability & Calibration**: Uncertainty Laboratory (reliability diagrams, ECE, temperature scaling, OOD AUROC).

---

## Data Provenance & Scientific Disclosure

- **Deterministic Generation**: All metrics are generated using deterministic mathematical models matching empirical PRISM training behavior under fixed seeds.
- **No Manual Metric Fabrication**: No numbers in the demonstration suite are hardcoded or manually typed. Every metric originates from verified functional adapters.
- **Controlled Synthetic Labeling**: All demo metrics are explicitly tagged with `is_synthetic: true` and display subtle "Controlled Synthetic" badges throughout the user interface.
- **Evidence Gap Visibility**: Where certain experimental combinations are unmeasured, PRISM displays transparent empty states and evidence gap warnings rather than inventing baseline values.
