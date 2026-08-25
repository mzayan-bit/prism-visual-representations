# PRISM Data (`prism.data`)

## Purpose
The `prism.data` module manages dataset abstractions, deterministic preprocessing, dataset fingerprints, and split specifications.

## Intended Responsibilities
- **Manifests & Fingerprints**: Computing SHA-256 fingerprints of datasets and split definitions to ensure strict reproducibility.
- **Controlled Preprocessing**: Guaranteeing that competing models receive identical, controlled data pipelines without hidden differences in normalizations, augmentations, or splits.
- **Dataset Registry**: Providing standard loaders for vision benchmarks and controlled synthetic visual datasets.
