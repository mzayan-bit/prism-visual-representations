# PRISM Experiments (`prism.experiments`)

## Purpose
The `prism.experiments` module defines the experiment runner, configuration resolution, execution harness, and provenance tracking.

## Intended Responsibilities
- **Experiment Manifests**: Recording git commit SHA, dependency versions, random seeds, hardware specs, and configuration hashes.
- **Fair Trial Runners**: Executing multi-paradigm experiments under strictly identical conditions.
- **Experiment Lifecycle**: Managing experiment setup, execution, checkpointing, metric capture, and artifact archival.
