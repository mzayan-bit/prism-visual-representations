# PRISM Models (`prism.models`)

## Purpose
The `prism.models` module provides unified interfaces for visual representation learners across paradigms (linear models, MLPs, CNNs, Transformers, Self-Supervised backbones, and probe heads).

## Intended Responsibilities
- **Unified Forward Contract**: Exposing both downstream task predictions and intermediate layer representations/features across architectures.
- **Model Registry**: Modular instantiation of backbones with explicit parameter counts, receptive fields, and computational costs.
- **Probe Heads**: Standard linear probes, non-linear probes, and segmentation/detection adaptors for frozen representation evaluation.
