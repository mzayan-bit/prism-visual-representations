# PRISM Training (`prism.training`)

## Purpose
The `prism.training` module provides deterministic training loops, learning rate schedules, optimizer configurations, and checkpointing routines.

## Intended Responsibilities
- **Controlled Optimization**: Enforcing exact equality of optimization budgets (e.g. total gradient steps, FLOPs, or sample exposures) when comparing paradigms.
- **Deterministic Loop Protocol**: Seeding all RNGs (Python, NumPy, PyTorch CPU/CUDA) and recording deterministic hardware flags.
- **Checkpointing Contracts**: Saving standardized checkpoint states alongside complete training telemetry and hyperparameters.
