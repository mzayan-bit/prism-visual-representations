# PRISM Utilities (`prism.utils`)

## Purpose
The `prism.utils` module hosts shared low-level utilities including structured logging, deterministic seeding, data hashing, and filesystem path helpers.

## Intended Responsibilities
- **Deterministic Seeding**: Setting and verifying seeds across Python `random`, `numpy`, and PyTorch backends.
- **Hashing & Checksums**: Computing SHA-256 digests of experiment configurations, datasets, and checkpoint tensors for auditability.
- **Structured Logging**: Consistent JSON and console logging for experiment runs.
