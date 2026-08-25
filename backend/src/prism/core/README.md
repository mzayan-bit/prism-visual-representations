# PRISM Core (`prism.core`)

## Purpose
The `prism.core` module defines the fundamental abstractions, base configurations, and domain contracts shared across all PRISM subpackages.

## Intended Responsibilities
- **Base Domain Contracts**: Common base classes, protocols, and interfaces for models, datasets, and evaluators.
- **Configuration Schemas**: Validated Pydantic models for experiment configs, run metadata, and environment contracts.
- **Deterministic Constants**: System-wide defaults, seed enforcement rules, and precision policies.
- **Exceptions**: Custom domain exceptions for configuration mismatches, contract violations, and reproducibility failures.
