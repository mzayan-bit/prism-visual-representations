# PRISM Robustness (`prism.robustness`)

## Purpose
The `prism.robustness` module evaluates how representations degrade under visual corruptions, out-of-distribution shifts, geometric transforms, and adversarial attacks.

## Intended Responsibilities
- **Corruption Suites**: Common image corruptions (noise, blur, weather, digital artifacts) evaluated at calibrated severity levels.
- **Distribution Shifts**: Natural distribution shifts, sketch/domain transfer, and spatial perturbation resistance.
- **Calibration Under Shift**: Assessing predictive uncertainty and confidence degradation on shifted inputs.
