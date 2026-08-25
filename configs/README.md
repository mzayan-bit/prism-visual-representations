# PRISM Configuration Registry

## Purpose
This directory stores structured YAML/JSON configuration files for PRISM experiment definitions, dataset specifications, model architectures, and training hyperparameters.

## Structure
- `base/`: Base configurations and system defaults.
- `datasets/`: Dataset split specifications, preprocessing definitions, and benchmark configs.
- `experiments/`: Full end-to-end experiment recipe definitions combining model, dataset, training, and evaluation parameters.
- `models/`: Model architecture specifications, parameter configs, and probe head setups.
- `training/`: Optimizer configs, learning rate schedules, and training budget definitions.
