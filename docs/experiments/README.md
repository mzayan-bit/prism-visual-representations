# PRISM Experiments Guide

## Purpose
This directory stores documentation and tracking protocols for PRISM research campaigns.

## Experiment Lifecycle
1. **Definition**: Author a declarative experiment recipe in `configs/experiments/`.
2. **Pre-flight Audit**: Verify that random seeds, dataset fingerprints, and compute budgets match baseline controls.
3. **Execution**: Run the experiment harness via `prism.experiments`.
4. **Validation**: Verify reproducibility across multiple random seeds.
5. **Synthesis**: Generate benchmark scorecards, CKA geometry plots, and analytical summaries in `experiments/reports/`.
