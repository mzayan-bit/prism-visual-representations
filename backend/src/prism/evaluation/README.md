# PRISM Evaluation (`prism.evaluation`)

## Purpose
The `prism.evaluation` module implements standardized benchmark evaluation protocols, statistical testing, and metric aggregations.

## Intended Responsibilities
- **Standardized Benchmark Metrics**: Accuracy, top-k error, macro/micro F1, ECE (Expected Calibration Error), and parameter-normalized efficiency metrics.
- **Statistical Significance**: Confidence interval estimation (e.g. bootstrap resampling) across multiple seeds.
- **Benchmark Reports**: Generating immutable evaluation records conforming to the PRISM Research Contract.
