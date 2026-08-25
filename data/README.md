# PRISM Data Directory

## Policy & Overview
This directory stores local dataset caches, preprocessed manifests, and external visual resources.

### Data Storage Rules
- **NEVER commit raw or bulky dataset files to git.**
- Manifest files (`manifest.json`) containing dataset fingerprints (SHA-256 hashes, sample counts, split assignments) MAY be tracked for reproducibility.
- Automated download scripts in `scripts/` or `prism.data` will fetch external data on-demand into this directory.

## Structure
- `raw/`: Unmodified raw datasets downloaded from original upstream sources.
- `processed/`: Formatted, indexed, or canonical preprocessed datasets.
- `external/`: External benchmarks, corruption sets, or validation splits.
