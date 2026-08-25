# PRISM Artifacts Directory

## Policy & Overview
This directory holds generated outputs from PRISM experiment runs, including trained checkpoints, structured metrics, visual figures, and compiled benchmark reports.

### Artifact Commit Policy
- **NEVER commit binary model checkpoints** (`*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`).
- **MAY commit** lightweight benchmark metric summaries (`summary.json`), curated publication figures, and canonical reports when explicitly intended.
- All artifacts must be traceable to a specific experiment configuration and code revision.

## Structure
- `checkpoints/`: Model weights, optimizer states, and training snapshots.
- `metrics/`: JSON/CSV telemetry, loss curves, and evaluation metrics.
- `figures/`: Rendered SVG/PNG figures, representation geometry charts, and attention heatmaps.
- `reports/`: Markdown and compiled experiment evaluation summaries.
