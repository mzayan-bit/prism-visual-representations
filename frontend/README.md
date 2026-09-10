# PRISM Research Operating System (Frontend v1.1.0)

## Overview
The PRISM frontend is a modern, scientific research operating system built with Next.js and Tailwind CSS. It is engineered around the principle of **Progressive Disclosure**, providing calm, high-density visualization workspaces for representation learning without visual noise or telemetry clutter.

## Key Architectural Highlights
- **Global Shell (`AppShell`)**: Quiet collapsible navigation rail, sticky contextual breadcrumb bar, and collapsible experiment provenance inspector.
- **Design System (`app/components/ui`)**: Neutral graphite palette (`#0B0F17` background, slate surfaces), unified typography, single cyan/emerald semantic accents, `StatStrip` metrics, `Tabs`, `Button`, `Select`, `Badge`, `ChartFrame`, `EmptyState`.
- **Research Workspaces (`app/components/research`)**:
  - `OverviewHub`: Problem-first research question entry points and platform capability maps.
  - `GeometryWorkspace`: Large-canvas 2D PCA representation manifold projection with progressive layer evolution diagnostics.
  - `RobustnessWorkspace`: Corruption degradation curves, drift geometry, and failure explorer.
  - `ExplainabilityWorkspace`: Dual attribution workstation (Input Gradients, Grad-CAM, Attention Maps, Perturbation Stability).
  - `TransferWorkspace`: Downstream transfer strategy comparison, sample efficiency curves, layer probes, and freeze maps.
  - `SSLWorkspace`: Contrastive SimCLR dynamics, dimensional collapse analysis, and supervised alignment.
  - `ReconstructionWorkspace`: Masked autoencoding visual triptych (Original, Masked, Reconstructed) and latent space geometry.
  - `SpatialWorkspace`: Dense spatial representation transfer for bounding-box detection and pixel-level semantic segmentation.
  - `TemporalWorkspace`: Video sequence trajectory dynamics, aggregation operators, and motion perturbation stress-testing.
  - `MultimodalWorkspace`: Vision-language shared embedding space, bidirectional retrieval, and prompt sensitivity sweeps.
  - `UncertaintyWorkspace`: Predictive calibration (ECE, MCE, NLL), post-hoc temperature scaling, and feature-space OOD detection.
  - `BenchmarkObservatoryView`: Cross-paradigm synthesis matrix, Pareto frontiers, and automated markdown/LaTeX report generation.

## Stack
- **Framework**: Next.js 16 (App Router, Turbopack)
- **Language**: TypeScript 5 (Strict mode)
- **Styling**: Tailwind CSS 4 + Design System Tokens
- **Icons**: Lucide React
- **Package Manager**: npm

## Development
To start the local development server:

```bash
cd frontend
npm install
npm run dev
```

The application will be available at [http://localhost:3000](http://localhost:3000).

## Build & Validation
```bash
npm run lint
npm run build
```
