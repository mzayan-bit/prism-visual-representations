# PRISM API (`prism.api`)

## Purpose
The `prism.api` module is reserved for the future programmatic research API and lightweight server layer (e.g. FastAPI) connecting experiment runs and artifact stores to the frontend research interface.

## Intended Responsibilities
- **Run Querying**: Reading experiment manifests, metrics, representation geometries, and robustness benchmarks.
- **Artifact Serving**: Providing structured JSON endpoints for figures, embedding projections, and attention rollout visualizations.
- **Contract Enforcement**: Ensuring API schemas strictly mirror backend Pydantic models.
