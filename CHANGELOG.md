# PRISM Changelog

All notable changes to the PRISM research platform are documented in this file.

---

## [1.0.0] — 2026-09-06

### Major Highlights
PRISM v1.0.0 marks the completed public release of the **PRISM (Probing the Evolution of Visual Representations)** research platform. PRISM unifies deep neural network architectures, pretraining objectives, downstream probing tasks, representation geometry, attribution, corruption robustness, uncertainty calibration, and cross-paradigm benchmark synthesis into a reproducible, deterministic, portfolio-grade research system.

---

### 1. Foundations & Engineering Rigor
- **Deterministic Seeding & SHA-256 Fingerprinting**: Implemented seed-derived pseudorandomness and cryptographic hashing across all data partition generators, weight initializations, augmentations, and patch masking, eliminating hidden global RNG mutations.
- **Dataset Universes & Manifests**: Strongly typed immutable Pydantic contracts (`SampleUniverse`, `PartitionManifest`, `MaterializedDataset`) guaranteeing deterministic data ordering and repeatable batch generation.
- **Numerical Safeguards**: Zero-norm Euclidean distance fallbacks, shift-invariant softmax, log-sum-exp cross-entropy and NT-Xent calculations, and bounded variance normalization across all tensor operations.

### 2. Model Architectures & Optimization
- **Linear & Shallow Classifiers**: Pure-Python softmax classifiers and linear probing heads with exact analytical gradient derivations.
- **Multi-Layer Perceptrons (MLPs)**: Fully connected deep networks with configurable activations (ReLU, GELU) and dropout layers.
- **Convolutional Neural Networks (CNNs)**: Composable 2D convolutions, max/average pooling, and feature extraction across arbitrary spatial depths.
- **Deep Residual Networks (ResNets)**: Multi-stage residual architectures featuring `ResidualBlock`, `ResidualAdd`, `IdentityShortcut`, `ProjectionShortcut`, and dual-branch gradient backpropagation.
- **Vision Transformers (ViTs)**: `PatchExtractor`, `PatchEmbedding`, `ClassToken`, `PositionalEmbedding`, and multi-head self-attention (`MultiHeadSelfAttention`) with exact derivative caching.
- **Optimization & Schedulers**: Momentum SGD with analytical weight decay, gradient clipping, parameter freezing, and learning rate schedulers (`Constant`, `Step`, `Exponential`, `CosineAnnealing`, `WarmupScheduler`).

### 3. Representation Geometry & Interpretability
- **Manifold Geometry Analysis**: In-memory $k$-nearest neighbors, label consistency auditing, class centroid dispersion $\mu_c$, and global separation-to-compactness ratios $\mathcal{S}/\mathcal{C}$.
- **Exact Jacobi PCA**: Pure-Python, deterministic eigenvalue solver with sign-standardized coordinate projections for reproducible 2D/3D visualization.
- **Corruption Robustness**: 6 perturbation families (`gaussian_noise`, `blur`, `brightness`, `contrast`, `occlusion`, `resolution_degradation`) with paired representation drift vectors $\mathbf{d}_i$ in shared PCA space.
- **Visual Explainability**: Gradient saliency, Gradient $\times$ Input, sliding-window occlusion maps, Grad-CAM, and ViT CLS attention rollout with cross-attribution agreement metrics.

### 4. Learning Paradigms & Pretraining
- **Transfer Learning**: 4 transfer regimes (`SCRATCH_BASELINE`, `LINEAR_PROBE`, `PARTIAL_FINE_TUNE`, `FULL_FINE_TUNE`), layer-wise linear separability probes, and sample-efficiency scaling curves.
- **Contrastive Self-Supervision (SimCLR)**: Deterministic 2N-view augmentation pipeline, 2-layer MLP projection heads, NT-Xent contrastive loss, and dimensional representation collapse diagnostics.
- **Reconstruction & Masked Autoencoding**: Deterministic SHA-256 patch masking ($r \in [0, 1]$), learnable mask tokens, spatial/patch reconstruction decoders, and masked MSE optimization.

### 5. Downstream Probes & Transfer
- **Spatial Transfer (Detection & Segmentation)**: Grid bounding box detection (`GridDetectionHead`, IoU, matching) and dense semantic segmentation (`SegmentationHead`, pixel cross-entropy, mIoU) probing spatial feature maps.
- **Temporal & Video Representations**: Canonical multi-frame sequences, temporal pooling aggregators (`Mean`, `Max`, `Last`, `Learned`, `SimpleRNN`), motion velocity correlation, and temporal perturbation robustness.
- **Multimodal Vision-Language Alignment**: Dual-encoder architectures, deterministic vocabulary/tokenization, symmetric InfoNCE contrastive loss, bidirectional text/image retrieval (R@1/3/5, MRR), and zero-shot open-vocabulary classification.

### 6. Reliability, Calibration & Uncertainty
- **Predictive Uncertainty Descriptors**: Shift-invariant predictive entropy, normalized entropy, logit margins, and probability margins.
- **Probability Calibration**: Reliability diagrams, equal-width and equal-frequency binning, Expected Calibration Error (ECE), Maximum Calibration Error (MCE), and Brier scores.
- **Post-Hoc Temperature Scaling**: Deterministic grid-search optimization of validation NLL, preserving argmax class rankings while restoring confidence calibration.
- **Out-of-Distribution (OOD) Scoring**: MSP, predictive entropy, nearest class centroid distance, $k$-NN feature distance, and Free Energy scoring with exact Mann-Whitney AUROC and AUPR discrimination metrics.

### 7. Cross-Paradigm Benchmark Orchestration & Synthesis
- **Canonical Metrics & Registries**: 25 standardized metrics across 10 evaluation axes with methodological safeguards.
- **Pure Functional Report Adapters**: Zero-loss adapters bridging all 11 PRISM domain report schemas into standard `BenchmarkResultCell` records.
- **Multi-Seed Statistical Aggregation**: Automatic mean, variance, min/max, median, and strict $N=1$ warning attachment.
- **Controlled Comparison & Audit Engine**: Enforces strict experimental factor invariance and computes direction-aware effect sizes.
- **Pareto Frontiers & Tradeoff Discovery**: Computes Pareto-optimal frontiers across conflicting research objectives (e.g., accuracy vs corruption stability).
- **Grounded Research Findings**: Automated generation of evidence-backed findings with explicit caveat flags and strength ratings.
- **Formal Report Compiler**: Compiles publication-ready Markdown, JSON, and CSV research summaries with full reproducibility manifests.

### 8. Research Workstation & Frontend Experience
- **Structured Domain Navigation**: Clean top-level navigation grouping 11 laboratories into 5 core research domains without UI clutter.
- **Interactive Visualizations**: Live SVG PCA scatterplots, neighborhood failure drawers, paired corruption displacement vectors, attention entropy bars, and Pareto trade-off explorers.
- **Official Showcase Campaign**: 810 observed evaluation cells from deterministic synthetic pretraining and evaluation runs.
- **Research-Grade Aesthetics**: Modern dark/light theme, high-contrast badges, clear scientific tooltips, and responsive layout across desktop and laptop screens.

### 9. Developer Experience & Release Quality
- **One-Command Developer Workflows**: Comprehensive `Makefile` supporting `make setup`, `make demo`, `make dev`, `make check`, `make test`, `make lint`, and `make typecheck`.
- **Zero Hidden Dependencies**: No references to personal paths (`/Users/zayan/`), secret environment keys, or unversioned files.
- **Release Smoke Suite**: High-level end-to-end integration test validating geometry, benchmarking, report generation, and demo artifact generation (`tests/smoke/test_smoke_release.py`).

### 10. Verification & Test Metrics
- **Pytest**: 657 / 657 passing tests (611 unit tests, 46 smoke tests).
- **Ruff**: 0 linting errors across 447 files; 100% formatted.
- **Mypy**: 0 type errors in strict mode across 394 source modules.
- **Frontend**: 0 ESLint errors; Next.js 16.3.2 Turbopack static prerendering passing with 0 errors.

---

## Known Limitations & Future Work
- **Compute Regime**: PRISM employs CPU-optimized pure-Python reference implementations designed for exact mathematical inspection and reproducibility rather than multi-GPU distributed throughput.
- **Controlled Datasets**: Benchmarks are evaluated on controlled synthetic and micro-scale vision universes to isolate representation properties without confounding internet-scale data distribution biases.
- **Future Directions**: Optional extensions may include scaling to larger real-world datasets, GPU acceleration backends, and additional self-supervised architectures.
