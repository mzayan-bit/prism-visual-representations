# PRISM Research Contract

## Phase 13 architecture-comparison contract

Architecture comparisons are measurements, not universal rankings. Each suite
records dataset and controlled-data identities, preprocessing, reproducibility
policy, training budget, evaluation protocol, architecture metadata, exact
trainable parameter count, and intentionally varied factors. Strict suites
fail when an undeclared typed factor differs; architecture-appropriate suites
retain every permitted difference for audit.

Unavailable metrics remain unavailable. CNN and residual CNN attention is not
applicable; ViT attention summaries are included only when real attention
weights exist. Representation dimensions are explicit, and gradient and
distribution summaries are descriptive rather than causal or inferential.

## Core Scientific Principles & Non-Negotiable Standards

PRISM is a scientific research platform designed to investigate how visual representations are formed, structured, and transferred across different learning paradigms.

To maintain scientific integrity and prevent invalid comparisons, all contributions and experiments within PRISM must adhere strictly to this **Research Contract**.

---

### 1. Fair Comparison & Controlled Data Identity
Different learning paradigms (e.g. Linear Models, MLPs, CNNs, ResNets, Vision Transformers, Self-Supervised Learning) must be compared under strictly controlled and explicitly documented conditions:
- **Input Consistency**: Evaluated models must receive identical input dimensions, color space normalizations, and test split partitions.
- **Canonical Sample Universes**: Datasets must declare deterministic sample identities (`SampleRecord`) and ordered universes (`CanonicalSampleManifest`).
- **Fixed Benchmark Partitions**: Benchmark partitions (`PartitionManifest`) must derive validation splits deterministically while keeping official benchmark test splits strictly isolated.
- **Nested Data-Budget Subsets**: In low-supervision / data-efficiency regimes (1%, 5%, 10%, 25%, 50%, 100%), subsets must satisfy mathematical nesting ($S_{1\%} \subseteq S_{5\%} \subseteq S_{10\%} \subseteq S_{25\%} \subseteq S_{50\%} \subseteq S_{100\%}$). Independent random sampling across budgets is strictly prohibited.
- **Sample Identity Preservation**: Runtime materialization (`MaterializedSample`, `MaterializedDataset`) must preserve original `sample_id`, `source_split`, `source_index`, and ground-truth targets.
- **Deterministic Batch Traceability**: Every batch emitted by `DeterministicBatchLoader` must retain its constituent list of `sample_ids` alongside numerical payloads.
- **Ordering Fingerprints**: Traversal and shuffle sequences are cryptographically hashed (`compute_ordering_fingerprint`) across sample identities, strategy, seed, and epoch.
- **Compute & Parameter Parity**: Experimental configurations must explicitly document parameter counts, FLOPs, and gradient step budgets.
- **Controlled Augmentation**: Data augmentations used during training or evaluation must be precisely recorded and held constant when isolating architectural differences.

### 2. Spatial Inductive Bias & Equivariance Standards
When evaluating Convolutional Neural Networks against non-spatial models (Linear classifiers, MLPs):
- **Preservation of Spatial Structure**: CNNs operate natively on $N \times C \times H \times W$ feature maps. Images must not be flattened prior to spatial convolution layers.
- **Local Connectivity & Weight Sharing**: Kernels slide over local receptive fields with shared weights across spatial translations.
- **Translation Equivariance vs Invariance**:
  - **Equivariance**: A spatial translation of the input translates the internal feature map correspondingly: $f(T_g(x)) = T_g(f(x))$.
  - **Invariance**: Downstream pooling and dense classifier projections compress spatial maps toward translation invariance: $f(T_g(x)) \approx f(x)$.
  - Claims regarding spatial properties must distinguish equivariance in intermediate feature maps from invariance in classifier logits.
- **Receptive Field Traceability**: Convolutional architectures must provide deterministic receptive field ($RF$) and effective stride ($J$) calculations per layer.
- **Spatial Feature Map Extraction**: Intermediate spatial feature maps (`"conv_0"`, `"pool_0"`, `"final_spatial"`) must be extractable with preserved spatial dimensions $[N, C, H, W]$.

### 3. Normalization, Optimization & Distribution Standards
When studying the effect of normalization layers (`BatchNorm1D`, `BatchNorm2D`):
- **Explicit Train vs Eval Semantics**:
  - Training mode: Computes batch statistics ($\mu_B, \sigma_B^2$) and updates non-trainable running statistics via exponential moving average.
  - Evaluation mode: Normalizes strictly using accumulated running statistics without updating running state or using test batch statistics.
- **Trainable Parameters vs Model State**:
  - Scale ($\gamma$) and shift ($\beta$) are learnable parameters updated solely by the optimizer.
  - Running mean and running variance are non-trainable model state and must never receive optimizer gradients or weight decay penalties.
- **Channel-Wise Spatial Statistics**: In convolutional feature maps $[N, C, H, W]$, statistics are computed channel-wise across all $M = N \times H \times W$ elements. Independent coordinate-wise normalization is prohibited.
- **Avoid Oversimplified Explanations**: Research documentation must distinguish observed optimization effects (gradient smoothing, scaling robustness, loss landscape conditioning) from contested causal hypotheses like "internal covariate shift elimination".
- **Small-Batch Limitations**: Acknowledge that batch normalization variance estimates become noisy at small batch sizes ($N < 4$) and degenerate at $N=1$.
- **Distribution Auditing**: Feature distributions must be measurable via `FeatureDistributionSummary` (mean, variance, min/max, zero fraction, channel stats) without requiring full activation dumps.

### 4. Residual Learning & Gradient Flow Standards
When evaluating Residual Neural Networks (`ResidualNeuralNetwork`) against Plain Deep CNNs:
- **Explicit Addition Formulation**: The residual connection computes $y = \text{activation}(F(x) + S(x))$ with explicit shape validation. No implicit cropping or shape coercion.
- **Dual-Branch Analytical Backpropagation**: Upstream gradient $dZ$ routes identically to both main path ($dF = dZ$) and shortcut path ($dS = dZ$). Total input gradient is the exact sum $dX = dX_{\text{main}} + dX_{\text{shortcut}}$.
- **Identity vs Projection Shortcut Disciplines**:
  - Identity shortcuts ($S(x) = x$) are strictly parameter-free when spatial and channel dimensions match.
  - Projection shortcuts ($S(x) = \text{Norm}(\text{Conv2D}(x, 1\times 1, \text{stride}=s))$) are used exclusively when spatial resolution downsamples or channel width expands.
- **Gradient Flow Auditing**:
  - Gradient flow is monitored across depth via `ParameterGradientSummary` and `ModelGradientFlowSummary` recording L2 norm, mean, std dev, min/max, zero fraction, and finiteness.
  - Research claims regarding gradient flow must distinguish gradient norm magnitudes from convergence quality. High gradient norms do not inherently guarantee optimal representations.
- **Plain vs Residual Controlled Comparisons**: Comparisons must strictly match total block depths, stage widths, dataset manifests, RNG seeds, and training hyperparameters via `create_residual_comparison()`.

### 5. Learning Rate Scheduling & Optimization Control Standards
When studying learning rate control strategies (Constant, Step, Exponential, Cosine, Linear Warmup, Composed Warmup):
- **Schedule as an Experimental Variable**: The learning rate schedule must be treated as an explicit, auditable variable. Changing schedule strategies alters the experiment configuration fingerprint.
- **Clean Ownership Boundary**: Schedulers strictly own learning rate progression over logical steps/epochs; Optimizers strictly own model parameters and update mechanics. Schedulers never mutate model parameters directly.
- **Deterministic Mathematical Formulations**: Schedulers must adhere to mathematically unambiguous stepping and boundary conventions without off-by-one ambiguities or floating-point instability.
- **Exact State Restoration**: Schedulers must support state snapshots (`SchedulerState`) enabling serialized models to continue along identical future learning rate trajectories upon restoration.
- **Warmup Continuity**: Composed warmup schedules must ensure exact continuity at the warmup horizon without unintended jumps or discontinuities.
- **Controlled Schedule Comparisons**: Schedule comparisons must be formally defined via `create_scheduler_comparison()`, holding dataset, architecture, initialization, batch size, and total training budgets strictly invariant.
- **Scientific Humility**: PRISM documentation and findings must not claim that any particular schedule guarantees superior performance across all datasets or architectures; empirical effects on convergence rate, final loss, and representation geometry must be evaluated objectively.

### 6. Vision Transformer & Self-Attention Standards
When investigating patch representations and self-attention operations:
- **Explicit Patch Geometry**: Images must be partitioned into non-overlapping patches using a documented, row-major spatial sequence. Analytical backward passes must reconstruct original spatial image dimensions exactly without coordinate distortion.
- **Global Contextual Dependency**: Self-attention maps pairs of tokens globally across sequence positions ($L \times L$) unlike localized sliding convolutional receptive fields.
- **Multi-Head Subspace Diversity**: Multiple attention heads project inputs into independent subspaces ($D_{\text{head}} = D_{\text{embed}} / H$), enabling the model to jointly attend to information from different representation subspaces at different positions.
- **Shared Class Token Discipline**: The learnable class token $[1, 1, D_{\text{embed}}]$ is shared across all batch items. Parameter updates must aggregate gradients across the entire batch dimension.
- **Exact Analytical Gradient Routing**: Upstream gradients must propagate analytically through softmax derivatives and all three projection paths ($Q, K, V$), accumulating exactly into the input gradient $dX = dX_Q + dX_K + dX_V$.
- **Attention Distribution Audits**: Attention weight matrices must be monitored using `summarize_attention_weights` (`AttentionTensorSummary`), validating row normalization ($\sum_j A_{ij} = 1$) and tracking Shannon entropy without corrupting or mutating forward state.

### 7. Strict Reproducibility & Deep Learning Invariants
Every experimental result generated in PRISM must be fully reproducible and explicitly audited prior to execution:
- **Configuration Fingerprinting**: Semantic SHA-256 digests (`compute_fingerprint()`) calculated over model, data, partition, subset, scheduler, and optimizer specifications.
- **Deterministic Parameter Initialization**: Model parameters must be deterministically initialized (e.g. Xavier for linear layers, He/Kaiming for ReLU hidden layers and Conv2D kernels, $\gamma=1, \beta=0$ for normalization) using configured seeds without dependence on accidental global RNG state.
- **Multi-Backend RNG Seeding**: Explicit initialization of seeds across Python standard library `random`, `PYTHONHASHSEED`, `numpy.random`, and PyTorch CPU/CUDA/MPS RNGs.
- **Deterministic Dropout Masking**: Stochastic dropout masks during training must be deterministically derived from explicit experiment seed, layer index, and step counters. In evaluation mode, dropout must be strictly disabled (identity).
- **Controlled Learning Rate Schedules**: Learning rate decay (Step, Cosine Annealing, Warmup) must be deterministic and logged into metric telemetry.
- **Explicit Weight Decay Semantics**: Regularization must be applied through a single documented pathway (e.g. optimizer step) without duplicate loss penalties.
- **Representation Extraction**: Intermediate hidden representations (`"input"`, `"stem"`, `"stage_0_block_0_post_add"`, `"final_spatial"`, `"final_hidden"`) must be extracted without parameter mutation or stochastic dropout noise.
- **Code Revision Provenance**: Active Git commit SHA, branch, and working tree cleanliness (`-uno` tracked modifications) captured via `inspect_git_provenance()`.
- **Hardware & Environment**: Python runtime version, host OS, primary compute backend, and installed versions of allowlisted dependencies (`pydantic`, `torch`, `torchvision`, etc.).

### 6. No Silent Comparisons
PRISM strictly prohibits unrecorded or implicit divergences between experimental conditions. The following are non-negotiable prohibitions:
- **No Hidden Split Alterations**: Test sets and validation sets are immutable once benchmark manifests are registered.
- **No Uncontrolled Preprocessing**: Data pipelines must execute through deterministic abstractions in `prism.data`.
- **No Selective Tuning**: Hyperparameter search budgets must be balanced across compared paradigms.
- **No In-Place Evaluation Modifications**: Evaluation routines must never mutate model parameters or gradient buffers.

### 7. Controlled Comparison Contracts
All experimental comparisons must be formally defined via `ControlledComparison`, explicitly documenting:
- The baseline and candidate experiment identifiers.
- The isolated varied factor(s) (e.g. `{"architecture": {"baseline": "plain_cnn", "candidate": "residual_cnn"}}`).
- The fixed invariant factors (dataset fingerprint, partition fingerprint, seed, optimization schedule).

### 8. Experiment Traceability
Every figure, table, metric entry, or embedding projection generated by PRISM must be traceable backwards through the provenance chain:

```
Artifact (Figure / Metric / Summary / GradFlow)
  ↳ Experiment Run ID
    ↳ PreparedExecution Runtime Context & DataRuntimeContext
      ↳ Complete Configuration File (YAML)
        ↳ Canonical Universe (SHA-256) + Partition (SHA-256) + Subset (SHA-256) + Ordering (SHA-256) + Git SHA + Seed
```

---

### 9. Honest Research
- **Zero Synthetic Results**: Synthetic, fabricated, or mocked results must never be committed as experimental findings.
- **Explicit Labeling**: All artifacts and documentation must explicitly distinguish their status:
  - **Planned**: Conceptually defined experiments with pending implementation.
  - **Implemented**: Executable experiment configurations ready for evaluation.
  - **Validated**: Executed runs whose metrics and artifacts have been verified across multiple random seeds.
  - **Illustrative**: Synthetic or stylized diagrams created purely for didactic architectural explanations.

### 10. Representation Geometry & Manifold Analysis Standards
When evaluating representation geometry across layers and architectures:
- **Coordinate Space Independence**: Distinct models and layers operate in independent feature spaces with unaligned bases. Projections (e.g. PCA) must never be overlaid as a single shared coordinate system. Comparisons must evaluate invariant scalar geometric properties (compactness, separation, $k$-NN consistency, spectrum decay) rather than direct point overlays.
- **Vectorization Traceability**: Spatial feature tensors $[N, C, H, W]$ and sequence tokens $[N, S, D]$ must declare explicit vectorization policies (`GLOBAL_AVERAGE_POOL` or `FLATTEN`) and record original shapes in metadata.
- **Deterministic PCA Projections**: Low-dimensional PCA projections must use deterministic solvers (e.g. Jacobi rotations with canonical sign orientation) ensuring identical coordinates across repeated runs on identical inputs.
- **Statistical Significance in Compactness**: Flag estimation warnings when individual class sample counts ($n_c < 3$) are insufficient for stable intra-class variance estimation.
- **Dimensionality Caveat Disclaimers**: 2D/3D visualizations capture only a subspace of total variance; reports must always display cumulative explained variance ratios.

### 11. Robustness & Distribution Shift Evaluation Standards
When evaluating visual models under corruptions and distribution shifts:
- **Non-Destructive Corruptions**: Clean sample records, images, and dataset manifests must never be overwritten, modified, or permanently duplicated. Corruptions are applied on-the-fly via deterministic `CorruptedDatasetView` wrappers preserving sample IDs and target linkages.
- **Frozen Model Evaluation**: Robustness evaluation runs strictly over frozen models. Model parameters must never be updated, and normalization layers (`BatchNorm2D`) must not update running statistics during corruption evaluation.
- **Shared PCA Basis Protocol**: For clean-vs-corrupted manifold comparisons within a single model and layer, PCA must be fitted strictly on clean representations $\mathbf{X}_{\text{clean}}$. Corrupted representations $\mathbf{X}_{\text{corrupt}}$ are transformed into the identical coordinate basis $\mathbf{Z}_{\text{corrupt}} = (\mathbf{X}_{\text{corrupt}} - \boldsymbol{\mu}_{\text{clean}})\mathbf{V}_{\text{clean}}$, ensuring that displacement vectors $\mathbf{d}_i = \mathbf{z}_i' - \mathbf{z}_i$ represent valid geometric trajectories in a shared coordinate space.
- **ViT Attention Drift Applicability**: Attention pattern shifts and entropy changes are computed exclusively when multi-head self-attention mechanisms exist (Vision Transformers). They are marked explicitly as not applicable for CNN and ResNet models.
- **Multi-Severity Trajectory Auditing**: Corruptions must be evaluated across multiple calibrated severities (1 to 5) to produce monotonic or structured degradation curves with normalized Area Under Curve (AUC) metrics.

### 12. Explainability & Visual Attribution Standards
When computing, comparing, and visualizing model-attribution maps:
- **Attribution is Descriptive Evidence, Not Causal Truth**: Attribution maps measure mathematical sensitivity ($\frac{\partial S_c}{\partial x}$), perturbation sensitivity ($\Delta S_c$), activation weighting ($\alpha_k A^k$), or attention routing. They must never be described as definitive causal explanations of model decisions.
- **Frozen Model & Invariant State**: Attribution routines must never mutate model weights, biases, or `BatchNorm` running statistics. Forward and backward passes run strictly in evaluation mode (`model.eval()`).
- **Explicit Target Class Semantics**: Every attribution map must record its target class $c$, target mode (`PREDICTED_CLASS`, `TRUE_CLASS`, `EXPLICIT_CLASS`), and pre-softmax logit score $S_c$.
- **Architectural Applicability**:
  - Grad-CAM requires intermediate spatial feature activations and gradients; it applies to CNNs and ResNets and is explicitly rejected for Vision Transformers.
  - ViT CLS-to-patch attention attribution applies exclusively to Vision Transformers and is marked unsupported for CNNs and ResNets.
  - Input Gradient, Gradient $\times$ Input, and Occlusion Sensitivity are model-agnostic and apply to all differentiable/forward-evaluable vision models.
- **Deterministic Normalization & Colormaps**: Signed gradient maps must preserve positive and negative polarity and be rendered with diverging colormaps. Non-negative heatmaps (Grad-CAM, Attention) use sequential colormaps (`turbo`, `plasma`, `viridis`) with explicit min-max or sum scaling policies.
- **Cross-Method Agreement & Attribution Drift**: Cross-method comparisons and clean-vs-corrupted drift must report objective quantitative metrics (cosine similarity, top-10% Jaccard overlap, center-of-mass Euclidean displacement) on matched spatial grids.

### 13. Transfer Learning & Representation Reuse Standards
When evaluating learned representation reuse and downstream adaptation across tasks:
- **Self-Contained Model Provenance**: PRISM studies representation reuse strictly on models trained within PRISM's controlled environment. Downloading opaque external weights (e.g. ImageNet checkpoints) is strictly prohibited.
- **Strict Parameter Freezing Semantics**: Freezing is not merely setting learning rate to zero. Frozen parameters must be completely excluded from optimizer parameter lists, momentum velocity accumulation, and weight decay penalties.
- **Controlled Baseline Matching**: Every transfer experiment must evaluate a matched `SCRATCH_BASELINE` (identical architecture, target dataset partition, optimizer hyperparameters, and step budget initialized without source weights).
- **Four Distinct Adaptation Regimes**:
  - `SCRATCH_BASELINE`: Random initialization on target data without source pretraining.
  - `LINEAR_PROBE`: Backbone strictly frozen; only newly initialized linear classifier head is trained.
  - `PARTIAL_FINE_TUNE`: Early spatial/token layers frozen; late semantic stages and classifier head updated.
  - `FULL_FINE_TUNE`: All backbone and head parameters updated end-to-end with backpropagation.
- **BatchNorm Transfer Policies**: Transfer specifications must declare whether normalization statistics are frozen (`FREEZE_SOURCE_STATS`, normalization layers kept in evaluation mode) or adapted (`ADAPT_RUNNING_STATS`, tracking target domain running mean and variance).
- **Layer Transferability Discipline**: Linear probes trained across intermediate layer activations must extract features in evaluation mode (`model.eval()`) without gradient propagation into the backbone.
- **Representation Retention & Drift**: Retention analysis must measure Euclidean distance, cosine similarity, and relative norm change on a shared reference dataset between pre-transfer and post-transfer states.
- **Shared PCA Basis Protocol for Transfer**: To visualize feature drift during fine-tuning, PCA must be fitted on pre-transfer representations $\mathbf{X}_{\text{pre}}$ and both pre- and post-representations projected into that shared basis $\mathbf{Z}_{\text{pre}}, \mathbf{Z}_{\text{post}}$, yielding meaningful displacement trajectories.
- **Target Label-Efficiency Trajectories**: Data budgets must use mathematically nested target partitions ($S_{10\%} \subseteq S_{25\%} \subseteq S_{50\%} \subseteq S_{100\%}$) to compute normalized Area Under Curve (AUC) transfer advantage.

---

## Data and Artifact Policy

### Prohibited from Version Control:
- Raw image files and bulky downloaded datasets (`data/raw/*`, `data/processed/*`, `data/external/*`).
- Binary model weights, checkpoints, and optimizer state dictionaries (`artifacts/checkpoints/*`, `*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`).
- Temporary run logs and voluminous raw step logs.

### Permitted in Version Control:
- Small metadata files and dataset manifests (`manifest.json`) recording checksums and split definitions.
- Declarative experiment recipes (`configs/**/*.yaml`).
- Lightweight benchmark metric summaries (`summary.json`).
- Selected publication-quality vector and raster figures (`artifacts/figures/*`).
- Research analysis reports (`experiments/reports/*`).
