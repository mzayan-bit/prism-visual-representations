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

## 10. Self-Supervised Representation Learning Contracts

- **Complete Class Label Exclusion**: Class labels must never contribute to the encoder pretraining objective or parameter update gradients. Labels are only attached during post-hoc linear evaluation or geometric clustering.
- **Deterministic Paired Augmentations**: Two views $x_i, x_j$ derived from the same source image are transformed using deterministic, seed-derived augmentation policies with audited transform traces.
- **Non-Linear Projection Head**: The projection head $g(\mathbf{h}) = \mathbf{W}_2 \text{ReLU}(\mathbf{W}_1 \mathbf{h} + \mathbf{b}_1) + \mathbf{b}_2$ is used solely during contrastive pretraining and discarded for downstream linear probing.
- **Normalized Temperature-Scaled Cross-Entropy (NT-Xent)**:
  $$\ell_{i,j} = -\log \frac{\exp(\text{sim}(\hat{\mathbf{z}}_i, \hat{\mathbf{z}}_j)/\tau)}{\sum_{k=1}^{2N} \mathbb{I}_{[k \neq i]} \exp(\text{sim}(\hat{\mathbf{z}}_i, \hat{\mathbf{z}}_k)/\tau)}$$
  with exact analytical gradient backpropagation to both projection head and upstream encoder backbones.
- **Representation Collapse Invariants**: Continuous monitoring of feature dimensional standard deviations $\sigma_d = \sqrt{\text{Var}(h_d)}$, active channel fractions, and distinct-sample angular spreads to detect complete or dimensional collapse.
- **Evaluation Protocol**: Fixed linear probe on frozen SSL representations across matched target splits and data budgets compared against supervised pretraining and random initialization baselines.

---

## 11. Reconstruction-Based Representation Learning & Masked Modeling Contracts

- **Unsupervised Label Independence Invariant**: Target class labels are strictly prohibited from entering the reconstruction loss computation, patch masking decision logic, or backward gradient graphs.
- **Deterministic Seed-Derived Patch Masking**: Patch masks are generated using deterministic SHA-256 hash digests (`MaskingContext`) ensuring perfect reproducibility across epochs, batches, and seeds without side effects on global RNG.
- **Strict Mask Partitioning**: Mask ratio $r \in [0.0, 1.0)$ determines the exact masked count $M = \lfloor T \cdot r \rfloor$. Selected patch indices must be strictly valid ($0 \le p_i < T$) with zero duplicates.
- **Analytical Masked MSE Loss & Zero-Gradient Routing**:
  $$\mathcal{L}_{\text{masked}} = \frac{1}{M \cdot D} \sum_{i \in \mathcal{M}} \|\hat{\mathbf{p}}_i - \mathbf{p}_i\|_2^2$$
  Visible patches receive exactly zero analytical gradient ($d\mathbf{p}_j = \mathbf{0}, \forall j \notin \mathcal{M}$). Upstream encoder gradients propagate exclusively through masked positions and the learnable mask token.
- **Reconstruction Quality $\neq$ Semantic Representation Quality**: Low reconstruction MSE does not automatically imply high semantic representation utility. Models may minimize pixel-level MSE by predicting low-frequency blur or smooth interpolations without learning linearly separable class representations.
- **Failure Taxonomy Discipline**: Diagnostic failures must be objectively categorized according to mathematical thresholds (`HIGH_RECONSTRUCTION_ERROR`, `LOCALIZED_PATCH_FAILURE`, `LOW_LATENT_VARIANCE`, `OVER_SMOOTH_RECONSTRUCTION`, `CORRUPTION_RECOVERY_FAILURE`) without speculative assertions.

---

## 12. Spatial Transfer Contracts (Detection & Segmentation)

- **Canonical Coordinate Format**: All bounding boxes are represented as normalized coordinates $(x_{\min}, y_{\min}, x_{\max}, y_{\max}) \in [0.0, 1.0]^4$ satisfying $0.0 \le x_{\min} < x_{\max} \le 1.0$ and $0.0 \le y_{\min} < y_{\max} \le 1.0$. Malformed coordinates, negative areas, inverted coordinates, and non-finite values are strictly rejected at the schema boundary.
- **2D Pixel Segmentation Masks**: Segmentation ground truths are 2D integer arrays $H \times W$ with $M(y, x) \in \{0, 1, \dots, K-1\}$. Spatial dimensions must match the source image dimensions.
- **ViT Patch-Spatial Unflattening Invariant**: ViT patch tokens $\mathbf{T} \in \mathbb{R}^{N \times T \times D}$ (excluding CLS token) are reshaped into 4D spatial feature maps $[N, D, H_p, W_p]$ using the explicit patch geometry descriptor ($T = H_p \cdot W_p$). CLS tokens are strictly excluded from the spatial feature grid.
- **Lightweight Spatial Head Design**: Spatial probing uses $1 \times 1$ conv projections directly on 4D feature maps without anchor pyramids, Region Proposal Networks (RPN), non-maximum suppression (NMS) dependencies, or multi-scale feature pyramids (FPN).
- **Exact Evaluation Protocols vs Production Baselines**: Detection transfer is evaluated using exact bounding box IoU and deterministic greedy 1-to-1 matching at configured IoU thresholds. Reports must NOT claim COCO mAP unless full COCO-style PR curves and area-tiered AP are computed. Segmentation transfer is evaluated using full confusion matrices, pixel accuracy, per-class IoU, and mean IoU.
- **Parameter Freeze Fidelity**: Frozen spatial probes (`FROZEN_SPATIAL_PROBE`) must leave encoder parameters bitwise identical throughout training. Upstream encoder gradients propagate only under partial or full fine-tuning.
- **Representation Drift Tracking**: Spatial fine-tuning drift is quantified via normalized cosine distance and RMSE between spatial features extracted before vs after transfer.

---

## 13. Video & Temporal Representation Learning Contracts

- **Temporal Sample Identity & Canonical Tensor Shapes**: Video sequences are represented as canonical 4D tensors $T \times C \times H \times W$ with explicit frame IDs, sequential temporal indices, video-level categorical labels, and optional ground-truth motion trajectories (`MotionTrajectory`). Every frame preserves immutable lineage (`video_id`, `frame_index`, `frame_id`).
- **Shared Frame Encoder Weight Discipline**: 100% of image encoder weights are shared across all timesteps. Instantiating separate encoder weights per frame is strictly prohibited. Frame features are extracted by flattening $[N \cdot T, C, H, W] \to [N, T, D]$.
- **Lightweight Temporal Aggregators & Sequence Representations**:
  - **Mean Temporal Pooling**: $\mathbf{z} = \frac{1}{T} \sum_{t=1}^T \mathbf{h}_t$, distributing upstream gradients equally ($d\mathbf{h}_t = \frac{1}{T} d\mathbf{z}$).
  - **Max Temporal Pooling**: $z_d = \max_{t} h_{t, d}$, routing upstream gradient strictly to the deterministic argmax timestep (deterministic tie-breaking policy).
  - **Last-Frame Baseline**: $\mathbf{z} = \mathbf{h}_T$, evaluating final-timestep representation sufficiency without temporal aggregation.
  - **Learned Temporal Pooling**: $\alpha_t = \text{softmax}(\mathbf{w}^T \mathbf{h}_t + b)$, $\mathbf{z} = \sum_t \alpha_t \mathbf{h}_t$. Softmax weights are aggregation weights, NOT causal explanations.
  - **Simple Recurrent Neural Network (SimpleRNN)**: $\mathbf{h}_t = \tanh(\mathbf{W}_x \mathbf{x}_t + \mathbf{W}_h \mathbf{h}_{t-1} + \mathbf{b})$ with initial hidden state $\mathbf{h}_0 = \mathbf{0}$ and exact analytical Backpropagation Through Time (BPTT).
- **Temporal Consistency & Motion Sensitivity Metrics**:
  - **Adjacent Distance & Cosine Similarity**: Measures frame-to-frame representation trajectory smoothness $\|\mathbf{h}_t - \mathbf{h}_{t-1}\|_2$ and cosine alignment $\frac{\mathbf{h}_t \cdot \mathbf{h}_{t-1}}{\|\mathbf{h}_t\|_2 \|\mathbf{h}_{t-1}\|_2}$.
  - **Temporal Drift Curves**: Long-range representation drift $d(\mathbf{h}_0, \mathbf{h}_t)$ tracking representation displacement across the entire sequence length.
  - **Motion-Drift Sensitivity**: Pearson correlation between spatial object displacement $\|\mathbf{p}_t - \mathbf{p}_{t-1}\|_2$ and representation feature delta $\|\mathbf{h}_t - \mathbf{h}_{t-1}\|_2$.
- **Static-Sequence Control Invariant**: In identical-frame sequences ($\mathbf{x}_t \equiv \mathbf{x}_0$), frame representations must be identical in eval mode and temporal drift must equal zero ($d \approx 0.0$).
- **Order Sensitivity Invariant**: Set-based pooling methods (Mean, Max, Learned Temporal Pooling without positional embeddings) are mathematically order-invariant. SimpleRNN is order-sensitive. Research reports must explicitly state order invariance properties.
- **Deterministic Temporal Corruptions**: Robustness testing uses audited perturbations (`FRAME_DROP`, `FRAME_DUPLICATION`, `FRAME_SHUFFLE`, `TEMPORAL_SUBSAMPLING`, `SPATIAL_COMPOSITE`) with complete record lineage.
- **Cross-Objective Pretraining Transfer**: Controlled comparison of Supervised, SimCLR, Reconstruction, and Scratch representations under identical downstream temporal classification tasks without altering the source representation structure.
- **Strict Scope Boundaries & Non-Goals**: PRISM is not a production video analytics pipeline, multi-object tracker, action detector, or video captioner. The objective is fundamental representation learning across time.

---

## 14. Multimodal Vision-Language Representation Alignment Contracts

- **Dual-Encoder Architecture & Shared Metric Space**: Visual representations $\mathbf{v} \in \mathbb{R}^D$ and textual representations $\mathbf{t} \in \mathbb{R}^D$ are mapped into a shared, unit-normalized metric space via dedicated projection heads $g_v(\mathbf{h}_v)$ and $g_t(\mathbf{h}_t)$, where $\|\hat{\mathbf{v}}\|_2 = 1.0$ and $\|\hat{\mathbf{t}}\|_2 = 1.0$.
- **Symmetric Dual Contrastive Loss**:
  $$\mathcal{L} = \frac{1}{2N} \left( \sum_{i=1}^N -\log \frac{\exp(\hat{\mathbf{v}}_i \cdot \hat{\mathbf{t}}_i / \tau)}{\sum_{j=1}^N \exp(\hat{\mathbf{v}}_i \cdot \hat{\mathbf{t}}_j / \tau)} + \sum_{j=1}^N -\log \frac{\exp(\hat{\mathbf{v}}_j \cdot \hat{\mathbf{t}}_j / \tau)}{\sum_{i=1}^N \exp(\hat{\mathbf{v}}_i \cdot \hat{\mathbf{t}}_j / \tau)} \right)$$
  Analytical derivatives are evaluated with exact gradient backpropagation through L2 vector normalization.
- **Strict Unsupervised Label Independence Invariant**: Contrastive alignment is learned strictly through paired image-text association. Class labels are completely excluded from the contrastive forward loss and backward gradient computation graphs.
- **Deterministic Vocabulary & Tokenizer Governance**:
  - Special tokens are immutably pinned to indices: `<PAD>` = 0, `<UNK>` = 1, `<BOS>` = 2, `<EOS>` = 3.
  - Lexical tokens are deterministically sorted alphabetically starting at index 4.
  - Tokenizers enforce explicit padding and attention masks without reliance on external NLP tokenization libraries.
- **Bidirectional Cross-Modal Retrieval Metrics**: Retrieval evaluation reports Recall@1, Recall@3, Recall@5, and Mean Reciprocal Rank (MRR) for both Image-to-Text ($I \to T$) and Text-to-Image ($T \to I$) directions.
- **Zero-Shot Classification Protocol**: Open-vocabulary classification is performed by embedding class prompt texts ("a photo of a {class}"), measuring cosine similarity with image embeddings, and predicting class via $\arg\max_c (\hat{\mathbf{v}} \cdot \hat{\mathbf{t}}_c)$.
- **Shared Geometry & Multimodal Collapse Invariants**:
  - Joint PCA is fitted on the concatenated embedding matrix $[\mathbf{V}, \mathbf{T}]$, projecting both visual and textual points into a shared 2D coordinate basis.
  - Multimodal collapse diagnostics track dimensional standard deviation ($\sigma_d > 0.05$), non-zero variance fractions, and paired similarity gaps to detect complete, dimensional, or modality collapse.
- **Multimodal Robustness Evaluation**: Visual corruptions are applied to image inputs while holding textual captions fixed, measuring paired Euclidean visual drift, cosine degradation, and alignment retention.
- **Strict Scientific Boundaries**: PRISM does not download OpenAI CLIP weights, call external LLM APIs, train chatbots, or scrape internet datasets. All alignment research is conducted within controlled, reproducible synthetic or benchmark settings.

---

## 15. Uncertainty, Calibration & Out-of-Distribution Representation Analysis Contracts

- **Predictive Confidence vs Correctness Distinction**: Model confidence (e.g. max softmax probability $\max_i p_i$) is an uncalibrated network output descriptor, NOT an objective probability of ground-truth correctness. Research documentation and reports must never equate high confidence with verified correctness.
- **Numerically Stable Softmax & Finite Probability Invariants**:
  $$p_i = \frac{\exp(z_i - \max(z))}{\sum_{j=1}^K \exp(z_j - \max(z))}$$
  All probabilities must satisfy $\sum_{i=1}^K p_i \approx 1.0$ within floating point tolerance ($| \sum p_i - 1.0 | < 10^{-4}$), $p_i \in [0.0, 1.0]$, and strictly reject non-finite (NaN/Inf) logits.
- **Predictive Shannon Entropy & Normalization**:
  $$H(p) = -\sum_{i=1}^K p_i \ln(p_i + \epsilon), \quad H_{\text{norm}}(p) = \frac{H(p)}{\ln(K)}$$
  Normalized entropy is bounded in $[0.0, 1.0]$, reaching 0.0 for deterministic one-hot distributions and 1.0 for uniform distributions over $K$ classes.
- **Reliability Diagram Binning & ECE / MCE Invariants**:
  - Equal-width binning partitions $[0, 1]$ into $B$ intervals $[l_b, u_b)$. Confidence $1.0$ is strictly mapped into the final bin.
  - Empty bins ($n_b = 0$) are preserved explicitly and do NOT contribute to Expected Calibration Error (ECE) or Maximum Calibration Error (MCE).
  - Expected Calibration Error must use sample-weighted bin gaps:
    $$\text{ECE} = \sum_{b=1}^B \frac{n_b}{N} |\text{acc}_b - \text{conf}_b|$$
    Unweighted bin averaging is strictly prohibited.
  - Multiclass Brier Score is evaluated as $\frac{1}{N}\sum_{n=1}^N \sum_{k=1}^K (p_{nk} - y_{nk})^2$ and Negative Log-Likelihood as $-\frac{1}{N}\sum_{n=1}^N \ln(p_{n, y_n} + \epsilon)$.
- **Temperature Scaling Optimization & Invariance**:
  - Scalar temperature $T^* > 0$ is optimized exclusively on held-out **validation** data to minimize validation NLL. Test partitions and test labels must NEVER be used to fit $T^*$.
  - Model weights, convolutional filters, and backbone representations remain strictly unchanged during temperature scaling.
  - **Argmax Class Invariance**: For any scalar $T > 0$, $\arg\max_k (z_k / T) = \arg\max_k (z_k)$. Classification accuracy before and after temperature scaling must be bitwise identical.
- **Out-of-Distribution (OOD) Scoring & Consistent Polarity**:
  - All normalized OOD scoring methods adhere to the polarity invariant: **higher score = more OOD-like**.
  - Maximum Softmax Probability: $\text{score}_{\text{MSP}} = 1.0 - \max_i p_i$.
  - Normalized Entropy: $\text{score}_{\text{Entropy}} = H_{\text{norm}}(p)$.
  - Class-Centroid Distance: $\text{score}_{\text{Centroid}} = \min_{c} d(\mathbf{h}, \boldsymbol{\mu}_c)$ with nearest class $\arg\min_c d(\mathbf{h}, \boldsymbol{\mu}_c)$.
  - Deterministic $k\text{NN}$ Distance: $\text{score}_{k\text{NN}} = \frac{1}{k}\sum_{j=1}^k d(\mathbf{h}, \mathbf{r}_j)$ using in-distribution reference vectors.
  - Free Energy Score: $\text{score}_{\text{Energy}} = -T \ln \sum_{i=1}^K \exp(z_i / T)$.
- **Exact AUROC & Threshold Selection Contracts**:
  - AUROC is evaluated using exact Mann-Whitney $U$ rank-sum integration with fractional average ranks for tied score pairs. Crude threshold approximation is prohibited.
  - Decision threshold selection ($\theta$) is calibrated on in-distribution validation/reference scores (e.g. target 95% ID TPR $\theta_{0.95}$). Tuning decision thresholds on test OOD labels is strictly prohibited.
- **Corruption Uncertainty Trajectories & Non-Monotonicity**:
  - Uncertainty metrics (accuracy, mean confidence, predictive entropy, ECE, representation drift, prediction flips) are tracked across corruption severities 1..5.
  - Confidence is NOT assumed to decrease monotonically with corruption severity. Non-monotonic overconfidence under corruption is explicitly measured and flagged as a diagnostic failure.
- **Representation Novelty vs Confidence Relationships**:
  - Pearson correlations between geometric distance (centroid / $k\text{NN}$) and predictive confidence are reported descriptively. Causation must NOT be claimed without interventional evidence.
- **Multimodal & Cosine Disclaimers**: Raw cosine similarity in vision-language models or representation geometry is a geometric alignment score, NOT a calibrated probability.
- **Scope & Non-Goals**: PRISM is not Bayesian deep learning (no Monte Carlo Dropout or Bayes-by-Backprop), not deep ensembles, and not production anomaly detection. All analyses investigate core deterministic representation properties.

### 12. Cross-Paradigm Benchmark Orchestration & Evidence Synthesis Standards
When synthesizing experimental results across learning paradigms (Phase 24):
- **Strict No-Fake-Result Policy**:
  Every cell in a benchmark matrix or table must carry an explicit scientific status (`OBSERVED`, `AGGREGATED`, `MISSING`, `FAILED`, `NOT_APPLICABLE`). Missing or unexecuted experiments must NEVER be imputed, interpolated, or defaulted to `0.0`.
- **Controlled Factor Auditing**:
  Pairwise comparisons and factor sweeps must verify that shared factors (`dataset`, `task`, `data_budget`, `seed`) are strictly aligned. Deviations must be flagged as `PARTIALLY_CONTROLLED`, `DESCRIPTIVE_ONLY`, or `INVALID_COMPARISON`.
- **Repeated Seed Requirement for Strong Claims**:
  Claims of statistical superiority or ranking require $N \ge 3$ random seeds with reported sample standard deviations. Single-seed observations must carry explicit `SINGLE_SEED_RESULT` caveats and are rated as `SUPPORTED_BY_SINGLE_RUN` or `DESCRIPTIVE_ONLY`.
- **Multi-Objective Tradeoffs**:
  Visual representation quality is fundamentally multi-dimensional (spanning semantic accuracy, robustness, calibration, efficiency, geometry, attribution, and multimodal alignment). Compressing representation quality into a single scalar rank is scientifically prohibited; tradeoffs must be analyzed via non-dominated Pareto frontiers.
- **Reproducibility Manifest**:
  Every synthesized research report must include a cryptographic `ReproducibilityManifest` documenting campaign fingerprints, experiment hashes, seed lists, environment metadata, and metric registry provenance.

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
