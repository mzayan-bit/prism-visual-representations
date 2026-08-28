# PRISM Research Contract

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
