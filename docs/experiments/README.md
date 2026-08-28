# PRISM Experiments Guide

## Purpose
This directory stores documentation, interactive notebooks, and synthesized research findings for PRISM experimental campaigns.

---

## Experiment Domain Concepts

PRISM separates the declarative research specification from physical execution attempts through strongly typed domain contracts:

```
ExperimentDefinition (Immutable scientific intent)
        │
        ├── Bound ControlledDataReference (Canonical universe + Partition + Nested subset)
        ├── Validate Definition & Compute SHA-256 Fingerprint
        │
        ▼
ExperimentExecutionHarness.prepare(experiment)
        │
        ├── Probe Host Hardware (CPU, CUDA, MPS)
        ├── Capture Environment Snapshot & Git Revision
        ├── Initialize Multi-Backend RNG (Python, NumPy, PyTorch)
        └── Output Immutable PreparedExecution Context
        │
        ▼
DataPreparer.prepare(prepared_execution, ...)
        │
        ├── Resolve Exact Canonical Sample Identities
        ├── Execute Deterministic Preprocessing
        ├── Bind Deterministic Ordering (Sequential / Fixed Shuffle / Epoch-Aware)
        └── Output MaterializedDataset + DeterministicBatchLoader + DataRuntimeContext
        │
        ▼
TrainingEngine.train(...)
        │
        ├── Execute Deterministic Epoch Loops (Forward -> Loss -> Backward -> SGD)
        ├── Route Analytical Gradients through Residual Skip Branches (dF + dS)
        ├── Update BatchNorm Running Statistics during training
        ├── Step Learning Rate Scheduler (Constant / Step / Cosine Annealing)
        ├── Record Real-time MetricRecords (Loss, Accuracy, Learning Rate) into ExperimentRun
        ├── Evaluate Test Partition via EvaluationEngine in Evaluation Mode
        └── Transition Run Lifecycle (RUNNING -> COMPLETED)
        │
        ▼
TrainingResult (Consolidated execution metrics and evaluation summaries)
        │
        ▼
compute_gradient_flow_summary & extract_representations (Gradient tracking & feature analysis)
```

---

## Defining, Preparing, and Training a Residual CNN Experiment

```python
from prism.core.enums import (
    TaskType,
    ModelFamily,
    MetricDirection,
    PrecisionMode,
    OrderingStrategy,
)
from prism.data.adapters import CIFAR10Adapter
from prism.data.manifests import ControlledDataReference, DatasetManifest
from prism.data.preparer import DataPreparer
from prism.models.resnet import ResidualNeuralNetwork
from prism.models.specifications import ModelSpecification
from prism.training.configuration import (
    TrainingConfiguration,
    OptimizerSpecification,
    SchedulerSpecification,
)
from prism.training.engine import TrainingEngine
from prism.training.gradient_flow import (
    compute_gradient_flow_summary,
    compare_gradient_flow_summaries,
)
from prism.evaluation.configuration import EvaluationConfiguration, MetricSpecification
from prism.experiments.definitions import ExperimentDefinition
from prism.experiments.reproducibility import ReproducibilityConfiguration
from prism.experiments.harness import ExperimentExecutionHarness
from prism.experiments.comparisons import create_residual_comparison

# 1. Obtain standardized CIFAR-10 manifests & 25% nested subset
adapter = CIFAR10Adapter()
canonical = adapter.get_canonical_manifest()
partition = adapter.get_default_partition(seed=42)
subsets = adapter.get_nested_subsets(seed=42)

subset_25pct = subsets[0.25]
controlled_ref = ControlledDataReference(
    canonical_manifest_fingerprint=canonical.compute_fingerprint(),
    partition_manifest_fingerprint=partition.compute_fingerprint(),
    subset_manifest_fingerprint=subset_25pct.compute_fingerprint(),
    partition_id=partition.partition_id,
    subset_id=subset_25pct.subset_id,
    budget_ratio=0.25,
)

dataset = DatasetManifest(
    **adapter.get_dataset_manifest().model_dump(exclude={"controlled_data"}),
    controlled_data=controlled_ref,
)

# 2. Declare Model Architecture (Multi-Stage ResNet with BatchNorm)
model = ModelSpecification(
    model_id="model-cifar10-resnet-bn",
    name="CIFAR-10 ResNet with Skip Connections",
    family=ModelFamily.RESNET,
    architecture="resnet",
    compatible_tasks=[TaskType.CLASSIFICATION],
    input_shape=(3, 32, 32),
    num_classes=10,
    hyperparameters={
        "stem_channels": 16,
        "stage_widths": [16, 32, 64],
        "blocks_per_stage": [2, 2, 2],
        "strides": [1, 2, 2],
        "activation": "relu",
        "normalization": "batch_norm",
        "norm_eps": 1e-5,
        "norm_momentum": 0.1,
        "norm_affine": True,
        "dropout": 0.0,
    },
)

# 3. Declare Training & Evaluation Budget with Cosine Annealing Schedule
training = TrainingConfiguration(
    epochs=50,
    batch_size=64,
    optimizer=OptimizerSpecification(
        type="sgd", lr=0.05, momentum=0.9, weight_decay=1e-4
    ),
    scheduler=SchedulerSpecification(type="cosine", min_lr=0.001, warmup_epochs=5),
    precision=PrecisionMode.FP32,
)

evaluation = EvaluationConfiguration(
    target_splits=["test"],
    metrics=[
        MetricSpecification(name="top1_accuracy", direction=MetricDirection.MAXIMIZE),
        MetricSpecification(name="loss", direction=MetricDirection.MINIMIZE),
    ],
)

# 4. Construct Immutable Experiment Definition
experiment = ExperimentDefinition(
    experiment_id="exp-cifar10-resnet-25pct",
    name="CIFAR-10 Residual CNN 25% Data Efficiency",
    task_type=TaskType.CLASSIFICATION,
    dataset=dataset,
    model=model,
    training=training,
    evaluation=evaluation,
    reproducibility=ReproducibilityConfiguration(seed=42, deterministic=True),
)

# 5. Prepare Execution Context via Harness
harness = ExperimentExecutionHarness()
run, prepared_context = harness.prepare(experiment)

# 6. Materialize Train and Test Datasets
preparer = DataPreparer()
train_dataset, train_loader, _ = preparer.prepare(
    adapter=adapter,
    canonical_manifest=canonical,
    partition_manifest=partition,
    subset_manifest=subset_25pct,
    batch_size=64,
    ordering_strategy=OrderingStrategy.EPOCH_AWARE_SHUFFLE,
    seed=42,
    prepared_execution=prepared_context,
)

test_dataset, test_loader, _ = preparer.prepare(
    adapter=adapter,
    canonical_manifest=canonical,
    partition_manifest=partition,
    split_name="test",
    batch_size=100,
    ordering_strategy=OrderingStrategy.SEQUENTIAL,
    seed=42,
    prepared_execution=prepared_context,
)

# 7. Execute Training and Evaluation Loop
engine = TrainingEngine()
result = engine.train(
    experiment=experiment,
    prepared_execution=prepared_context,
    train_dataset=train_dataset,
    train_loader=train_loader,
    test_dataset=test_dataset,
    test_loader=test_loader,
    run=run,
)

# 8. Compute Gradient Flow Summaries across Model Depth
res_model = ResidualNeuralNetwork(spec=model, seed=42)
test_batch = [test_dataset[i].data for i in range(10)]
_ = res_model.forward(test_batch)
res_model.backward(
    [
        [1.0 / 10 if j == test_dataset[i].target else 0.0 for j in range(10)]
        for i in range(10)
    ]
)

grad_summary = compute_gradient_flow_summary(res_model)
print(f"Global Gradient L2 Norm: {grad_summary.global_grad_norm_l2:.6f}")
for param_s in grad_summary.parameter_summaries[:5]:
    print(
        f"Layer {param_s.parameter_name} ({param_s.logical_stage}): Norm={param_s.norm_l2:.6f}"
    )
```

---

## Controlled Learning Rate Schedule Comparison

```python
from prism.experiments.comparisons import create_scheduler_comparison

# Declaratively isolate learning rate schedule effect (Constant vs Cosine Annealing + Warmup)
comparison = create_scheduler_comparison(
    comparison_id="comp-lr-constant-vs-warmup-cosine",
    name="Constant vs Warmup-Cosine Schedule on ResNet",
    baseline_experiment_id="exp-cifar10-resnet-constant-lr",
    candidate_experiment_id="exp-cifar10-resnet-warmup-cosine-lr",
    baseline_scheduler_type="constant",
    candidate_scheduler_type="cosine",
    baseline_scheduler_params={"min_lr": 0.0},
    candidate_scheduler_params={"warmup_epochs": 5, "min_lr": 0.001},
    dataset_fingerprint=dataset.compute_fingerprint(),
    seed=42,
    description="Controlled study of warmup and cosine annealing on convergence speed and representation geometry.",
)

print(f"Comparison Fingerprint: {comparison.compute_fingerprint()}")
```

---

## Vision Transformer Foundations & Attention Analysis

```python
from prism.models.patches import (
    PatchExtractor,
    PatchEmbedding,
    ClassToken,
    PositionalEmbedding,
)
from prism.models.attention import MultiHeadSelfAttention
from prism.representations.attention import summarize_attention_weights

# 1. Extract 4x4 non-overlapping patches from 32x32 image (64 patches of dim 48)
patch_ext = PatchExtractor(patch_size=4)
patches = patch_ext.forward(image_batch)  # [N, 64, 48]

# 2. Linear projection to embedding dimension (48 -> 128)
patch_emb = PatchEmbedding(in_features=48, embed_dim=128, seed=42)
tokens = patch_emb.forward(patches)  # [N, 64, 128]

# 3. Prepend learnable classification token [1, 1, 128] -> 65 tokens
cls_token = ClassToken(embed_dim=128, seed=42)
tokens_with_cls = cls_token.forward(tokens)  # [N, 65, 128]

# 4. Add learnable 1D position embeddings
pos_emb = PositionalEmbedding(num_positions=65, embed_dim=128, seed=42)
embedded_tokens = pos_emb.forward(tokens_with_cls)  # [N, 65, 128]

# 5. Multi-Head Self-Attention (128 dim, 4 heads -> 32 dim per head)
mhsa = MultiHeadSelfAttention(embed_dim=128, num_heads=4, seed=42)
contextualized_tokens = mhsa.forward(embedded_tokens)  # [N, 65, 128]

# 6. Audit attention distributions across heads
attn_summary = summarize_attention_weights(mhsa.last_attention_weights)
print(f"Mean Attention Entropy: {attn_summary.mean_entropy:.4f} nats")
print(f"Row Normalized: {attn_summary.is_row_normalized}")
for head in attn_summary.head_summaries:
    print(
        f"Head {head.head_index}: entropy={head.entropy:.4f}, min={head.min_value:.4f}, max={head.max_value:.4f}"
    )
```

