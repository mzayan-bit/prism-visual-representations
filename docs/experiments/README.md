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
        ├── Step Learning Rate Scheduler (Constant / Step / Cosine Annealing)
        ├── Record Real-time MetricRecords (Loss, Accuracy, Learning Rate) into ExperimentRun
        ├── Evaluate Test Partition via EvaluationEngine in Evaluation Mode
        └── Transition Run Lifecycle (RUNNING -> COMPLETED)
        │
        ▼
TrainingResult (Consolidated execution metrics and evaluation summaries)
        │
        ▼
extract_representations(...) (Extract intermediate spatial feature maps & vector embeddings)
```

---

## Defining, Preparing, and Training a Convolutional Neural Network (CNN) Experiment

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
from prism.models.cnn import ConvolutionalNeuralNetwork
from prism.models.specifications import ModelSpecification
from prism.training.configuration import (
    TrainingConfiguration,
    OptimizerSpecification,
    SchedulerSpecification,
)
from prism.training.engine import TrainingEngine
from prism.evaluation.configuration import EvaluationConfiguration, MetricSpecification
from prism.experiments.definitions import ExperimentDefinition
from prism.experiments.reproducibility import ReproducibilityConfiguration
from prism.experiments.harness import ExperimentExecutionHarness
from prism.representations.contracts import (
    RepresentationDescriptor,
    RepresentationBatch,
)

# 1. Obtain standardized CIFAR-10 manifests & 10% nested subset
adapter = CIFAR10Adapter()
canonical = adapter.get_canonical_manifest()
partition = adapter.get_default_partition(seed=42)
subsets = adapter.get_nested_subsets(seed=42)

# Bind 10% data-budget subset
subset_10pct = subsets[0.10]
controlled_ref = ControlledDataReference(
    canonical_manifest_fingerprint=canonical.compute_fingerprint(),
    partition_manifest_fingerprint=partition.compute_fingerprint(),
    subset_manifest_fingerprint=subset_10pct.compute_fingerprint(),
    partition_id=partition.partition_id,
    subset_id=subset_10pct.subset_id,
    budget_ratio=0.10,
)

dataset = DatasetManifest(
    **adapter.get_dataset_manifest().model_dump(exclude={"controlled_data"}),
    controlled_data=controlled_ref,
)

# 2. Declare Model Architecture (2-Block Spatial CNN)
model = ModelSpecification(
    model_id="model-cifar10-cnn",
    name="CIFAR-10 2-Block ConvNet",
    family=ModelFamily.CNN,
    architecture="cnn",
    compatible_tasks=[TaskType.CLASSIFICATION],
    input_shape=(3, 32, 32),
    num_classes=10,
    hyperparameters={
        "conv_channels": [32, 64],
        "kernel_sizes": 3,
        "strides": 1,
        "paddings": 1,
        "pool_sizes": 2,
        "pool_strides": 2,
        "activation": "relu",
        "dropout": 0.1,
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
    experiment_id="exp-cifar10-cnn-10pct",
    name="CIFAR-10 CNN 10% Low-Data Baseline",
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
    subset_manifest=subset_10pct,
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

# 8. Extract Spatial Feature Maps and Final Vector Representations
cnn_model = ConvolutionalNeuralNetwork(spec=model, seed=42)
test_batch = [test_dataset[i].data for i in range(10)]

# Extract intermediate spatial feature maps: [10, 64, 8, 8]
spatial_maps = cnn_model.extract_representations(test_batch, layer="final_spatial")

# Extract final flattened vector representations: [10, 4096]
vector_reps = cnn_model.extract_representations(test_batch, layer="final_hidden")

print(f"Run Status: {result.status}")
print(f"Final Train Loss: {result.final_train_loss:.4f}")
print(f"Test Accuracy: {result.summary_metrics.get('test_top1_accuracy', 0.0):.4f}")
print(
    f"Final Spatial Feature Map Shape: {len(spatial_maps)}x{len(spatial_maps[0])}x{len(spatial_maps[0][0])}x{len(spatial_maps[0][0][0])}"
)
print(f"Final Vector Representation Shape: {len(vector_reps)}x{len(vector_reps[0])}")
print(f"Receptive Field: {cnn_model.receptive_field} pixels")
```
