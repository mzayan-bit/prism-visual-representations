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
ExperimentRun (Execution instance & lifecycle state machine)
        │
        ├── [PLANNED] ➔ [RUNNING] ➔ [COMPLETED / FAILED / CANCELLED]
        │
        ├── Record MetricRecords & Register ArtifactReferences
        │
        ▼
EvaluationReport (Immutable compiled evaluation summary)
```

---

## Defining and Preparing a Controlled Experiment

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
from prism.models.specifications import ModelSpecification
from prism.training.configuration import TrainingConfiguration, OptimizerSpecification
from prism.evaluation.configuration import EvaluationConfiguration, MetricSpecification
from prism.experiments.definitions import ExperimentDefinition
from prism.experiments.reproducibility import ReproducibilityConfiguration
from prism.experiments.harness import ExperimentExecutionHarness

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

# 2. Declare Model Architecture
model = ModelSpecification(
    model_id="model-resnet18",
    name="ResNet-18",
    family=ModelFamily.RESNET,
    architecture="resnet18",
    compatible_tasks=[TaskType.CLASSIFICATION],
    num_classes=10,
)

# 3. Declare Training & Evaluation Budget
training = TrainingConfiguration(
    epochs=100,
    batch_size=128,
    optimizer=OptimizerSpecification(type="adamw", lr=1e-3),
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
    experiment_id="exp-cifar10-resnet18-10pct",
    name="CIFAR-10 ResNet-18 10% Low-Data Regime",
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

# 6. Explicitly Prepare Materialized Data and Batch Loader
preparer = DataPreparer()
mat_dataset, batch_loader, data_context = preparer.prepare(
    adapter=adapter,
    canonical_manifest=canonical,
    partition_manifest=partition,
    subset_manifest=subset_10pct,
    batch_size=128,
    ordering_strategy=OrderingStrategy.EPOCH_AWARE_SHUFFLE,
    seed=42,
    epoch=0,
    prepared_execution=prepared_context,
)

print(f"Materialized samples: {len(mat_dataset)}")
print(f"Batches per epoch: {len(batch_loader)}")
print(f"Ordering fingerprint: {data_context.ordering_fingerprint}")
```

---

## Executing and Tracking a Run

```python
from prism.experiments.metrics import MetricRecord
from prism.artifacts.contracts import ArtifactReference
from prism.core.enums import ArtifactType

# Start planned run
run.start()

# Iterate through traceable batches
for batch in batch_loader:
    # batch.sample_ids contains traceable sample identities
    # batch.data contains preprocessed payload
    pass

# Record telemetry during training/evaluation
run.add_metric(MetricRecord(metric_name="top1_accuracy", value=0.885, split="test"))

# Register produced artifacts
run.add_artifact(
    ArtifactReference(
        artifact_id="art-checkpoint-best",
        artifact_type=ArtifactType.CHECKPOINT,
        logical_name="best_checkpoint",
        uri="artifacts/checkpoints/best.pt",
        producing_run_id=run.run_id,
    )
)

# Complete run
run.complete(summary_metrics={"top1_accuracy": 0.885})
```
