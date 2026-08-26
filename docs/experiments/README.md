# PRISM Experiments Guide

## Purpose
This directory stores documentation, interactive notebooks, and synthesized research findings for PRISM experimental campaigns.

---

## Experiment Domain Concepts

PRISM separates the declarative research specification from physical execution attempts through strongly typed domain contracts:

```
ExperimentDefinition (Immutable scientific intent)
        │
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

## Defining and Preparing an Experiment

```python
from prism.core.enums import TaskType, ModelFamily, MetricDirection, PrecisionMode
from prism.data.manifests import (
    DatasetManifest,
    SplitSpecification,
    PreprocessingPolicy,
)
from prism.models.specifications import ModelSpecification
from prism.training.configuration import TrainingConfiguration, OptimizerSpecification
from prism.evaluation.configuration import EvaluationConfiguration, MetricSpecification
from prism.experiments.definitions import ExperimentDefinition
from prism.experiments.reproducibility import ReproducibilityConfiguration
from prism.experiments.harness import ExperimentExecutionHarness

# 1. Declare Dataset Manifest
dataset = DatasetManifest(
    dataset_id="ds-cifar10",
    name="CIFAR-10",
    compatible_tasks=[TaskType.CLASSIFICATION],
    splits=[
        SplitSpecification(split_name="train", num_samples=50000),
        SplitSpecification(split_name="test", num_samples=10000),
    ],
    num_classes=10,
    preprocessing=PreprocessingPolicy(resize=(32, 32)),
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
    experiment_id="exp-cifar10-resnet18",
    name="CIFAR-10 ResNet-18 Baseline",
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

# Inspect reproducibility report
report = prepared_context.get_reproducibility_report()
print(f"Fingerprint: {prepared_context.configuration_fingerprint}")
print(f"Seeded backends: {prepared_context.seeding_result.configured_backends}")
```

---

## Executing and Tracking a Run

```python
from prism.experiments.metrics import MetricRecord
from prism.artifacts.contracts import ArtifactReference
from prism.core.enums import ArtifactType

# Start planned run
run.start()

# Record telemetry during training/evaluation
run.add_metric(MetricRecord(metric_name="top1_accuracy", value=0.925, split="test"))

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
run.complete(summary_metrics={"top1_accuracy": 0.925})
```
