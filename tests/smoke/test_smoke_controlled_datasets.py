"""Smoke test demonstrating the Phase 4 controlled dataset & partition workflow.

NOTE: This is a pure contract and manifest validation verification.
No actual images are loaded, no model is trained, and no network access occurs.
"""

import pytest

from prism.core.enums import ModelFamily, PrecisionMode, RunStatus, TaskType
from prism.data.manifests import (
    ControlledDataReference,
    DatasetManifest,
    PreprocessingPolicy,
    SplitSpecification,
)
from prism.data.partitions import (
    PartitionManifest,
    generate_partition_manifest,
)
from prism.data.samples import CanonicalSampleManifest, SampleRecord
from prism.data.subsets import SubsetManifest, generate_nested_subsets
from prism.evaluation.configuration import (
    EvaluationConfiguration,
    MetricSpecification,
)
from prism.experiments.context import PreparedExecution
from prism.experiments.definitions import ExperimentDefinition
from prism.experiments.harness import ExperimentExecutionHarness
from prism.experiments.runs import ExperimentRun
from prism.models.specifications import ModelSpecification
from prism.training.configuration import (
    OptimizerSpecification,
    TrainingConfiguration,
)


@pytest.mark.smoke
def test_smoke_controlled_datasets_flow() -> None:
    """Demonstrate end-to-end controlled data manifests and preparation."""
    # 1. Construct canonical sample universe (1,000 synthetic samples)
    samples: list[SampleRecord] = []
    for i in range(800):
        samples.append(
            SampleRecord(
                sample_id=f"ds-synth/train/{i:06d}",
                source_split="train",
                source_index=i,
                target=i % 2,
                metadata={"class_name": f"class_{i % 2}"},
            )
        )
    for i in range(200):
        samples.append(
            SampleRecord(
                sample_id=f"ds-synth/test/{i:06d}",
                source_split="test",
                source_index=i,
                target=i % 2,
                metadata={"class_name": f"class_{i % 2}"},
            )
        )

    canonical = CanonicalSampleManifest.create(
        dataset_id="ds-synthetic-controlled",
        samples=samples,
        dataset_version="1.0.0",
        metadata={"created_for": "phase4_smoke_test"},
    )
    assert canonical.num_samples == 1000

    # 2. Generate deterministic stratified partition
    partition = generate_partition_manifest(
        canonical_manifest=canonical,
        split_ratios={"train": 0.875, "val": 0.125},  # 700 train, 100 val
        seed=42,
        strategy="stratified",
        source_split_filter="train",
        isolated_splits={"test": "test"},  # 200 isolated test
    )
    assert partition.total_samples == 1000
    assert partition.get_split("train").num_samples == 700
    assert partition.get_split("val").num_samples == 100
    assert partition.get_split("test").num_samples == 200

    # 3. Generate nested data-budget subsets
    budgets = (0.01, 0.05, 0.10, 0.25, 0.50, 1.00)
    subsets = generate_nested_subsets(
        partition_manifest=partition,
        canonical_manifest=canonical,
        budget_ratios=budgets,
        target_split="train",
        seed=42,
        strategy="nested_stratified",
    )

    # 4. Verify strict nesting property: S_1 ⊆ S_5 ⊆ S_10 ⊆ S_25 ⊆ S_50 ⊆ S_100
    s1 = set(subsets[0.01].sample_ids)
    s5 = set(subsets[0.05].sample_ids)
    s10 = set(subsets[0.10].sample_ids)
    s25 = set(subsets[0.25].sample_ids)
    s50 = set(subsets[0.50].sample_ids)
    s100 = set(subsets[1.00].sample_ids)

    assert s1.issubset(s5)
    assert s5.issubset(s10)
    assert s10.issubset(s25)
    assert s25.issubset(s50)
    assert s50.issubset(s100)
    assert s100 == set(partition.get_split("train").sample_ids)

    # 5. Verify serialization round-trips
    canonical_restored = CanonicalSampleManifest.from_json(canonical.to_json())
    assert canonical.compute_fingerprint() == canonical_restored.compute_fingerprint()

    partition_restored = PartitionManifest.from_json(partition.to_json())
    assert partition.compute_fingerprint() == partition_restored.compute_fingerprint()

    subset_restored = SubsetManifest.from_json(subsets[0.10].to_json())
    assert subsets[0.10].compute_fingerprint() == subset_restored.compute_fingerprint()

    # 6. Bind controlled-data reference to ExperimentDefinition
    controlled_ref = ControlledDataReference(
        canonical_manifest_fingerprint=canonical.compute_fingerprint(),
        partition_manifest_fingerprint=partition.compute_fingerprint(),
        subset_manifest_fingerprint=subsets[0.10].compute_fingerprint(),
        partition_id=partition.partition_id,
        subset_id=subsets[0.10].subset_id,
        budget_ratio=0.10,
    )

    dataset_manifest = DatasetManifest(
        dataset_id="ds-synthetic-controlled",
        name="Synthetic Controlled Vision Dataset",
        version="1.0.0",
        compatible_tasks=[TaskType.CLASSIFICATION],
        splits=[
            SplitSpecification(split_name="train", num_samples=70),
            SplitSpecification(split_name="val", num_samples=100),
            SplitSpecification(split_name="test", num_samples=200),
        ],
        classes=["class_0", "class_1"],
        num_classes=2,
        preprocessing=PreprocessingPolicy(resize=(32, 32)),
        controlled_data=controlled_ref,
    )

    model_spec = ModelSpecification(
        model_id="model-synthetic-cnn",
        name="Synthetic Classifier",
        family=ModelFamily.CNN,
        architecture="custom_cnn",
        compatible_tasks=[TaskType.CLASSIFICATION],
        num_classes=2,
    )

    training_cfg = TrainingConfiguration(
        epochs=5,
        batch_size=16,
        optimizer=OptimizerSpecification(type="adam", lr=1e-3),
        precision=PrecisionMode.FP32,
    )

    eval_cfg = EvaluationConfiguration(
        target_splits=["test"],
        metrics=[MetricSpecification(name="top1_accuracy")],
    )

    experiment = ExperimentDefinition(
        experiment_id="exp-synthetic-controlled-10pct",
        name="Synthetic Controlled 10% Low-Data Experiment",
        task_type=TaskType.CLASSIFICATION,
        dataset=dataset_manifest,
        model=model_spec,
        training=training_cfg,
        evaluation=eval_cfg,
    )

    # 7. Prepare execution via harness
    harness = ExperimentExecutionHarness()
    run, prepared = harness.prepare(experiment)

    assert isinstance(run, ExperimentRun)
    assert isinstance(prepared, PreparedExecution)
    assert run.status == RunStatus.PLANNED
    assert run.configuration_fingerprint == experiment.compute_fingerprint()
    assert prepared.configuration_fingerprint == experiment.compute_fingerprint()
