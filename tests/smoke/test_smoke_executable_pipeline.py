"""End-to-end smoke test for the Phase 5 executable dataset & batching pipeline.

Guarantees:
- Zero external network requests
- Zero GPU requirements (CPU-safe)
- Zero actual model training
"""

import pytest

from prism.core.enums import ModelFamily, OrderingStrategy, PrecisionMode, TaskType
from prism.data.batching import MaterializedBatch
from prism.data.context import DataRuntimeContext
from prism.data.manifests import ControlledDataReference, DatasetManifest
from prism.data.materialized import MaterializedDataset
from prism.data.preparer import DataPreparer
from prism.data.synthetic import SyntheticVisionAdapter
from prism.evaluation.configuration import (
    EvaluationConfiguration,
    MetricSpecification,
)
from prism.experiments.definitions import ExperimentDefinition
from prism.experiments.harness import ExperimentExecutionHarness
from prism.experiments.reproducibility import ReproducibilityConfiguration
from prism.models.specifications import ModelSpecification
from prism.training.configuration import (
    OptimizerSpecification,
    TrainingConfiguration,
)


@pytest.mark.smoke
def test_smoke_executable_pipeline_flow() -> None:
    """Demonstrate end-to-end executable data preparation and batch iteration."""
    # 1. Initialize Synthetic Vision Adapter
    adapter = SyntheticVisionAdapter(
        num_train=100, num_test=20, num_classes=2, image_shape=(3, 32, 32)
    )
    canonical = adapter.get_canonical_manifest()
    partition = adapter.get_default_partition(seed=42)
    subsets = adapter.get_nested_subsets(seed=42)

    # 2. Build ControlledDataReference for 10% low-data regime
    subset_10pct = subsets[0.10]
    controlled_ref = ControlledDataReference(
        canonical_manifest_fingerprint=canonical.compute_fingerprint(),
        partition_manifest_fingerprint=partition.compute_fingerprint(),
        subset_manifest_fingerprint=subset_10pct.compute_fingerprint(),
        partition_id=partition.partition_id,
        subset_id=subset_10pct.subset_id,
        budget_ratio=0.10,
    )

    dataset_manifest = DatasetManifest(
        **adapter.get_dataset_manifest().model_dump(exclude={"controlled_data"}),
        controlled_data=controlled_ref,
    )

    # 3. Assemble ExperimentDefinition
    model = ModelSpecification(
        model_id="model-synthetic-cnn",
        name="Synthetic Vision Model",
        family=ModelFamily.CNN,
        architecture="custom_cnn",
        compatible_tasks=[TaskType.CLASSIFICATION],
        num_classes=2,
    )
    training = TrainingConfiguration(
        epochs=5,
        batch_size=4,
        optimizer=OptimizerSpecification(type="adam", lr=1e-3),
        precision=PrecisionMode.FP32,
    )
    evaluation = EvaluationConfiguration(
        target_splits=["test"],
        metrics=[MetricSpecification(name="top1_accuracy")],
    )
    experiment = ExperimentDefinition(
        experiment_id="exp-synthetic-pipeline-smoke",
        name="Synthetic Executable Pipeline Smoke Test",
        task_type=TaskType.CLASSIFICATION,
        dataset=dataset_manifest,
        model=model,
        training=training,
        evaluation=evaluation,
        reproducibility=ReproducibilityConfiguration(seed=42),
    )

    # 4. Prepare execution runtime via Harness
    harness = ExperimentExecutionHarness()
    run, prepared_exec = harness.prepare(experiment)

    assert run.configuration_fingerprint == experiment.compute_fingerprint()

    # 5. Explicitly prepare executable dataset and batch loader
    preparer = DataPreparer()
    mat_dataset, batch_loader, data_context = preparer.prepare(
        adapter=adapter,
        canonical_manifest=canonical,
        partition_manifest=partition,
        subset_manifest=subset_10pct,
        batch_size=4,
        ordering_strategy=OrderingStrategy.EPOCH_AWARE_SHUFFLE,
        seed=42,
        epoch=0,
        drop_last=False,
        prepared_execution=prepared_exec,
    )

    assert isinstance(mat_dataset, MaterializedDataset)
    assert len(mat_dataset) == 8  # 10% of 80 train samples = 8 samples
    assert len(batch_loader) == 2  # 8 samples / batch_size 4 = 2 batches
    assert isinstance(data_context, DataRuntimeContext)

    # 6. Iterate through batches and verify sample traceability
    observed_sample_ids: list[str] = []
    for batch in batch_loader:
        assert isinstance(batch, MaterializedBatch)
        assert len(batch.sample_ids) == 4
        assert len(batch.data) == 4
        observed_sample_ids.extend(batch.sample_ids)

    assert len(observed_sample_ids) == 8
    assert set(observed_sample_ids) == set(subset_10pct.sample_ids)
