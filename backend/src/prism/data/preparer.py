"""Explicit dataset preparation and runtime context binding."""

from __future__ import annotations

from typing import Any

from prism.core.enums import OrderingStrategy
from prism.data.adapters import BenchmarkDatasetAdapter
from prism.data.batching import DeterministicBatchLoader
from prism.data.context import DataRuntimeContext
from prism.data.manifests import PreprocessingPolicy
from prism.data.materialized import MaterializedDataset
from prism.data.materializer import DatasetMaterializer
from prism.data.partitions import PartitionManifest
from prism.data.samples import CanonicalSampleManifest
from prism.data.subsets import SubsetManifest
from prism.experiments.context import PreparedExecution


class DataPreparer:
    """Explicitly orchestrates data materialization and batching preparation."""

    def __init__(self, materializer: DatasetMaterializer | None = None) -> None:
        self.materializer = materializer or DatasetMaterializer()

    def prepare(
        self,
        adapter: BenchmarkDatasetAdapter,
        canonical_manifest: CanonicalSampleManifest,
        partition_manifest: PartitionManifest | None = None,
        subset_manifest: SubsetManifest | None = None,
        split_name: str | None = None,
        batch_size: int = 32,
        ordering_strategy: OrderingStrategy | str = OrderingStrategy.SEQUENTIAL,
        seed: int | None = 42,
        epoch: int = 0,
        drop_last: bool = False,
        preprocessing_policy: PreprocessingPolicy | None = None,
        prepared_execution: PreparedExecution | None = None,
        backend_name: str = "in_memory",
        metadata: dict[str, Any] | None = None,
    ) -> tuple[MaterializedDataset, DeterministicBatchLoader, DataRuntimeContext]:
        """Materialize dataset and configure batch loader with runtime audit context."""
        # 1. Materialize executable dataset
        dataset = self.materializer.materialize(
            adapter=adapter,
            canonical_manifest=canonical_manifest,
            partition_manifest=partition_manifest,
            subset_manifest=subset_manifest,
            split_name=split_name,
            preprocessing_policy=preprocessing_policy,
        )

        # 2. Configure deterministic batch loader
        loader = DeterministicBatchLoader(
            dataset=dataset,
            batch_size=batch_size,
            ordering_strategy=ordering_strategy,
            seed=seed,
            epoch=epoch,
            drop_last=drop_last,
        )

        # 3. Assemble DataRuntimeContext
        canon_fp = canonical_manifest.compute_fingerprint()
        part_fp = (
            partition_manifest.compute_fingerprint() if partition_manifest else None
        )
        sub_fp = subset_manifest.compute_fingerprint() if subset_manifest else None
        ordering_fp = loader.get_ordering_fingerprint()

        strat_str = (
            ordering_strategy.value
            if isinstance(ordering_strategy, OrderingStrategy)
            else str(ordering_strategy)
        )

        context_metadata = dict(metadata or {})
        if prepared_execution is not None:
            context_metadata["experiment_id"] = prepared_execution.experiment_id
            context_metadata["run_id"] = prepared_execution.run_id
            context_metadata["configuration_fingerprint"] = (
                prepared_execution.configuration_fingerprint
            )

        runtime_context = DataRuntimeContext(
            dataset_id=adapter.dataset_id,
            canonical_manifest_fingerprint=canon_fp,
            partition_manifest_fingerprint=part_fp,
            subset_manifest_fingerprint=sub_fp,
            resolved_sample_count=len(dataset),
            ordering_strategy=strat_str,
            ordering_fingerprint=ordering_fp,
            batch_size=batch_size,
            drop_last=drop_last,
            backend_name=backend_name,
            metadata=context_metadata,
        )

        return dataset, loader, runtime_context
