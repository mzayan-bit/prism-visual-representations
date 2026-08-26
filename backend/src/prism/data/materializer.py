"""Dataset materialization layer resolving sample identities into datasets."""

from collections.abc import Callable
from typing import Any

from prism.core.errors import (
    DatasetMaterializationError,
    SampleResolutionError,
    ValidationError,
)
from prism.data.adapters import BenchmarkDatasetAdapter
from prism.data.manifests import PreprocessingPolicy
from prism.data.materialized import MaterializedDataset, MaterializedSample
from prism.data.partitions import PartitionManifest
from prism.data.preprocessing import create_executable_preprocessing
from prism.data.samples import CanonicalSampleManifest
from prism.data.subsets import SubsetManifest


class DatasetMaterializer:
    """Orchestrates resolution of manifest identities into datasets."""

    def materialize(
        self,
        adapter: BenchmarkDatasetAdapter,
        canonical_manifest: CanonicalSampleManifest,
        partition_manifest: PartitionManifest | None = None,
        subset_manifest: SubsetManifest | None = None,
        split_name: str | None = None,
        preprocessing_policy: PreprocessingPolicy | None = None,
        custom_transform: Callable[[Any], Any] | None = None,
    ) -> MaterializedDataset:
        """Resolve exact sample identities into a MaterializedDataset."""
        if adapter.dataset_id != canonical_manifest.dataset_id:
            raise DatasetMaterializationError(
                f"Adapter dataset_id '{adapter.dataset_id}' does not match "
                f"canonical manifest dataset_id '{canonical_manifest.dataset_id}'."
            )

        # 1. Determine the exact ordered sequence of sample IDs to materialize
        if subset_manifest is not None:
            if (
                partition_manifest is not None
                and subset_manifest.partition_fingerprint
                != partition_manifest.compute_fingerprint()
            ):
                raise ValidationError(
                    f"Subset fingerprint '{subset_manifest.partition_fingerprint}' "
                    f"does not match partition manifest fingerprint."
                )
            target_sample_ids = subset_manifest.sample_ids
            active_split = subset_manifest.target_split
        elif partition_manifest is not None:
            partition_manifest.validate_against_canonical(canonical_manifest)
            target_split_name = split_name or "train"
            split_obj = partition_manifest.get_split(target_split_name)
            target_sample_ids = split_obj.sample_ids
            active_split = target_split_name
        elif split_name is not None:
            records = canonical_manifest.filter_by_source_split(split_name)
            target_sample_ids = [r.sample_id for r in records]
            active_split = split_name
        else:
            target_sample_ids = [s.sample_id for s in canonical_manifest.samples]
            active_split = None

        if not target_sample_ids:
            raise DatasetMaterializationError(
                f"No sample IDs resolved for materialization with split='{split_name}'."
            )

        # 2. Build executable transform pipeline
        executable_prep = (
            create_executable_preprocessing(preprocessing_policy)
            if preprocessing_policy
            else None
        )

        def combined_transform(data: Any) -> Any:
            res = executable_prep(data) if executable_prep else data
            return custom_transform(res) if custom_transform else res

        # 3. Resolve samples against adapter
        resolved_samples: list[MaterializedSample] = []
        for sid in target_sample_ids:
            try:
                record = canonical_manifest.get_sample(sid)
            except KeyError as exc:
                raise SampleResolutionError(
                    f"Sample '{sid}' does not exist in canonical manifest."
                ) from exc

            sample = adapter.resolve_sample(sid)

            # Validate sample integrity against manifest
            if sample.sample_id != record.sample_id:
                raise SampleResolutionError(
                    f"Resolved sample ID '{sample.sample_id}' "
                    f"does not match manifest ID '{record.sample_id}'."
                )
            if sample.source_index != record.source_index:
                raise SampleResolutionError(
                    f"Resolved source index {sample.source_index} does not match "
                    f"manifest index {record.source_index} for '{sid}'."
                )
            if sample.target != record.target:
                raise SampleResolutionError(
                    f"Resolved target '{sample.target}' does not match "
                    f"manifest target '{record.target}' for '{sid}'."
                )

            resolved_samples.append(sample)

        canon_fp = canonical_manifest.compute_fingerprint()
        part_fp = (
            partition_manifest.compute_fingerprint() if partition_manifest else None
        )
        sub_fp = subset_manifest.compute_fingerprint() if subset_manifest else None

        return MaterializedDataset(
            dataset_id=adapter.dataset_id,
            samples=resolved_samples,
            split_name=active_split,
            transform=combined_transform,
            metadata={
                "canonical_manifest_fingerprint": canon_fp,
                "partition_fingerprint": part_fp,
                "subset_fingerprint": sub_fp,
                "num_materialized_samples": len(resolved_samples),
            },
        )
