"""Synthetic in-memory dataset adapter for fast, deterministic testing."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from prism.core.enums import TaskType
from prism.data.adapters import BenchmarkDatasetAdapter
from prism.data.manifests import (
    AugmentationPolicy,
    DatasetManifest,
    PreprocessingPolicy,
    SplitSpecification,
)
from prism.data.materialized import MaterializedDataset, MaterializedSample
from prism.data.partitions import (
    PartitionManifest,
    generate_partition_manifest,
)
from prism.data.samples import CanonicalSampleManifest, SampleRecord
from prism.data.subsets import (
    DEFAULT_DATA_BUDGETS,
    SubsetManifest,
    generate_nested_subsets,
)


class SyntheticVisionAdapter(BenchmarkDatasetAdapter):
    """Deterministic synthetic in-memory dataset adapter for testing."""

    def __init__(
        self,
        dataset_id: str = "ds-synthetic-vision",
        name: str = "Synthetic Vision Dataset",
        num_train: int = 100,
        num_test: int = 20,
        num_classes: int = 2,
        image_shape: tuple[int, int, int] = (3, 32, 32),
    ) -> None:
        self._dataset_id = dataset_id
        self._name = name
        self._num_train = num_train
        self._num_test = num_test
        self._num_classes = num_classes
        self._image_shape = image_shape
        self._classes = [f"class_{i}" for i in range(num_classes)]

    @property
    def dataset_id(self) -> str:
        return self._dataset_id

    @property
    def name(self) -> str:
        return self._name

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def classes(self) -> list[str]:
        return list(self._classes)

    def get_dataset_manifest(self) -> DatasetManifest:
        return DatasetManifest(
            dataset_id=self.dataset_id,
            name=self.name,
            description="Synthetic in-memory vision benchmark for testing",
            version="1.0.0",
            compatible_tasks=[
                TaskType.CLASSIFICATION,
                TaskType.REPRESENTATION_LEARNING,
            ],
            splits=[
                SplitSpecification(split_name="train", num_samples=self._num_train),
                SplitSpecification(split_name="test", num_samples=self._num_test),
            ],
            classes=self.classes,
            num_classes=self.num_classes,
            preprocessing=PreprocessingPolicy(
                resize=(self._image_shape[1], self._image_shape[2]),
                normalization_mean=(0.5, 0.5, 0.5),
                normalization_std=(0.5, 0.5, 0.5),
                color_space="RGB",
            ),
            augmentation=AugmentationPolicy(name="none", enabled=False),
        )

    def get_canonical_manifest(self) -> CanonicalSampleManifest:
        samples: list[SampleRecord] = []

        for idx in range(self._num_train):
            target = idx % self._num_classes
            samples.append(
                SampleRecord(
                    sample_id=f"{self.dataset_id}/train/{idx:06d}",
                    source_split="train",
                    source_index=idx,
                    target=target,
                    metadata={"class_name": self._classes[target]},
                )
            )

        for idx in range(self._num_test):
            target = idx % self._num_classes
            samples.append(
                SampleRecord(
                    sample_id=f"{self.dataset_id}/test/{idx:06d}",
                    source_split="test",
                    source_index=idx,
                    target=target,
                    metadata={"class_name": self._classes[target]},
                )
            )

        return CanonicalSampleManifest.create(
            dataset_id=self.dataset_id,
            samples=samples,
            dataset_version="1.0.0",
            metadata={"source": "synthetic_in_memory"},
        )

    def get_default_partition(
        self,
        seed: int = 42,
        val_ratio: float = 0.2,
    ) -> PartitionManifest:
        canonical = self.get_canonical_manifest()
        train_ratio = 1.0 - val_ratio
        return generate_partition_manifest(
            canonical_manifest=canonical,
            split_ratios={"train": train_ratio, "val": val_ratio},
            seed=seed,
            strategy="stratified",
            partition_id=f"part-{self.dataset_id}-s{seed}",
            source_split_filter="train",
            isolated_splits={"test": "test"},
        )

    def get_nested_subsets(
        self,
        budget_ratios: Sequence[float] | None = None,
        seed: int = 42,
        val_ratio: float = 0.2,
    ) -> dict[float, SubsetManifest]:
        canonical = self.get_canonical_manifest()
        partition = self.get_default_partition(seed=seed, val_ratio=val_ratio)
        budgets = budget_ratios if budget_ratios is not None else DEFAULT_DATA_BUDGETS
        return generate_nested_subsets(
            partition_manifest=partition,
            canonical_manifest=canonical,
            budget_ratios=budgets,
            target_split="train",
            seed=seed,
            strategy="nested_stratified",
        )

    def is_available_locally(self, root: Path | str | None = None) -> bool:
        return True

    def resolve_sample(self, sample_id: str) -> MaterializedSample:
        """Deterministically generate the payload for a synthetic sample ID."""
        canonical = self.get_canonical_manifest()
        record = canonical.get_sample(sample_id)

        c, h, w = self._image_shape
        synthetic_payload = [
            [
                [(float((record.source_index + ch) % 255) / 255.0) for _ in range(w)]
                for _ in range(h)
            ]
            for ch in range(c)
        ]

        return MaterializedSample(
            sample_id=record.sample_id,
            source_split=record.source_split,
            source_index=record.source_index,
            data=synthetic_payload,
            target=record.target,
            metadata=record.metadata,
        )

    def load_raw_dataset(
        self,
        split: str = "train",
        root: Path | str | None = None,
        download: bool = False,
        transform: Any = None,
    ) -> MaterializedDataset:
        canonical = self.get_canonical_manifest()
        samples = canonical.filter_by_source_split(split)
        materialized = [self.resolve_sample(s.sample_id) for s in samples]
        return MaterializedDataset(
            dataset_id=self.dataset_id,
            samples=materialized,
            split_name=split,
            transform=transform,
        )
