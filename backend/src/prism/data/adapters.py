"""Benchmark dataset adapters for controlled computer vision experiments."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from prism.core.enums import TaskType
from prism.core.errors import ConfigurationError
from prism.data.manifests import (
    AugmentationPolicy,
    DatasetManifest,
    PreprocessingPolicy,
    SplitSpecification,
)
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

CIFAR10_CLASSES: list[str] = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

CIFAR100_CLASSES: list[str] = [
    "apple",
    "aquarium_fish",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    "cattle",
    "chair",
    "chimpanzee",
    "clock",
    "cloud",
    "cockroach",
    "couch",
    "crab",
    "crocodile",
    "cup",
    "dinosaur",
    "dolphin",
    "elephant",
    "flatfish",
    "forest",
    "fox",
    "girl",
    "hamster",
    "house",
    "kangaroo",
    "keyboard",
    "lamp",
    "lawn_mower",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "man",
    "maple_tree",
    "motorcycle",
    "mountain",
    "mouse",
    "mushroom",
    "oak_tree",
    "orange",
    "orchid",
    "otter",
    "palm_tree",
    "pear",
    "pickup_truck",
    "pine_tree",
    "plain",
    "plate",
    "poppy",
    "porcupine",
    "possum",
    "rabbit",
    "raccoon",
    "ray",
    "road",
    "rocket",
    "rose",
    "sea",
    "seal",
    "shark",
    "shrew",
    "skunk",
    "skyscraper",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "streetcar",
    "sunflower",
    "sweet_pepper",
    "table",
    "tank",
    "telephone",
    "television",
    "tiger",
    "tractor",
    "train",
    "trout",
    "tulip",
    "turtle",
    "wardrobe",
    "whale",
    "willow_tree",
    "wolf",
    "woman",
    "worm",
]


def _load_torchvision_dataset(
    dataset_name: str,
    root: str,
    train: bool,
    download: bool,
    transform: Any,
) -> Any:
    """Helper to safely load torchvision datasets with clean error handling."""
    try:
        import torchvision  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ConfigurationError(
            f"torchvision is required to load raw {dataset_name} datasets. "
            "Install torchvision or prism[cv]."
        ) from exc

    if dataset_name == "CIFAR10":
        return torchvision.datasets.CIFAR10(
            root=root,
            train=train,
            download=download,
            transform=transform,
        )
    if dataset_name == "CIFAR100":
        return torchvision.datasets.CIFAR100(
            root=root,
            train=train,
            download=download,
            transform=transform,
        )
    raise ConfigurationError(f"Unsupported benchmark dataset '{dataset_name}'.")


class BenchmarkDatasetAdapter(ABC):
    """Abstract contract for standardized vision benchmark dataset adapters."""

    @property
    @abstractmethod
    def dataset_id(self) -> str:
        """Unique dataset identifier (e.g. 'ds-cifar10')."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable display name."""
        ...

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Total category class count."""
        ...

    @property
    @abstractmethod
    def classes(self) -> list[str]:
        """Ordered list of class names."""
        ...

    @abstractmethod
    def get_dataset_manifest(self) -> DatasetManifest:
        """Return the declarative DatasetManifest contract."""
        ...

    @abstractmethod
    def get_canonical_manifest(self) -> CanonicalSampleManifest:
        """Generate or retrieve the canonical sample universe manifest."""
        ...

    @abstractmethod
    def get_default_partition(
        self,
        seed: int = 42,
        val_ratio: float = 0.1,
    ) -> PartitionManifest:
        """Generate standard PRISM benchmark partition."""
        ...

    @abstractmethod
    def get_nested_subsets(
        self,
        budget_ratios: Sequence[float] | None = None,
        seed: int = 42,
        val_ratio: float = 0.1,
    ) -> dict[float, SubsetManifest]:
        """Generate standard nested data-budget subsets."""
        ...

    @abstractmethod
    def is_available_locally(self, root: Path | str | None = None) -> bool:
        """Check whether dataset raw files exist locally."""
        ...

    @abstractmethod
    def load_raw_dataset(
        self,
        split: str = "train",
        root: Path | str | None = None,
        download: bool = False,
        transform: Any = None,
    ) -> Any:
        """Load raw dataset instance via optional provider library."""
        ...


class CIFAR10Adapter(BenchmarkDatasetAdapter):
    """Adapter for the CIFAR-10 classification benchmark."""

    @property
    def dataset_id(self) -> str:
        return "ds-cifar10"

    @property
    def name(self) -> str:
        return "CIFAR-10"

    @property
    def num_classes(self) -> int:
        return 10

    @property
    def classes(self) -> list[str]:
        return list(CIFAR10_CLASSES)

    def get_dataset_manifest(self) -> DatasetManifest:
        return DatasetManifest(
            dataset_id=self.dataset_id,
            name=self.name,
            description="CIFAR-10 classification benchmark (60k 32x32 images)",
            source_uri="https://www.cs.toronto.edu/~kriz/cifar.html",
            version="1.0.0",
            compatible_tasks=[
                TaskType.CLASSIFICATION,
                TaskType.REPRESENTATION_LEARNING,
            ],
            splits=[
                SplitSpecification(split_name="train", num_samples=50000),
                SplitSpecification(split_name="test", num_samples=10000),
            ],
            classes=self.classes,
            num_classes=self.num_classes,
            preprocessing=PreprocessingPolicy(
                resize=(32, 32),
                normalization_mean=(0.4914, 0.4822, 0.4465),
                normalization_std=(0.2470, 0.2435, 0.2616),
                color_space="RGB",
            ),
            augmentation=AugmentationPolicy(name="none", enabled=False),
        )

    def get_canonical_manifest(self) -> CanonicalSampleManifest:
        """Build canonical sample universe for CIFAR-10 (50k train + 10k test)."""
        samples: list[SampleRecord] = []

        # Canonical training partition: 50,000 samples (5,000 per class)
        for idx in range(50000):
            target = idx % 10
            samples.append(
                SampleRecord(
                    sample_id=f"cifar10/train/{idx:06d}",
                    source_split="train",
                    source_index=idx,
                    target=target,
                    metadata={"class_name": self.classes[target]},
                )
            )

        # Canonical test partition: 10,000 samples (1,000 per class)
        for idx in range(10000):
            target = idx % 10
            samples.append(
                SampleRecord(
                    sample_id=f"cifar10/test/{idx:06d}",
                    source_split="test",
                    source_index=idx,
                    target=target,
                    metadata={"class_name": self.classes[target]},
                )
            )

        return CanonicalSampleManifest.create(
            dataset_id=self.dataset_id,
            samples=samples,
            dataset_version="1.0.0",
            metadata={
                "source": "official_cifar10",
                "train_samples": 50000,
                "test_samples": 10000,
            },
        )

    def get_default_partition(
        self,
        seed: int = 42,
        val_ratio: float = 0.1,
    ) -> PartitionManifest:
        """PRISM Benchmark Partition Policy:
        - 50,000 Train -> 45,000 PRISM Train (90%) + 5,000 Val (10%).
        - 10,000 Test -> 10,000 PRISM Test isolated.
        """
        canonical = self.get_canonical_manifest()
        train_ratio = 1.0 - val_ratio
        return generate_partition_manifest(
            canonical_manifest=canonical,
            split_ratios={"train": train_ratio, "val": val_ratio},
            seed=seed,
            strategy="stratified",
            partition_id=f"part-cifar10-s{seed}",
            source_split_filter="train",
            isolated_splits={"test": "test"},
        )

    def get_nested_subsets(
        self,
        budget_ratios: Sequence[float] | None = None,
        seed: int = 42,
        val_ratio: float = 0.1,
    ) -> dict[float, SubsetManifest]:
        """Generate strictly nested data-budget subsets from training partition."""
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
        data_root = Path(root or "./data/datasets/cifar10")
        return (data_root / "cifar-10-batches-py").is_dir()

    def load_raw_dataset(
        self,
        split: str = "train",
        root: Path | str | None = None,
        download: bool = False,
        transform: Any = None,
    ) -> Any:
        data_root = str(root or "./data/datasets/cifar10")
        is_train = split.lower() == "train"
        return _load_torchvision_dataset(
            dataset_name="CIFAR10",
            root=data_root,
            train=is_train,
            download=download,
            transform=transform,
        )


class CIFAR100Adapter(BenchmarkDatasetAdapter):
    """Adapter for the CIFAR-100 classification benchmark."""

    @property
    def dataset_id(self) -> str:
        return "ds-cifar100"

    @property
    def name(self) -> str:
        return "CIFAR-100"

    @property
    def num_classes(self) -> int:
        return 100

    @property
    def classes(self) -> list[str]:
        return list(CIFAR100_CLASSES)

    def get_dataset_manifest(self) -> DatasetManifest:
        return DatasetManifest(
            dataset_id=self.dataset_id,
            name=self.name,
            description="CIFAR-100 image classification benchmark (60k 32x32 images)",
            source_uri="https://www.cs.toronto.edu/~kriz/cifar.html",
            version="1.0.0",
            compatible_tasks=[
                TaskType.CLASSIFICATION,
                TaskType.REPRESENTATION_LEARNING,
            ],
            splits=[
                SplitSpecification(split_name="train", num_samples=50000),
                SplitSpecification(split_name="test", num_samples=10000),
            ],
            classes=self.classes,
            num_classes=self.num_classes,
            preprocessing=PreprocessingPolicy(
                resize=(32, 32),
                normalization_mean=(0.5071, 0.4867, 0.4408),
                normalization_std=(0.2675, 0.2565, 0.2761),
                color_space="RGB",
            ),
            augmentation=AugmentationPolicy(name="none", enabled=False),
        )

    def get_canonical_manifest(self) -> CanonicalSampleManifest:
        """Build canonical sample universe for CIFAR-100 (50k train + 10k test)."""
        samples: list[SampleRecord] = []

        # Canonical training partition: 50,000 samples (500 per class)
        for idx in range(50000):
            target = idx % 100
            samples.append(
                SampleRecord(
                    sample_id=f"cifar100/train/{idx:06d}",
                    source_split="train",
                    source_index=idx,
                    target=target,
                    metadata={"class_name": self.classes[target]},
                )
            )

        # Canonical test partition: 10,000 samples (100 per class)
        for idx in range(10000):
            target = idx % 100
            samples.append(
                SampleRecord(
                    sample_id=f"cifar100/test/{idx:06d}",
                    source_split="test",
                    source_index=idx,
                    target=target,
                    metadata={"class_name": self.classes[target]},
                )
            )

        return CanonicalSampleManifest.create(
            dataset_id=self.dataset_id,
            samples=samples,
            dataset_version="1.0.0",
            metadata={
                "source": "official_cifar100",
                "train_samples": 50000,
                "test_samples": 10000,
            },
        )

    def get_default_partition(
        self,
        seed: int = 42,
        val_ratio: float = 0.1,
    ) -> PartitionManifest:
        """PRISM Benchmark Partition Policy:
        - 50,000 Train -> 45,000 PRISM Train (90%) + 5,000 Val (10%).
        - 10,000 Test -> 10,000 PRISM Test isolated.
        """
        canonical = self.get_canonical_manifest()
        train_ratio = 1.0 - val_ratio
        return generate_partition_manifest(
            canonical_manifest=canonical,
            split_ratios={"train": train_ratio, "val": val_ratio},
            seed=seed,
            strategy="stratified",
            partition_id=f"part-cifar100-s{seed}",
            source_split_filter="train",
            isolated_splits={"test": "test"},
        )

    def get_nested_subsets(
        self,
        budget_ratios: Sequence[float] | None = None,
        seed: int = 42,
        val_ratio: float = 0.1,
    ) -> dict[float, SubsetManifest]:
        """Generate strictly nested data-budget subsets from training partition."""
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
        data_root = Path(root or "./data/datasets/cifar100")
        return (data_root / "cifar-100-python").is_dir()

    def load_raw_dataset(
        self,
        split: str = "train",
        root: Path | str | None = None,
        download: bool = False,
        transform: Any = None,
    ) -> Any:
        data_root = str(root or "./data/datasets/cifar100")
        is_train = split.lower() == "train"
        return _load_torchvision_dataset(
            dataset_name="CIFAR100",
            root=data_root,
            train=is_train,
            download=download,
            transform=transform,
        )
