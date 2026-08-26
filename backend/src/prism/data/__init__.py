"""Dataset loading, manifest tracking, and deterministic data pipelines."""

from prism.data.adapters import (
    BenchmarkDatasetAdapter,
    CIFAR10Adapter,
    CIFAR100Adapter,
)
from prism.data.manifests import (
    AugmentationPolicy,
    ControlledDataReference,
    DatasetManifest,
    PreprocessingPolicy,
    SplitSpecification,
)
from prism.data.materialized import (
    MaterializedDataset,
    MaterializedSample,
)
from prism.data.materializer import DatasetMaterializer
from prism.data.partitions import (
    PartitionManifest,
    PartitionSplit,
    generate_partition_manifest,
)
from prism.data.preprocessing import (
    ExecutablePreprocessing,
    create_executable_preprocessing,
)
from prism.data.samples import (
    CanonicalSampleManifest,
    SampleRecord,
)
from prism.data.subsets import (
    DEFAULT_DATA_BUDGETS,
    SubsetManifest,
    generate_nested_subsets,
)
from prism.data.synthetic import SyntheticVisionAdapter

__all__ = [
    "DEFAULT_DATA_BUDGETS",
    "AugmentationPolicy",
    "BenchmarkDatasetAdapter",
    "CIFAR10Adapter",
    "CIFAR100Adapter",
    "CanonicalSampleManifest",
    "ControlledDataReference",
    "DatasetManifest",
    "DatasetMaterializer",
    "ExecutablePreprocessing",
    "MaterializedDataset",
    "MaterializedSample",
    "PartitionManifest",
    "PartitionSplit",
    "PreprocessingPolicy",
    "SampleRecord",
    "SplitSpecification",
    "SubsetManifest",
    "SyntheticVisionAdapter",
    "create_executable_preprocessing",
    "generate_nested_subsets",
    "generate_partition_manifest",
]
