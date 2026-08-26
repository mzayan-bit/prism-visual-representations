"""Dataset loading, manifest tracking, and deterministic data pipelines."""

from prism.data.adapters import (
    BenchmarkDatasetAdapter,
    CIFAR10Adapter,
    CIFAR100Adapter,
)
from prism.data.batching import (
    DeterministicBatchLoader,
    MaterializedBatch,
    default_collate_fn,
)
from prism.data.context import DataRuntimeContext
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
from prism.data.ordering import (
    OrderingSpecification,
    compute_ordering_fingerprint,
    compute_sample_order,
)
from prism.data.partitions import (
    PartitionManifest,
    PartitionSplit,
    generate_partition_manifest,
)
from prism.data.preparer import DataPreparer
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
    "DataPreparer",
    "DataRuntimeContext",
    "DatasetManifest",
    "DatasetMaterializer",
    "DeterministicBatchLoader",
    "ExecutablePreprocessing",
    "MaterializedBatch",
    "MaterializedDataset",
    "MaterializedSample",
    "OrderingSpecification",
    "PartitionManifest",
    "PartitionSplit",
    "PreprocessingPolicy",
    "SampleRecord",
    "SplitSpecification",
    "SubsetManifest",
    "SyntheticVisionAdapter",
    "compute_ordering_fingerprint",
    "compute_sample_order",
    "create_executable_preprocessing",
    "default_collate_fn",
    "generate_nested_subsets",
    "generate_partition_manifest",
]
