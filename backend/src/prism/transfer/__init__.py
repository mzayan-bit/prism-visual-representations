"""PRISM Transfer Learning and Representation Reuse Subsystem."""

from prism.transfer.freezing import (
    ParameterFreezePlan,
    count_tensor_elements,
    create_freeze_plan,
    get_architecture_stages,
)
from prism.transfer.head import replace_classifier_head
from prism.transfer.probes import (
    LayerTransferProbeResult,
    probe_all_layers_transferability,
    probe_layer_transferability,
)
from prism.transfer.reports import (
    DataEfficiencyTransferPoint,
    SampleEfficiencyTransferSummary,
    TransferExperimentSuite,
    TransferLearningReport,
    TransferStrategyComparisonSummary,
)
from prism.transfer.retention import (
    TransferRepresentationDriftSummary,
    compute_representation_retention,
    compute_transfer_shared_pca,
)
from prism.transfer.runner import TransferTrainingRunner
from prism.transfer.snapshot import (
    ModelStateSnapshot,
    compute_tensor_checksum,
    create_model_state_snapshot,
    restore_model_from_snapshot,
    validate_snapshot_compatibility,
)
from prism.transfer.specification import (
    NormalizationTransferPolicy,
    TransferLearningSpecification,
    TransferStrategy,
)

__all__ = [
    "DataEfficiencyTransferPoint",
    "LayerTransferProbeResult",
    "ModelStateSnapshot",
    "NormalizationTransferPolicy",
    "ParameterFreezePlan",
    "SampleEfficiencyTransferSummary",
    "TransferExperimentSuite",
    "TransferLearningReport",
    "TransferLearningSpecification",
    "TransferRepresentationDriftSummary",
    "TransferStrategy",
    "TransferStrategyComparisonSummary",
    "TransferTrainingRunner",
    "compute_representation_retention",
    "compute_tensor_checksum",
    "compute_transfer_shared_pca",
    "count_tensor_elements",
    "create_freeze_plan",
    "create_model_state_snapshot",
    "get_architecture_stages",
    "probe_all_layers_transferability",
    "probe_layer_transferability",
    "replace_classifier_head",
    "restore_model_from_snapshot",
    "validate_snapshot_compatibility",
]
