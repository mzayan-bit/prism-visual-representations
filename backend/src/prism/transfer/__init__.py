"""PRISM Transfer Learning and Representation Reuse Subsystem."""

from prism.transfer.freezing import (
    ParameterFreezePlan,
    count_tensor_elements,
    create_freeze_plan,
    get_architecture_stages,
)
from prism.transfer.head import replace_classifier_head
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
    "ModelStateSnapshot",
    "NormalizationTransferPolicy",
    "ParameterFreezePlan",
    "TransferLearningSpecification",
    "TransferStrategy",
    "compute_tensor_checksum",
    "count_tensor_elements",
    "create_freeze_plan",
    "create_model_state_snapshot",
    "get_architecture_stages",
    "replace_classifier_head",
    "restore_model_from_snapshot",
    "validate_snapshot_compatibility",
]
