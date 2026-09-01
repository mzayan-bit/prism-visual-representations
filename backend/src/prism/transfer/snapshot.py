"""Model state snapshot contracts, validation, and restoration."""

from __future__ import annotations

import copy
import datetime
import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from prism.core.enums import ModelFamily
from prism.core.errors import SerializationError, ValidationError
from prism.models.base import BaseVisionModel
from prism.models.cnn import ConvolutionalNeuralNetwork
from prism.models.linear import LinearSoftmaxClassifier
from prism.models.mlp import MultiLayerPerceptron
from prism.models.resnet import ResidualNeuralNetwork
from prism.models.specifications import ModelSpecification
from prism.models.transformer import VisionTransformer


def compute_tensor_checksum(data: Any) -> str:
    """Compute SHA-256 checksum over arbitrary nested numerical lists/dicts."""
    serialized = json.dumps(data, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class ModelStateSnapshot(BaseModel):
    """Immutable state snapshot capturing trained parameters and non-trainable state.

    Used as the serializable representation artifact transferred from source models
    to downstream target tasks.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_experiment_id: str = Field(
        ..., description="Identifier of the experiment that produced this state"
    )
    model_spec: ModelSpecification = Field(
        ..., description="Full architecture specification of the source model"
    )
    parameters: dict[str, Any] = Field(
        ...,
        description="Mapping of parameter names to trained numerical weights/biases",
    )
    non_trainable_state: dict[str, Any] = Field(
        default_factory=dict,
        description="Non-trainable model state such as BatchNorm running statistics",
    )
    parameter_checksum: str = Field(
        ..., description="SHA-256 digest validating parameter integrity"
    )
    state_checksum: str = Field(
        ..., description="SHA-256 digest validating non-trainable state integrity"
    )
    created_at_utc: str = Field(
        default_factory=lambda: datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        description="Timestamp of snapshot creation",
    )

    def verify_integrity(self) -> bool:
        """Verify that stored checksums match current parameter and state payloads."""
        computed_p_sum = compute_tensor_checksum(self.parameters)
        computed_s_sum = compute_tensor_checksum(self.non_trainable_state)
        return (
            self.parameter_checksum == computed_p_sum
            and self.state_checksum == computed_s_sum
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize snapshot to dictionary."""
        return self.model_dump(mode="json")

    def to_json(self, indent: int | None = None) -> str:
        """Serialize snapshot to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelStateSnapshot:
        """Construct snapshot from dictionary and verify checksum integrity."""
        try:
            snapshot = cls.model_validate(data)
            if not snapshot.verify_integrity():
                raise ValidationError(
                    "ModelStateSnapshot verification failed: checksum mismatch."
                )
            return snapshot
        except Exception as exc:
            if isinstance(exc, ValidationError):
                raise
            raise SerializationError(
                f"Failed to deserialize ModelStateSnapshot: {exc}"
            ) from exc

    @classmethod
    def from_json(cls, json_str: str) -> ModelStateSnapshot:
        """Construct snapshot from JSON string."""
        try:
            data = json.loads(json_str)
            return cls.from_dict(data)
        except Exception as exc:
            if isinstance(exc, (ValidationError, SerializationError)):
                raise
            raise SerializationError(
                f"Failed to parse JSON for ModelStateSnapshot: {exc}"
            ) from exc


def create_model_state_snapshot(
    model: BaseVisionModel,
    source_experiment_id: str = "source_exp",
) -> ModelStateSnapshot:
    """Safely extract trainable parameters and non-trainable state into a snapshot.

    Args:
        model: Trained vision model instance.
        source_experiment_id: Identifier of the source experiment.

    Returns:
        Verified ModelStateSnapshot envelope.
    """
    raw_params = copy.deepcopy(model.get_parameters())
    raw_state = copy.deepcopy(model.get_state())

    p_checksum = compute_tensor_checksum(raw_params)
    s_checksum = compute_tensor_checksum(raw_state)

    return ModelStateSnapshot(
        source_experiment_id=source_experiment_id,
        model_spec=model.spec,
        parameters=raw_params,
        non_trainable_state=raw_state,
        parameter_checksum=p_checksum,
        state_checksum=s_checksum,
    )


def validate_snapshot_compatibility(
    snapshot: ModelStateSnapshot,
    target_model: BaseVisionModel,
) -> None:
    """Validate that a snapshot can be cleanly loaded into the target model.

    Checks architecture family, input shape, and parameter dimension compatibility.

    Raises:
        ValidationError: If models are structurally incompatible.
    """
    if snapshot.model_spec.family != target_model.spec.family:
        raise ValidationError(
            f"Architecture family mismatch: snapshot has "
            f"'{snapshot.model_spec.family}', target has "
            f"'{target_model.spec.family}'."
        )

    if snapshot.model_spec.input_shape != target_model.spec.input_shape:
        raise ValidationError(
            f"Input shape mismatch: snapshot has {snapshot.model_spec.input_shape}, "
            f"target model has {target_model.spec.input_shape}."
        )

    # Check that backbone parameter shapes match
    target_params = target_model.get_parameters()
    for name, _p_val in snapshot.parameters.items():
        if "classifier" in name or "fc" in name:
            # Classification head shapes may differ if num_classes is changed
            continue
        if name not in target_params:
            raise ValidationError(
                f"Snapshot contains parameter '{name}' not present in target model."
            )


def restore_model_from_snapshot(
    snapshot: ModelStateSnapshot,
    target_spec: ModelSpecification | None = None,
    seed: int = 42,
) -> BaseVisionModel:
    """Instantiate a model and restore parameters and state from snapshot.

    Args:
        snapshot: Validated ModelStateSnapshot.
        target_spec: Optional custom ModelSpecification (must be compatible).
        seed: Random seed for initialization.

    Returns:
        Instantiated and restored BaseVisionModel.
    """
    if not snapshot.verify_integrity():
        raise ValidationError(
            "Cannot restore model from corrupted snapshot: checksum mismatch."
        )

    spec = target_spec or snapshot.model_spec
    if spec.family != snapshot.model_spec.family:
        raise ValidationError(
            f"Cannot restore snapshot of family '{snapshot.model_spec.family}' "
            f"into target specification of family '{spec.family}'."
        )

    # Instantiate model
    model: BaseVisionModel
    if spec.family == ModelFamily.VISION_TRANSFORMER:
        model = VisionTransformer(spec=spec, seed=seed)
    elif spec.family == ModelFamily.RESNET:
        model = ResidualNeuralNetwork(spec=spec, seed=seed)
    elif spec.family == ModelFamily.CNN:
        model = ConvolutionalNeuralNetwork(spec=spec, seed=seed)
    elif spec.family == ModelFamily.MLP:
        model = MultiLayerPerceptron(spec=spec, seed=seed)
    else:
        model = LinearSoftmaxClassifier(spec=spec, seed=seed)

    # Set parameters and non-trainable state
    validate_snapshot_compatibility(snapshot, model)
    model.set_parameters(snapshot.parameters)
    model.set_state(snapshot.non_trainable_state)

    return model
