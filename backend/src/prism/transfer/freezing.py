"""Parameter freeze plans, stage grouping, and trainable parameter selection."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from prism.core.enums import ModelFamily
from prism.core.errors import ValidationError
from prism.models.base import BaseVisionModel
from prism.transfer.specification import TransferStrategy


def count_tensor_elements(val: Any) -> int:
    """Recursively compute total number of scalar elements in a tensor/list."""
    if isinstance(val, (list, tuple)):
        return sum(count_tensor_elements(item) for item in val)
    return 1


class ParameterFreezePlan(BaseModel):
    """Immutable plan declaring which model parameters are frozen vs trainable.

    Provides exact counts of trainable and frozen scalar elements, as well as
    trainable fractions for efficiency and transfer auditing.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    frozen_parameters: list[str] = Field(
        default_factory=list, description="Names of parameter tensors that are frozen"
    )
    trainable_parameters: list[str] = Field(
        default_factory=list,
        description="Names of parameter tensors that are trainable",
    )
    logical_stages: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Mapping from logical stage/block name to parameter keys",
    )
    total_tensors: int = Field(
        ..., description="Total count of parameter tensor blocks"
    )
    frozen_tensors: int = Field(
        ..., description="Count of frozen parameter tensor blocks"
    )
    trainable_tensors: int = Field(
        ..., description="Count of trainable parameter tensor blocks"
    )
    total_scalar_elements: int = Field(
        ..., description="Total scalar parameter count in the model"
    )
    frozen_scalar_elements: int = Field(
        ..., description="Total scalar parameter count that is frozen"
    )
    trainable_scalar_elements: int = Field(
        ..., description="Total scalar parameter count that is trainable"
    )
    trainable_fraction: float = Field(
        ..., ge=0.0, le=1.0, description="Ratio of trainable to total scalar parameters"
    )

    def is_trainable(self, param_name: str) -> bool:
        """Return True if parameter is designated as trainable."""
        return param_name in self.trainable_parameters

    def is_frozen(self, param_name: str) -> bool:
        """Return True if parameter is designated as frozen."""
        return param_name in self.frozen_parameters

    def to_dict(self) -> dict[str, Any]:
        """Convert freeze plan to dictionary."""
        return self.model_dump(mode="json")

    def to_json(self, indent: int | None = None) -> str:
        """Serialize freeze plan to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def get_architecture_stages(model: BaseVisionModel) -> dict[str, list[str]]:
    """Group model parameter keys into logical architectural stages.

    Returns:
        Mapping from stage identifier to parameter keys.
    """
    params = model.get_parameters()
    family = model.spec.family
    stages: dict[str, list[str]] = {}

    for key in params:
        stage_name = "other"

        if family == ModelFamily.CNN:
            if "classifier" in key or "fc" in key:
                stage_name = "classifier"
            elif key.startswith("conv_") or key.startswith("norm_"):
                # e.g. conv_0_weights -> conv_0
                parts = key.split("_")
                stage_name = f"{parts[0]}_{parts[1]}"
            else:
                stage_name = "backbone"

        elif family == ModelFamily.RESNET:
            if "classifier" in key or "fc" in key:
                stage_name = "classifier"
            elif "stem" in key:
                stage_name = "stem"
            elif "stage_" in key:
                # e.g. stage_0_block_0_conv1_weights -> stage_0
                parts = key.split("_")
                stage_name = f"{parts[0]}_{parts[1]}"
            else:
                stage_name = "backbone"

        elif family == ModelFamily.VISION_TRANSFORMER:
            if "classifier" in key:
                stage_name = "classifier"
            elif key.startswith("patch_embed"):
                stage_name = "patch_embed"
            elif key.startswith("cls_token") or key.startswith("pos_embed"):
                stage_name = "embeddings"
            elif key.startswith("encoder.blocks."):
                # e.g. encoder.blocks.0.attn... -> encoder_block_0
                b_idx = key.split(".")[2]
                stage_name = f"encoder_block_{b_idx}"
            elif key.startswith("norm"):
                stage_name = "final_norm"
            else:
                stage_name = "backbone"

        else:
            if "classifier" in key or "linear" in key or "fc" in key:
                stage_name = "classifier"
            else:
                stage_name = "backbone"

        if stage_name not in stages:
            stages[stage_name] = []
        stages[stage_name].append(key)

    return stages


def create_freeze_plan(
    model: BaseVisionModel,
    strategy: TransferStrategy = TransferStrategy.LINEAR_PROBE,
    frozen_prefixes: list[str] | None = None,
    trainable_prefixes: list[str] | None = None,
    custom_frozen_params: list[str] | None = None,
) -> ParameterFreezePlan:
    """Create a parameter freeze plan for a model given a transfer strategy.

    Args:
        model: Vision model under consideration.
        strategy: Transfer strategy (linear, partial, full, or scratch).
        frozen_prefixes: Explicit prefixes to freeze.
        trainable_prefixes: Explicit prefixes to keep trainable.
        custom_frozen_params: Explicit list of exact parameter keys to freeze.

    Returns:
        Validated ParameterFreezePlan.
    """
    params = model.get_parameters()
    stages = get_architecture_stages(model)

    frozen_set: set[str] = set()
    trainable_set: set[str] = set()

    all_keys = list(params.keys())
    if not all_keys:
        raise ValidationError(
            f"Model '{model.model_id}' exposes no trainable parameters."
        )

    if strategy in (TransferStrategy.FULL_FINE_TUNE, TransferStrategy.SCRATCH_BASELINE):
        # All parameters are trainable
        trainable_set = set(all_keys)

    elif strategy == TransferStrategy.LINEAR_PROBE:
        # Freeze all backbone parameters, only classification head is trainable
        for k in all_keys:
            if (
                "classifier" in k
                or "fc" in k
                or (k.endswith(".weights") and "encoder" not in k)
            ):
                # Check if it belongs to classifier stage
                if any(
                    k in stage_params
                    for s_name, stage_params in stages.items()
                    if s_name == "classifier"
                ):
                    trainable_set.add(k)
                else:
                    frozen_set.add(k)
            else:
                frozen_set.add(k)

        # Fallback if no classifier recognized by name
        if not trainable_set:
            if "classifier" in stages:
                trainable_set = set(stages["classifier"])
                frozen_set = set(all_keys) - trainable_set
            else:
                # Default: last parameter tensor is trainable
                trainable_set = {all_keys[-1]}
                frozen_set = set(all_keys[:-1])

    elif strategy == TransferStrategy.PARTIAL_FINE_TUNE:
        # Default partial fine-tuning: Freeze early stages, keep later stages trainable
        stage_names = list(stages.keys())
        non_cls_stages = [s for s in stage_names if s != "classifier"]
        num_to_freeze = (
            max(1, len(non_cls_stages) // 2) if len(non_cls_stages) > 1 else 1
        )

        default_frozen_stages = set(non_cls_stages[:num_to_freeze])

        for s_name, s_params in stages.items():
            if s_name in default_frozen_stages:
                frozen_set.update(s_params)
            else:
                trainable_set.update(s_params)

    # Apply explicit overrides if supplied
    if custom_frozen_params is not None:
        for p in custom_frozen_params:
            if p in all_keys:
                frozen_set.add(p)
                trainable_set.discard(p)

    if frozen_prefixes:
        for k in all_keys:
            if any(k.startswith(pfx) for pfx in frozen_prefixes):
                frozen_set.add(k)
                trainable_set.discard(k)

    if trainable_prefixes:
        for k in all_keys:
            if any(k.startswith(pfx) for pfx in trainable_prefixes):
                trainable_set.add(k)
                frozen_set.discard(k)

    # Ensure every parameter is in exactly one set
    for k in all_keys:
        if k not in frozen_set and k not in trainable_set:
            trainable_set.add(k)

    # Compute scalar element counts
    total_scalars = sum(count_tensor_elements(params[k]) for k in all_keys)
    frozen_scalars = sum(count_tensor_elements(params[k]) for k in frozen_set)
    trainable_scalars = sum(count_tensor_elements(params[k]) for k in trainable_set)

    trainable_fraction = (
        float(trainable_scalars) / float(total_scalars) if total_scalars > 0 else 0.0
    )

    return ParameterFreezePlan(
        frozen_parameters=sorted(frozen_set),
        trainable_parameters=sorted(trainable_set),
        logical_stages=stages,
        total_tensors=len(all_keys),
        frozen_tensors=len(frozen_set),
        trainable_tensors=len(trainable_set),
        total_scalar_elements=total_scalars,
        frozen_scalar_elements=frozen_scalars,
        trainable_scalar_elements=trainable_scalars,
        trainable_fraction=trainable_fraction,
    )
