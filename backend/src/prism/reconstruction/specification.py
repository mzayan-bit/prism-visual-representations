"""Reconstruction learning experiment specifications and identity hashing."""

from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, ConfigDict, Field

from prism.core.enums import ModelFamily
from prism.models.specifications import ModelSpecification
from prism.reconstruction.enums import ReconstructionMethod
from prism.robustness.corruptions import CorruptionType


class ReconstructionLearningSpecification(BaseModel):
    """Specification capturing all parameters of a reconstruction experiment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    reconstruction_id: str = Field(
        ..., description="Unique identifier for the reconstruction experiment"
    )
    method: ReconstructionMethod = Field(
        ..., description="Reconstruction objective (Masked Patch or Denoising)"
    )
    encoder_family: ModelFamily = Field(
        ..., description="Encoder architecture family (CNN, ResNet, ViT)"
    )
    encoder_spec: ModelSpecification = Field(
        ..., description="Full architecture specification of the encoder backbone"
    )
    decoder_hidden_dim: int | None = Field(
        default=None, description="Optional hidden dimension of decoder network"
    )
    input_shape: tuple[int, int, int] = Field(
        ..., description="Expected input shape (channels, height, width)"
    )
    patch_size: int | tuple[int, int] | None = Field(
        default=None, description="Patch height and width for patch-based modeling"
    )
    mask_ratio: float = Field(
        default=0.5,
        gt=0.0,
        lt=1.0,
        description="Fraction of patches masked in masked modeling",
    )
    corruption_type: CorruptionType | None = Field(
        default=None, description="Applied corruption type if denoising autoencoder"
    )
    corruption_severity: int = Field(
        default=1, ge=1, le=5, description="Corruption severity level (1 to 5)"
    )
    epochs: int = Field(default=10, gt=0, description="Pretraining epoch budget")
    batch_size: int = Field(default=16, gt=0, description="Pretraining batch size")
    learning_rate: float = Field(
        default=0.05, gt=0.0, description="Initial learning rate"
    )
    momentum: float = Field(
        default=0.9, ge=0.0, lt=1.0, description="SGD momentum coefficient"
    )
    weight_decay: float = Field(
        default=0.0001, ge=0.0, description="L2 weight decay regularization"
    )
    seed: int = Field(default=42, description="Root experiment seed")
    dataset_id: str = Field(..., description="Source dataset identifier")

    def fingerprint(self) -> str:
        """Compute cryptographic SHA-256 fingerprint of the experiment specification."""
        corr_val = self.corruption_type.value if self.corruption_type else None
        payload = {
            "reconstruction_id": self.reconstruction_id,
            "method": self.method.value,
            "encoder_family": self.encoder_family.value,
            "encoder_spec": self.encoder_spec.model_id,
            "input_shape": list(self.input_shape),
            "patch_size": self.patch_size,
            "mask_ratio": round(self.mask_ratio, 4),
            "corruption_type": corr_val,
            "corruption_severity": self.corruption_severity,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "dataset_id": self.dataset_id,
        }
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
