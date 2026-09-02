"""Self-supervised contrastive learning experiment specifications."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from prism.core.enums import ModelFamily
from prism.models.specifications import ModelSpecification


class SelfSupervisedTrainingSpecification(BaseModel):
    """Immutable, deterministic specification for a self-supervised pretraining run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ssl_id: str = Field(..., description="Unique identifier for this SSL run")
    method: str = Field(default="simclr", description="SSL method name")
    encoder_family: ModelFamily = Field(..., description="Encoder model family")
    encoder_spec: ModelSpecification = Field(
        ..., description="Underlying model specification for encoder"
    )
    dataset_id: str = Field(..., description="Source pretraining dataset identity")
    projection_hidden_dim: int = Field(
        default=32, ge=4, description="Hidden dimension of projection MLP"
    )
    projection_out_dim: int = Field(
        default=16, ge=4, description="Output metric projection dimension"
    )
    temperature: float = Field(
        default=0.5, gt=0.0, description="NT-Xent temperature parameter tau"
    )
    epochs: int = Field(default=5, ge=1, description="Number of pretraining epochs")
    batch_size: int = Field(
        default=16, ge=2, description="Number of source samples per batch (2N views)"
    )
    learning_rate: float = Field(
        default=0.05, gt=0.0, description="Initial learning rate"
    )
    weight_decay: float = Field(
        default=1e-4, ge=0.0, description="L2 weight decay factor"
    )
    momentum: float = Field(
        default=0.9, ge=0.0, le=1.0, description="Momentum coefficient"
    )
    seed: int = Field(default=42, description="Deterministic experiment seed")
    dataset_fingerprint: str = Field(
        default="synth_v1", description="Fingerprint of source data"
    )

    def sha256_fingerprint(self) -> str:
        """Compute deterministic SHA-256 fingerprint for this SSL specification."""
        payload = {
            "ssl_id": self.ssl_id,
            "method": self.method,
            "encoder_family": self.encoder_family.value,
            "encoder_architecture": self.encoder_spec.architecture,
            "dataset_id": self.dataset_id,
            "projection_hidden_dim": self.projection_hidden_dim,
            "projection_out_dim": self.projection_out_dim,
            "temperature": self.temperature,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "momentum": self.momentum,
            "seed": self.seed,
            "dataset_fingerprint": self.dataset_fingerprint,
        }
        serialized = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Convert specification to dictionary."""
        return self.model_dump()
