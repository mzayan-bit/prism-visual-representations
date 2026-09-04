"""Training Specification for Multimodal Vision-Language Alignment."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from prism.core.enums import ModelFamily
from prism.models.specifications import ModelSpecification


@dataclass(frozen=True)
class VisionLanguageTrainingSpecification:
    """Specification for dual-encoder vision-language contrastive training."""

    visual_family: ModelFamily
    visual_spec: ModelSpecification
    text_dim: int = 32
    shared_dim: int = 16
    temperature: float = 0.07
    use_mlp_projection: bool = False
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-4
    batch_size: int = 8
    epochs: int = 5
    seed: int = 42
    dataset_fingerprint: str = ""
    tokenizer_max_length: int = 16

    @property
    def fingerprint(self) -> str:
        """Deterministic fingerprint covering hyperparameters."""
        payload = {
            "visual_family": self.visual_family.value,
            "visual_spec_arch": self.visual_spec.architecture,
            "text_dim": self.text_dim,
            "shared_dim": self.shared_dim,
            "temperature": self.temperature,
            "use_mlp_projection": self.use_mlp_projection,
            "learning_rate": self.learning_rate,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "seed": self.seed,
            "dataset_fingerprint": self.dataset_fingerprint,
            "tokenizer_max_length": self.tokenizer_max_length,
        }
        encoded = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(encoded).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        """Serialize training specification."""
        return {
            "visual_family": self.visual_family.value,
            "visual_spec": self.visual_spec.model_dump(),
            "text_dim": self.text_dim,
            "shared_dim": self.shared_dim,
            "temperature": self.temperature,
            "use_mlp_projection": self.use_mlp_projection,
            "learning_rate": self.learning_rate,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "seed": self.seed,
            "dataset_fingerprint": self.dataset_fingerprint,
            "tokenizer_max_length": self.tokenizer_max_length,
            "fingerprint": self.fingerprint,
        }
