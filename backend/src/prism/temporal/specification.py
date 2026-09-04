"""Configuration specifications for temporal representation transfer experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from prism.core.enums import ModelFamily
from prism.temporal.enums import (
    PretrainingObjective,
    RNNAggregationMode,
    TemporalAggregationType,
    TemporalTransferStrategy,
)


@dataclass
class TemporalTransferSpecification:
    """Rigorous configuration specification for a temporal representation experiment."""

    source_objective: PretrainingObjective
    architecture: ModelFamily
    selected_layer: str
    temporal_aggregator: TemporalAggregationType
    transfer_strategy: TemporalTransferStrategy
    rnn_hidden_dim: int = 16
    rnn_mode: RNNAggregationMode = RNNAggregationMode.LAST_HIDDEN
    num_classes: int = 4
    sequence_length: int = 4
    learning_rate: float = 0.05
    epochs: int = 15
    seed: int = 42
    fingerprint: str = ""

    def __post_init__(self) -> None:
        """Compute deterministic SHA-256 configuration fingerprint."""
        if not self.fingerprint:
            payload = {
                "source_objective": self.source_objective.value,
                "architecture": self.architecture.value,
                "selected_layer": self.selected_layer,
                "temporal_aggregator": self.temporal_aggregator.value,
                "transfer_strategy": self.transfer_strategy.value,
                "rnn_hidden_dim": self.rnn_hidden_dim,
                "rnn_mode": self.rnn_mode.value,
                "num_classes": self.num_classes,
                "sequence_length": self.sequence_length,
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
                "seed": self.seed,
            }
            raw = json.dumps(payload, sort_keys=True).encode("utf-8")
            self.fingerprint = hashlib.sha256(raw).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration specification to dictionary."""
        data = asdict(self)
        data["source_objective"] = self.source_objective.value
        data["architecture"] = self.architecture.value
        data["temporal_aggregator"] = self.temporal_aggregator.value
        data["transfer_strategy"] = self.transfer_strategy.value
        data["rnn_mode"] = self.rnn_mode.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TemporalTransferSpecification:
        """Instantiate specification from dictionary."""
        return cls(
            source_objective=PretrainingObjective(data["source_objective"]),
            architecture=ModelFamily(data["architecture"]),
            selected_layer=str(data["selected_layer"]),
            temporal_aggregator=TemporalAggregationType(data["temporal_aggregator"]),
            transfer_strategy=TemporalTransferStrategy(data["transfer_strategy"]),
            rnn_hidden_dim=int(data.get("rnn_hidden_dim", 16)),
            rnn_mode=RNNAggregationMode(
                data.get("rnn_mode", RNNAggregationMode.LAST_HIDDEN.value)
            ),
            num_classes=int(data.get("num_classes", 4)),
            sequence_length=int(data.get("sequence_length", 4)),
            learning_rate=float(data.get("learning_rate", 0.05)),
            epochs=int(data.get("epochs", 15)),
            seed=int(data.get("seed", 42)),
            fingerprint=str(data.get("fingerprint", "")),
        )
