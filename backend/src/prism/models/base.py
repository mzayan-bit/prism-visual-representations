"""Base vision model interface for PRISM architectures."""

from abc import ABC, abstractmethod
from typing import Any

from prism.models.specifications import ModelSpecification


class BaseVisionModel(ABC):
    """Abstract base contract for trainable vision models in PRISM."""

    def __init__(self, spec: ModelSpecification) -> None:
        self.spec = spec

    @property
    def model_id(self) -> str:
        """Unique model identifier from specification."""
        return self.spec.model_id

    @property
    def num_classes(self) -> int:
        """Declared number of target classes."""
        if self.spec.num_classes is None:
            raise ValueError(
                f"Model '{self.model_id}' does not have num_classes specified."
            )
        return self.spec.num_classes

    @abstractmethod
    def forward(self, inputs: Any) -> list[list[float]]:
        """Compute forward pass and return raw output logits [B, num_classes]."""
        ...

    @abstractmethod
    def backward(self, d_logits: list[list[float]]) -> None:
        """Compute and store parameter gradients given upstream loss derivatives."""
        ...

    @abstractmethod
    def zero_grad(self) -> None:
        """Clear all stored parameter gradients."""
        ...

    @abstractmethod
    def get_parameters(self) -> dict[str, Any]:
        """Return a mapping of parameter names to their current values."""
        ...

    @abstractmethod
    def set_parameters(self, params: dict[str, Any]) -> None:
        """Load parameter values from a mapping."""
        ...

    @abstractmethod
    def get_gradients(self) -> dict[str, Any]:
        """Return a mapping of parameter names to their computed gradients."""
        ...
