"""Base vision model interface for PRISM architectures."""

from abc import ABC, abstractmethod
from typing import Any

from prism.models.specifications import ModelSpecification


class BaseVisionModel(ABC):
    """Abstract base contract for trainable vision models in PRISM."""

    def __init__(self, spec: ModelSpecification) -> None:
        self.spec = spec
        self._is_training: bool = True

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

    @property
    def is_training(self) -> bool:
        """Return True if model is in training mode, False if in evaluation mode."""
        return self._is_training

    def train(self, mode: bool = True) -> "BaseVisionModel":
        """Set training mode (enables dropout and training-specific behaviors)."""
        self._is_training = mode
        return self

    def eval(self) -> "BaseVisionModel":
        """Set evaluation mode (disables dropout and stochastic behaviors)."""
        self._is_training = False
        return self

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
        """Return a mapping of trainable parameter names to their current values."""
        ...

    @abstractmethod
    def set_parameters(self, params: dict[str, Any]) -> None:
        """Load trainable parameter values from a mapping."""
        ...

    @abstractmethod
    def get_gradients(self) -> dict[str, Any]:
        """Return a mapping of parameter names to their computed gradients."""
        ...

    def get_state(self) -> dict[str, Any]:
        """Return non-trainable model state (e.g. running statistics)."""
        return {}

    def set_state(self, state: dict[str, Any]) -> None:
        """Load non-trainable model state."""
        _ = state

    @abstractmethod
    def extract_representations(
        self, inputs: Any, layer: str = "final_hidden"
    ) -> Any:
        """Extract intermediate representations, feature vectors, or spatial maps."""
        ...
