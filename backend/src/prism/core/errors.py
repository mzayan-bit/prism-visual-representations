"""Domain-specific exceptions for PRISM."""

from typing import Any


class PrismError(Exception):
    """Base exception for all PRISM domain errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} (details: {self.details})"
        return self.message


class ConfigurationError(PrismError):
    """Raised when an experiment, model, or dataset config is invalid."""


class ValidationError(PrismError):
    """Raised when domain constraints or invariants are violated."""


class LifecycleError(PrismError):
    """Base exception for experiment run lifecycle errors."""


class InvalidTransitionError(LifecycleError):
    """Raised when an illegal lifecycle state transition is attempted."""

    def __init__(
        self,
        current_status: str,
        target_status: str,
        run_id: str | None = None,
        reason: str | None = None,
    ) -> None:
        msg = f"Cannot transition run from '{current_status}' to '{target_status}'"
        if run_id:
            msg = f"Run '{run_id}': {msg}"
        if reason:
            msg = f"{msg}. Reason: {reason}"
        super().__init__(
            msg,
            details={
                "current_status": current_status,
                "target_status": target_status,
                "run_id": run_id,
                "reason": reason,
            },
        )
        self.current_status = current_status
        self.target_status = target_status
        self.run_id = run_id


class SerializationError(PrismError):
    """Raised when serialization or deserialization of domain entities fails."""


class FingerprintError(PrismError):
    """Raised when configuration fingerprinting or hashing fails."""


class ReproducibilityError(PrismError):
    """Raised when strict reproducibility requirements cannot be satisfied."""


class RuntimeInitializationError(PrismError):
    """Raised when preparing or initializing the execution runtime fails."""


class ProvenanceError(PrismError):
    """Raised when capturing critical source code or dataset provenance fails."""


class DatasetMaterializationError(PrismError):
    """Raised when materializing dataset samples fails."""


class SampleResolutionError(PrismError):
    """Raised when resolving a sample ID against source data fails."""


class DataPreparationError(PrismError):
    """Raised when preparing executable datasets or batching fails."""
