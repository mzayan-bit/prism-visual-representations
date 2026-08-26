"""Centralized identifier generation and validation utilities for PRISM."""

import re
import uuid

from prism.core.errors import ValidationError

# Valid identifier pattern: letters, numbers, underscores, hyphens (length 3 to 64)
IDENTIFIER_PATTERN: re.Pattern[str] = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]{2,63}$")


def _generate_short_uuid() -> str:
    """Generate a clean, collision-resistant 12-character hex suffix."""
    return uuid.uuid4().hex[:12]


def generate_experiment_id(prefix: str = "exp") -> str:
    """Generate a structured experiment identifier (e.g. 'exp-a1b2c3d4e5f6')."""
    return f"{prefix}-{_generate_short_uuid()}"


def generate_run_id(prefix: str = "run") -> str:
    """Generate a structured experiment run identifier (e.g. 'run-a1b2c3d4e5f6')."""
    return f"{prefix}-{_generate_short_uuid()}"


def generate_artifact_id(prefix: str = "art") -> str:
    """Generate a structured artifact identifier (e.g. 'art-a1b2c3d4e5f6')."""
    return f"{prefix}-{_generate_short_uuid()}"


def generate_dataset_id(prefix: str = "ds") -> str:
    """Generate a structured dataset identifier (e.g. 'ds-a1b2c3d4e5f6')."""
    return f"{prefix}-{_generate_short_uuid()}"


def generate_partition_id(prefix: str = "part") -> str:
    """Generate a structured partition identifier (e.g. 'part-a1b2c3d4e5f6')."""
    return f"{prefix}-{_generate_short_uuid()}"


def generate_subset_id(prefix: str = "sub") -> str:
    """Generate a structured subset identifier (e.g. 'sub-a1b2c3d4e5f6')."""
    return f"{prefix}-{_generate_short_uuid()}"


def generate_model_id(prefix: str = "model") -> str:
    """Generate a structured model identifier (e.g. 'model-a1b2c3d4e5f6')."""
    return f"{prefix}-{_generate_short_uuid()}"


def generate_report_id(prefix: str = "rep") -> str:
    """Generate a structured evaluation report identifier (e.g. 'rep-a1b2c3d4e5f6')."""
    return f"{prefix}-{_generate_short_uuid()}"


def validate_identifier(identifier: str, expected_prefix: str | None = None) -> bool:
    """Check whether an identifier conforms to length, character, and prefix rules."""
    if not isinstance(identifier, str):
        return False
    if not IDENTIFIER_PATTERN.match(identifier):
        return False
    return not (expected_prefix and not identifier.startswith(f"{expected_prefix}-"))


def ensure_valid_identifier(
    identifier: str,
    expected_prefix: str | None = None,
    field_name: str = "identifier",
) -> str:
    """Validate an identifier and return it, or raise a ValidationError."""
    if not isinstance(identifier, str) or not identifier.strip():
        raise ValidationError(f"{field_name} must be a non-empty string.")

    cleaned = identifier.strip()
    if not IDENTIFIER_PATTERN.match(cleaned):
        raise ValidationError(
            f"{field_name} '{cleaned}' is invalid. Identifiers must start "
            "with a letter and contain only alphanumeric characters, underscores, "
            "or hyphens (3-64 chars)."
        )

    if expected_prefix and not cleaned.startswith(f"{expected_prefix}-"):
        raise ValidationError(
            f"{field_name} '{cleaned}' must start with prefix '{expected_prefix}-'."
        )

    return cleaned
