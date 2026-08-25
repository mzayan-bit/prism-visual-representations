"""Deterministic configuration hashing and fingerprint generation."""

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel

from prism.core.errors import FingerprintError

# Non-semantic fields to exclude from semantic configuration fingerprinting
DEFAULT_EXCLUDED_KEYS: frozenset[str] = frozenset(
    {
        "created_at",
        "created_by",
        "description",
        "hypothesis",
        "notes",
        "tags",
    }
)


def _canonicalize_value(val: Any, excluded_keys: frozenset[str]) -> Any:
    """Recursively convert data structures to canonical JSON-compatible objects."""
    if isinstance(val, BaseModel):
        # Convert Pydantic models using mode='json'
        dumped = val.model_dump(mode="json")
        return _canonicalize_value(dumped, excluded_keys)

    if isinstance(val, Enum):
        return val.value

    if isinstance(val, (datetime, date)):
        return val.isoformat()

    if isinstance(val, Mapping):
        # Sort dictionary keys and filter excluded keys
        return {
            str(k): _canonicalize_value(v, excluded_keys)
            for k, v in sorted(val.items())
            if str(k) not in excluded_keys
        }

    if isinstance(val, (list, tuple, Sequence)) and not isinstance(val, (str, bytes)):
        return [_canonicalize_value(item, excluded_keys) for item in val]

    if isinstance(val, float):
        # Format float consistently to prevent precision discrepancies
        if val.is_integer():
            return int(val)
        return round(val, 8)

    return val


def compute_configuration_fingerprint(
    config: BaseModel | dict[str, Any],
    excluded_keys: frozenset[str] = DEFAULT_EXCLUDED_KEYS,
) -> str:
    """Compute a deterministic SHA-256 cryptographic hex digest of a configuration.

    Parameters
    ----------
    config : BaseModel | dict[str, Any]
        The experiment or component configuration to fingerprint.
    excluded_keys : frozenset[str]
        Keys to ignore during canonical serialization (e.g. timestamps, descriptions).

    Returns
    -------
    str
        64-character SHA-256 hexadecimal string representing the semantic configuration.

    Raises
    ------
    FingerprintError
        If canonical serialization fails.
    """
    try:
        canonical_obj = _canonicalize_value(config, excluded_keys)
        canonical_json = json.dumps(
            canonical_obj,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    except Exception as exc:
        raise FingerprintError(
            f"Failed to compute configuration fingerprint: {exc}"
        ) from exc
