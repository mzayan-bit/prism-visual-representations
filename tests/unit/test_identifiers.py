"""Unit tests for centralized identifier generation and validation."""

import pytest

from prism.core.errors import ValidationError
from prism.core.identifiers import (
    ensure_valid_identifier,
    generate_artifact_id,
    generate_dataset_id,
    generate_experiment_id,
    generate_model_id,
    generate_report_id,
    generate_run_id,
    validate_identifier,
)


@pytest.mark.unit
def test_generate_identifiers_format() -> None:
    """Verify generated IDs have correct prefixes and are valid."""
    exp_id = generate_experiment_id()
    assert exp_id.startswith("exp-")
    assert validate_identifier(exp_id, expected_prefix="exp")

    run_id = generate_run_id()
    assert run_id.startswith("run-")
    assert validate_identifier(run_id, expected_prefix="run")

    art_id = generate_artifact_id()
    assert art_id.startswith("art-")
    assert validate_identifier(art_id, expected_prefix="art")

    ds_id = generate_dataset_id()
    assert ds_id.startswith("ds-")
    assert validate_identifier(ds_id, expected_prefix="ds")

    model_id = generate_model_id()
    assert model_id.startswith("model-")
    assert validate_identifier(model_id, expected_prefix="model")

    rep_id = generate_report_id()
    assert rep_id.startswith("rep-")
    assert validate_identifier(rep_id, expected_prefix="rep")


@pytest.mark.unit
def test_validate_custom_identifiers() -> None:
    """Verify custom valid identifiers pass validation."""
    assert validate_identifier("cifar10_resnet18_baseline")
    assert validate_identifier("exp-01-linear-probe")
    assert validate_identifier("ds-imagenet-1k")
    assert validate_identifier("model_vit_tiny")


@pytest.mark.unit
@pytest.mark.parametrize(
    "invalid_id",
    [
        "",
        " ",
        "123-starting-with-digit",
        "has spaces in id",
        "has$pecial#characters!",
        "ab",  # too short (< 3 chars)
        "a" * 65,  # too long (> 64 chars)
        "-leading-hyphen",
        "_leading-underscore",
    ],
)
def test_reject_invalid_identifiers(invalid_id: str) -> None:
    """Verify malformed identifiers fail validation."""
    assert not validate_identifier(invalid_id)
    with pytest.raises(ValidationError):
        ensure_valid_identifier(invalid_id)


@pytest.mark.unit
def test_prefix_mismatch_rejection() -> None:
    """Verify expected prefix mismatches are rejected."""
    assert not validate_identifier("run-123456", expected_prefix="exp")
    with pytest.raises(ValidationError, match="must start with prefix 'exp-'"):
        ensure_valid_identifier("run-123456", expected_prefix="exp")
