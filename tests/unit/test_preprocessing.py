"""Unit tests for executable preprocessing pipeline."""

import pydantic
import pytest

from prism.data.manifests import PreprocessingPolicy
from prism.data.preprocessing import (
    create_executable_preprocessing,
)


@pytest.mark.unit
def test_preprocessing_deterministic_normalization() -> None:
    """Verify ExecutablePreprocessing executes deterministic normalization."""
    policy = PreprocessingPolicy(
        normalization_mean=(0.5, 0.5, 0.5),
        normalization_std=(0.5, 0.5, 0.5),
    )
    prep = create_executable_preprocessing(policy)

    raw_data = [1.0, 0.5, 0.0]
    # (1.0 - 0.5) / 0.5 = 1.0; (0.5 - 0.5) / 0.5 = 0.0; (0.0 - 0.5) / 0.5 = -1.0
    processed1 = prep(raw_data)
    processed2 = prep(raw_data)

    assert processed1 == [1.0, 0.0, -1.0]
    assert processed1 == processed2


@pytest.mark.unit
def test_preprocessing_rejects_non_positive_std() -> None:
    """Verify PreprocessingPolicy rejects non-positive std values."""
    with pytest.raises(pydantic.ValidationError):
        PreprocessingPolicy(
            normalization_mean=(0.0, 0.0),
            normalization_std=(1.0, 0.0),  # non-positive std
        )


@pytest.mark.unit
def test_preprocessing_handles_none_gracefully() -> None:
    """Verify Preprocessing handles None payload gracefully."""
    prep = create_executable_preprocessing()
    assert prep(None) is None
