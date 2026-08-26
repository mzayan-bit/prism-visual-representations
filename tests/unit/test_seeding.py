"""Unit tests for deterministic multi-backend pseudo-random number generator seeding."""

import random
from unittest.mock import MagicMock

import pytest

from prism.core.errors import ReproducibilityError
from prism.experiments.seeding import (
    SeedInitializationResult,
    initialize_seeds,
)


@pytest.mark.unit
def test_python_random_seeding_reproducibility() -> None:
    """Verify that identical seeds produce identical Python random sequences."""
    seed = 42

    initialize_seeds(seed=seed, deterministic=False)
    seq1 = [random.random() for _ in range(5)]

    initialize_seeds(seed=seed, deterministic=False)
    seq2 = [random.random() for _ in range(5)]

    assert seq1 == seq2

    # Different seed should produce different sequence
    initialize_seeds(seed=999, deterministic=False)
    seq3 = [random.random() for _ in range(5)]
    assert seq1 != seq3


@pytest.mark.unit
def test_seed_initialization_result_structure() -> None:
    """Verify attributes and types of SeedInitializationResult."""
    result = initialize_seeds(seed=1234, deterministic=False)

    assert isinstance(result, SeedInitializationResult)
    assert result.requested_seed == 1234
    assert result.deterministic_requested is False
    assert result.python_seeded is True
    assert "python" in result.configured_backends
    assert isinstance(result.limitations, list)
    assert isinstance(result.warnings, list)


@pytest.mark.unit
def test_graceful_degradation_without_optional_backends() -> None:
    """Verify seeding succeeds cleanly when NumPy and PyTorch are not installed."""
    result = initialize_seeds(
        seed=42,
        deterministic=False,
        torch_module=None,
        numpy_module=None,
    )

    assert result.python_seeded is True
    assert result.numpy_seeded is False
    assert result.torch_seeded is False
    assert result.cuda_seeded is False
    assert "python" in result.configured_backends
    assert any("NumPy is not installed" in limit for limit in result.limitations)
    assert any("PyTorch is not installed" in limit for limit in result.limitations)


@pytest.mark.unit
def test_mock_numpy_seeding() -> None:
    """Verify NumPy seeding is called when NumPy module is provided."""
    mock_np = MagicMock()

    result = initialize_seeds(
        seed=100,
        deterministic=False,
        numpy_module=mock_np,
        torch_module=None,
    )

    assert result.numpy_available is True
    assert result.numpy_seeded is True
    assert "numpy" in result.configured_backends
    mock_np.random.seed.assert_called_once_with(100)


@pytest.mark.unit
def test_mock_torch_seeding_cpu_and_cuda() -> None:
    """Verify PyTorch CPU and CUDA seeding when mock torch with CUDA is present."""
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.backends.mps.is_available.return_value = False

    result = initialize_seeds(
        seed=777,
        deterministic=True,
        torch_module=mock_torch,
        numpy_module=None,
    )

    assert result.torch_available is True
    assert result.torch_seeded is True
    assert result.cuda_available is True
    assert result.cuda_seeded is True
    assert result.deterministic_algorithms_configured is True
    assert "torch_cpu" in result.configured_backends
    assert "torch_cuda" in result.configured_backends

    mock_torch.manual_seed.assert_called_once_with(777)
    mock_torch.cuda.manual_seed_all.assert_called_once_with(777)
    mock_torch.use_deterministic_algorithms.assert_called_once_with(True)
    assert mock_torch.backends.cudnn.deterministic is True
    assert mock_torch.backends.cudnn.benchmark is False


@pytest.mark.unit
def test_strict_reproducibility_raises_when_backend_missing() -> None:
    """Verify strict mode raises ReproducibilityError when PyTorch is missing."""
    with pytest.raises(
        ReproducibilityError, match="Strict deterministic mode requested"
    ):
        initialize_seeds(
            seed=42,
            deterministic=True,
            strict=True,
            torch_module=None,
        )


@pytest.mark.unit
def test_strict_reproducibility_raises_when_deterministic_fails() -> None:
    """Verify strict mode raises ReproducibilityError when torch deterministic fails."""
    mock_torch = MagicMock()
    mock_torch.use_deterministic_algorithms.side_effect = RuntimeError(
        "Deterministic algorithms not supported for current device"
    )

    with pytest.raises(
        ReproducibilityError,
        match="Deterministic algorithms could not be enabled",
    ):
        initialize_seeds(
            seed=42,
            deterministic=True,
            strict=True,
            torch_module=mock_torch,
        )
