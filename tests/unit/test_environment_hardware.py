"""Unit tests for environment snapshotting and hardware capability probing."""

from unittest.mock import MagicMock

import pytest

from prism.core.metadata import EnvironmentMetadata, HardwareMetadata
from prism.experiments.environment import capture_environment
from prism.experiments.hardware import probe_hardware


@pytest.mark.unit
def test_probe_hardware_cpu_default() -> None:
    """Verify hardware probe detects CPU cores and architecture on CPU-only system."""
    hw = probe_hardware(torch_module=None)

    assert isinstance(hw, HardwareMetadata)
    assert hw.cpu_count is None or hw.cpu_count > 0
    assert hw.compute_backend == "cpu"
    assert hw.cuda_available is False
    assert hw.cuda_device_count == 0


@pytest.mark.unit
def test_probe_hardware_mock_cuda() -> None:
    """Verify hardware probe populates CUDA details when mock CUDA is available."""
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.device_count.return_value = 2
    mock_torch.cuda.get_device_name.side_effect = lambda i: f"NVIDIA RTX {i}"
    mock_torch.version.cuda = "12.1"

    hw = probe_hardware(torch_module=mock_torch)

    assert hw.cuda_available is True
    assert hw.cuda_device_count == 2
    assert hw.cuda_device_names == ["NVIDIA RTX 0", "NVIDIA RTX 1"]
    assert hw.cuda_version == "12.1"
    assert hw.compute_backend == "cuda"


@pytest.mark.unit
def test_probe_hardware_mock_mps() -> None:
    """Verify hardware probe detects Apple Silicon MPS capabilities."""
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.backends.mps.is_built.return_value = True
    mock_torch.backends.mps.is_available.return_value = True

    hw = probe_hardware(torch_module=mock_torch)

    assert hw.mps_built is True
    assert hw.mps_available is True
    assert hw.cuda_available is False
    assert hw.compute_backend == "mps"


@pytest.mark.unit
def test_capture_environment_structure() -> None:
    """Verify capture_environment captures interpreter, platform, and packages."""
    env = capture_environment(allowlist_packages=["prism", "pydantic"])

    assert isinstance(env, EnvironmentMetadata)
    assert isinstance(env.python_version, str)
    assert isinstance(env.os, str)
    assert env.hardware_info is not None
    assert "pydantic" in env.packages
    assert "prism" in env.packages


@pytest.mark.unit
def test_environment_and_hardware_serialization_round_trip() -> None:
    """Verify serialization and deserialization of EnvironmentMetadata."""
    env = capture_environment()
    dumped = env.model_dump(mode="json")
    restored = EnvironmentMetadata.model_validate(dumped)

    assert restored.python_version == env.python_version
    assert restored.os == env.os
    assert restored.hardware == env.hardware
