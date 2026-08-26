"""Hardware and compute backend probing without heavy tensor allocations."""

import os
import platform
import sys
from typing import Any

from prism.core.metadata import HardwareMetadata


def probe_hardware(torch_module: Any | None = None) -> HardwareMetadata:
    """Probe system hardware and compute backend capabilities safely.

    This function discovers CPU and optional GPU acceleration (CUDA, Apple Silicon MPS)
    without allocating tensors, downloading weights, or initiating workloads.

    Args:
        torch_module: Optional pre-imported torch module or mock for testing.
                      If None, will attempt a safe local import.

    Returns:
        HardwareMetadata populated with detected capabilities.
    """
    cpu_count = os.cpu_count()
    cpu_arch = platform.machine() or None

    cuda_available = False
    cuda_device_count = 0
    cuda_device_names: list[str] = []
    cuda_version: str | None = None
    mps_available = False
    mps_built = False
    compute_backend = "cpu"

    # Attempt to resolve torch module safely
    torch = torch_module
    if torch is None and "torch" in sys.modules:
        torch = sys.modules["torch"]
    elif torch is None:
        try:
            import importlib

            torch = importlib.import_module("torch")
        except ImportError:
            torch = None

    if torch is not None:
        # Check CUDA capabilities
        try:
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                cuda_available = True
                cuda_device_count = torch.cuda.device_count()
                for i in range(cuda_device_count):
                    try:
                        name = torch.cuda.get_device_name(i)
                        cuda_device_names.append(str(name))
                    except Exception:
                        cuda_device_names.append(f"CUDA Device {i}")

                if hasattr(torch, "version") and hasattr(torch.version, "cuda"):
                    cuda_version = torch.version.cuda
                compute_backend = "cuda"
        except Exception:
            cuda_available = False

        # Check Apple Silicon MPS capabilities
        try:
            if hasattr(torch, "backends") and hasattr(torch.backends, "mps"):
                mps_built = bool(torch.backends.mps.is_built())
                mps_available = bool(torch.backends.mps.is_available())
                if mps_available and not cuda_available:
                    compute_backend = "mps"
        except Exception:
            mps_available = False
            mps_built = False

    return HardwareMetadata(
        cpu_count=cpu_count,
        cpu_architecture=cpu_arch,
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
        cuda_device_names=cuda_device_names,
        cuda_version=cuda_version,
        mps_available=mps_available,
        mps_built=mps_built,
        compute_backend=compute_backend,
    )
