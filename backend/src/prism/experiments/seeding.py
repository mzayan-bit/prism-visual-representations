"""Deterministic multi-backend RNG state management and seeding results."""

import os
import random
import sys
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from prism.core.errors import ReproducibilityError


def _utc_now() -> datetime:
    """Return timezone-aware current UTC datetime."""
    return datetime.now(timezone.utc)


class SeedInitializationResult(BaseModel):
    """Structured report detailing RNG configuration across all available backends."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    requested_seed: int = Field(description="The integer seed requested by caller")
    deterministic_requested: bool = Field(
        description="Whether deterministic execution was requested"
    )
    python_seeded: bool = Field(
        default=True,
        description="Whether Python's random and PYTHONHASHSEED were configured",
    )
    numpy_available: bool = Field(
        default=False,
        description="Whether NumPy was detected in the environment",
    )
    numpy_seeded: bool = Field(
        default=False,
        description="Whether NumPy RNG was seeded",
    )
    torch_available: bool = Field(
        default=False,
        description="Whether PyTorch was detected in the environment",
    )
    torch_seeded: bool = Field(
        default=False,
        description="Whether PyTorch CPU RNG was seeded",
    )
    cuda_available: bool = Field(
        default=False,
        description="Whether CUDA GPU acceleration was detected",
    )
    cuda_seeded: bool = Field(
        default=False,
        description="Whether CUDA device RNGs were seeded",
    )
    mps_available: bool = Field(
        default=False,
        description="Whether Apple Silicon MPS acceleration was detected",
    )
    deterministic_algorithms_configured: bool = Field(
        default=False,
        description="Whether strict deterministic algorithm modes were applied",
    )
    configured_backends: list[str] = Field(
        default_factory=list,
        description="Names of successfully seeded backends (e.g. ['python'])",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Warnings regarding non-critical seeding limitations",
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="Transparent declarations of inherent hardware limits",
    )
    timestamp: datetime = Field(
        default_factory=_utc_now,
        description="UTC timestamp when seeding was completed",
    )


def initialize_seeds(
    seed: int,
    deterministic: bool = True,
    strict: bool = False,
    torch_module: Any | None = None,
    numpy_module: Any | None = None,
) -> SeedInitializationResult:
    """Centrally seed all available RNG backends and configure deterministic execution.

    Configures:
    1. Python standard library `random` and `os.environ["PYTHONHASHSEED"]`.
    2. NumPy `numpy.random.seed(seed)` if available.
    3. PyTorch `torch.manual_seed(seed)`, `torch.cuda.manual_seed_all(seed)`,
       cuDNN deterministic flags, and `torch.use_deterministic_algorithms(True)`.

    Args:
        seed: Master non-negative integer seed.
        deterministic: If True, request deterministic algorithmic modes.
        strict: If True, raise ReproducibilityError if a requested deterministic mode
                cannot be enforced.
        torch_module: Optional pre-imported torch module or mock for testing.
        numpy_module: Optional pre-imported numpy module or mock for testing.

    Returns:
        SeedInitializationResult summarizing configured backends and limitations.

    Raises:
        ReproducibilityError: If strict is True and deterministic modes cannot be set.
    """
    configured_backends: list[str] = []
    warnings: list[str] = []
    limitations: list[str] = []

    # 1. Python standard library
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    configured_backends.append("python")
    python_seeded = True

    # 2. NumPy backend
    numpy_available = False
    numpy_seeded = False
    np = numpy_module
    if np is None and "numpy" in sys.modules:
        np = sys.modules["numpy"]
    elif np is None:
        try:
            import importlib

            np = importlib.import_module("numpy")
        except ImportError:
            np = None

    if np is not None:
        numpy_available = True
        try:
            np.random.seed(seed)
            numpy_seeded = True
            configured_backends.append("numpy")
        except Exception as exc:
            warnings.append(f"Failed to seed NumPy: {exc}")
    else:
        limitations.append(
            "NumPy is not installed in the environment; array operations are unseeded."
        )

    # 3. PyTorch backend
    torch_available = False
    torch_seeded = False
    cuda_available = False
    cuda_seeded = False
    mps_available = False
    deterministic_configured = False

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
        torch_available = True
        try:
            torch.manual_seed(seed)
            torch_seeded = True
            configured_backends.append("torch_cpu")
        except Exception as exc:
            warnings.append(f"Failed to seed PyTorch CPU: {exc}")

        # CUDA check & seeding
        try:
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                cuda_available = True
                torch.cuda.manual_seed_all(seed)
                cuda_seeded = True
                configured_backends.append("torch_cuda")
                limitations.append(
                    "Certain CUDA operations (e.g. atomicAdd) may exhibit "
                    "non-deterministic floating-point behavior across GPUs."
                )
        except Exception as exc:
            warnings.append(f"Failed to seed CUDA: {exc}")

        # MPS check
        try:
            if (
                hasattr(torch, "backends")
                and hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                mps_available = True
                limitations.append(
                    "Apple Silicon MPS backend may have platform-specific "
                    "non-deterministic operations across macOS versions."
                )
        except Exception:
            mps_available = False

        # Deterministic algorithms configuration
        if deterministic:
            det_errors: list[str] = []
            try:
                if hasattr(torch, "use_deterministic_algorithms"):
                    torch.use_deterministic_algorithms(True)
                    deterministic_configured = True
            except Exception as exc:
                det_errors.append(f"torch.use_deterministic_algorithms: {exc}")

            try:
                if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
                    deterministic_configured = True
            except Exception as exc:
                det_errors.append(f"torch.backends.cudnn: {exc}")

            if det_errors:
                msg = f"Deterministic algorithms could not be enabled: {det_errors}"
                warnings.append(msg)
                if strict:
                    raise ReproducibilityError(msg)
    else:
        limitations.append(
            "PyTorch is not installed in the environment; tensor RNG and cuDNN "
            "determinism cannot be configured."
        )
        if deterministic and strict:
            raise ReproducibilityError(
                "Strict deterministic mode requested, but PyTorch is not installed."
            )

    return SeedInitializationResult(
        requested_seed=seed,
        deterministic_requested=deterministic,
        python_seeded=python_seeded,
        numpy_available=numpy_available,
        numpy_seeded=numpy_seeded,
        torch_available=torch_available,
        torch_seeded=torch_seeded,
        cuda_available=cuda_available,
        cuda_seeded=cuda_seeded,
        mps_available=mps_available,
        deterministic_algorithms_configured=deterministic_configured,
        configured_backends=configured_backends,
        warnings=warnings,
        limitations=limitations,
    )
