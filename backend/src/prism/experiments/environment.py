"""Runtime host environment and package snapshotting."""

import importlib.metadata
import platform
import sys
from collections.abc import Sequence

from prism.core.metadata import EnvironmentMetadata, HardwareMetadata
from prism.experiments.hardware import probe_hardware

# Default allowlist of scientifically relevant packages
DEFAULT_ALLOWLIST_PACKAGES: tuple[str, ...] = (
    "prism",
    "pydantic",
    "pyyaml",
    "numpy",
    "torch",
    "torchvision",
    "scipy",
    "scikit-learn",
    "matplotlib",
)


def capture_environment(
    allowlist_packages: Sequence[str] | None = None,
    hardware_override: HardwareMetadata | None = None,
) -> EnvironmentMetadata:
    """Capture a structured snapshot of the runtime host environment.

    Captures the Python interpreter, operating system, hardware capabilities,
    and installed versions of allowlisted scientific dependencies.

    Args:
        allowlist_packages: Explicit list of package distribution names to probe.
                           If None, uses DEFAULT_ALLOWLIST_PACKAGES.
        hardware_override: Optional pre-probed HardwareMetadata for testing.

    Returns:
        Structured EnvironmentMetadata object.
    """
    python_version = sys.version.split()[0]
    python_impl = platform.python_implementation()
    os_name = f"{platform.system()} {platform.release()}".strip()
    platform_desc = platform.platform()

    hardware_info = hardware_override or probe_hardware()

    # Format human-readable hardware string
    hw_desc_parts: list[str] = []
    if hardware_info.cpu_count:
        hw_desc_parts.append(f"{hardware_info.cpu_count} CPU cores")
    if hardware_info.cpu_architecture:
        hw_desc_parts.append(hardware_info.cpu_architecture)
    if hardware_info.cuda_available:
        gpus_str = f"{hardware_info.cuda_device_count} GPUs"
        hw_desc_parts.append(f"CUDA ({gpus_str})")
    elif hardware_info.mps_available:
        hw_desc_parts.append("Apple Silicon MPS")
    else:
        hw_desc_parts.append("CPU Backend")
    hardware_summary = ", ".join(hw_desc_parts)

    packages_to_check = allowlist_packages or DEFAULT_ALLOWLIST_PACKAGES
    installed_packages: dict[str, str] = {}

    for pkg_name in packages_to_check:
        try:
            pkg_ver = importlib.metadata.version(pkg_name)
            installed_packages[pkg_name] = pkg_ver
        except importlib.metadata.PackageNotFoundError:
            continue
        except Exception:
            continue

    return EnvironmentMetadata(
        python_version=python_version,
        python_implementation=python_impl,
        os=os_name,
        platform=platform_desc,
        hardware=hardware_summary,
        hardware_info=hardware_info,
        cuda_version=hardware_info.cuda_version,
        packages=installed_packages,
    )
