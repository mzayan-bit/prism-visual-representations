"""Unit tests verifying package structure and export integrity."""

import pkgutil
from pathlib import Path

import pytest

import prism


@pytest.mark.unit
def test_package_exports() -> None:
    """Verify that prism exports version and author in __all__."""
    assert "__version__" in prism.__all__
    assert "__author__" in prism.__all__


@pytest.mark.unit
def test_py_typed_marker_exists(repo_root: Path) -> None:
    """Verify that py.typed exists in the package source for PEP 561 compliance."""
    py_typed = repo_root / "backend" / "src" / "prism" / "py.typed"
    assert py_typed.exists()
    assert py_typed.is_file()


@pytest.mark.unit
def test_expected_submodules_exist() -> None:
    """Verify that all planned architectural submodules are discoverable."""
    expected = {
        "api",
        "core",
        "data",
        "experiments",
        "models",
        "training",
        "evaluation",
        "representations",
        "robustness",
        "explainability",
        "visualization",
        "utils",
    }
    discovered = {
        name for _, name, is_pkg in pkgutil.iter_modules(prism.__path__) if is_pkg
    }
    assert expected.issubset(discovered)
