"""Smoke tests verifying that PRISM and all core submodules can be imported cleanly."""

import importlib

import pytest


@pytest.mark.smoke
def test_import_prism() -> None:
    """Verify that the top-level prism package imports and exposes metadata."""
    import prism

    assert hasattr(prism, "__version__")
    assert isinstance(prism.__version__, str)
    assert len(prism.__version__) > 0
    assert hasattr(prism, "__author__")


@pytest.mark.smoke
@pytest.mark.parametrize(
    "submodule",
    [
        "prism.api",
        "prism.core",
        "prism.data",
        "prism.experiments",
        "prism.models",
        "prism.training",
        "prism.evaluation",
        "prism.representations",
        "prism.robustness",
        "prism.explainability",
        "prism.visualization",
        "prism.utils",
    ],
)
def test_import_submodules(submodule: str) -> None:
    """Verify that every architectural subpackage is importable without errors."""
    mod = importlib.import_module(submodule)
    assert mod is not None
    assert hasattr(mod, "__doc__")
    assert mod.__doc__ is not None
