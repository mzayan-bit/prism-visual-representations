"""Pytest configuration and shared fixtures for PRISM tests."""

from pathlib import Path

import pytest


@pytest.fixture
def repo_root() -> Path:
    """Return the absolute path to the repository root."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def backend_src(repo_root: Path) -> Path:
    """Return the absolute path to backend/src."""
    return repo_root / "backend" / "src"
