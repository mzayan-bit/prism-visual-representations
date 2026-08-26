"""Unit tests for Git source code provenance and cleanliness inspection."""

from pathlib import Path

import pytest

from prism.experiments.provenance import GitProvenance, inspect_git_provenance


@pytest.mark.unit
def test_inspect_git_provenance_on_active_repo() -> None:
    """Verify git provenance inspection works on the current repository."""
    prov = inspect_git_provenance()

    assert isinstance(prov, GitProvenance)
    assert prov.git_available is True
    assert prov.is_repo is True
    assert prov.commit_sha is not None
    assert len(prov.commit_sha) >= 40
    assert prov.short_sha is not None
    assert prov.branch is not None

    meta = prov.to_code_revision_metadata()
    assert meta.git_commit == prov.commit_sha
    assert meta.git_branch == prov.branch


@pytest.mark.unit
def test_mock_git_clean_repository() -> None:
    """Verify git provenance correctly parses clean repository responses."""

    def mock_runner(
        args: list[str],
        cwd: Path | str | None,
        timeout: float,
    ) -> tuple[int, str, str]:
        cmd = " ".join(args)
        if cmd == "rev-parse --is-inside-work-tree":
            return 0, "true", ""
        if cmd == "rev-parse HEAD":
            return 0, "a" * 40, ""
        if cmd == "rev-parse --short HEAD":
            return 0, "aaaaaaa", ""
        if cmd == "rev-parse --abbrev-ref HEAD":
            return 0, "main", ""
        if cmd == "status --porcelain -uno":
            return 0, "", ""
        if cmd == "config --get remote.origin.url":
            return 0, "https://github.com/mzayan-bit/prism.git", ""
        return 0, "", ""

    prov = inspect_git_provenance(runner=mock_runner)

    assert prov.git_available is True
    assert prov.is_repo is True
    assert prov.commit_sha == "a" * 40
    assert prov.short_sha == "aaaaaaa"
    assert prov.branch == "main"
    assert prov.is_dirty is False
    assert prov.modified_files == []
    assert prov.repository_url == "https://github.com/mzayan-bit/prism.git"


@pytest.mark.unit
def test_mock_git_dirty_repository() -> None:
    """Verify git provenance captures modified tracked files when dirty."""

    def mock_runner(
        args: list[str],
        cwd: Path | str | None,
        timeout: float,
    ) -> tuple[int, str, str]:
        cmd = " ".join(args)
        if cmd == "rev-parse --is-inside-work-tree":
            return 0, "true", ""
        if cmd == "rev-parse HEAD":
            return 0, "b" * 40, ""
        if cmd == "rev-parse --short HEAD":
            return 0, "bbbbbbb", ""
        if cmd == "rev-parse --abbrev-ref HEAD":
            return 0, "feature/probe", ""
        if cmd == "status --porcelain -uno":
            return 0, " M backend/src/prism/core/metadata.py\n M README.md", ""
        return 0, "", ""

    prov = inspect_git_provenance(runner=mock_runner)

    assert prov.is_dirty is True
    assert "backend/src/prism/core/metadata.py" in prov.modified_files
    assert "README.md" in prov.modified_files


@pytest.mark.unit
def test_mock_git_not_installed() -> None:
    """Verify graceful handling when git executable is not found."""

    def mock_runner(
        args: list[str],
        cwd: Path | str | None,
        timeout: float,
    ) -> tuple[int, str, str]:
        return -1, "", "git executable not found in system PATH"

    prov = inspect_git_provenance(runner=mock_runner)

    assert prov.git_available is False
    assert prov.is_repo is False
    assert any("not installed" in w for w in prov.warnings)


@pytest.mark.unit
def test_mock_directory_not_a_git_repo() -> None:
    """Verify graceful handling when directory is not inside a git repo."""

    def mock_runner(
        args: list[str],
        cwd: Path | str | None,
        timeout: float,
    ) -> tuple[int, str, str]:
        return 128, "", "fatal: not a git repository"

    prov = inspect_git_provenance(runner=mock_runner)

    assert prov.git_available is True
    assert prov.is_repo is False
    assert any("not inside a git repository" in w for w in prov.warnings)
