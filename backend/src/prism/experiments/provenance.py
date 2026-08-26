"""Lightweight Git version control provenance and working tree inspection."""

import subprocess
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from prism.core.metadata import CodeRevisionMetadata


class GitProvenance(BaseModel):
    """Structured report detailing git source code provenance and cleanliness."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    git_available: bool = Field(
        default=False,
        description="Whether the git executable was found and callable",
    )
    is_repo: bool = Field(
        default=False,
        description="Whether working directory is inside a valid git repo",
    )
    commit_sha: str | None = Field(
        default=None,
        description="Full commit SHA-1/SHA-256 hash",
    )
    short_sha: str | None = Field(
        default=None,
        description="Shortened commit hash (7-8 characters)",
    )
    branch: str | None = Field(
        default=None,
        description="Current checked out branch name or 'HEAD'",
    )
    is_dirty: bool = Field(
        default=False,
        description="True if tracked files have uncommitted modifications",
    )
    repository_url: str | None = Field(
        default=None,
        description="Remote repository origin URL if configured",
    )
    modified_files: list[str] = Field(
        default_factory=list,
        description="Relative paths of modified tracked files",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Warnings encountered during provenance inspection",
    )

    def to_code_revision_metadata(self) -> CodeRevisionMetadata:
        """Convert provenance inspection to CodeRevisionMetadata schema."""
        return CodeRevisionMetadata(
            git_commit=self.commit_sha,
            short_commit=self.short_sha,
            git_branch=self.branch,
            is_dirty=self.is_dirty,
            repository_url=self.repository_url,
            modified_files=self.modified_files,
        )


def _run_git_cmd(
    args: list[str],
    cwd: Path | str | None = None,
    timeout: float = 2.0,
) -> tuple[int, str, str]:
    """Execute a single git command safely using subprocess."""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except FileNotFoundError:
        return -1, "", "git executable not found in system PATH"
    except subprocess.TimeoutExpired:
        return -2, "", f"git command '{' '.join(args)}' timed out after {timeout}s"
    except Exception as exc:
        return -3, "", f"git execution error: {exc}"


def inspect_git_provenance(
    cwd: Path | str | None = None,
    timeout: float = 2.0,
    runner: Any | None = None,
) -> GitProvenance:
    """Inspect and capture Git source code provenance safely.

    Discovers commit hashes, branch, remote URL, and working tree cleanliness.
    Tracked modified files are captured, while untracked files are excluded to
    prevent indexing arbitrary or private scratch files.

    Args:
        cwd: Working directory from which to run git commands.
        timeout: Subprocess execution timeout in seconds.
        runner: Optional callable `(args, cwd, timeout) -> (code, stdout, stderr)`
                for unit testing and mocking.

    Returns:
        GitProvenance instance describing source code revision status.
    """
    exec_git = runner or _run_git_cmd
    warnings: list[str] = []

    # 1. Test git availability & repo presence
    code, out, err = exec_git(["rev-parse", "--is-inside-work-tree"], cwd, timeout)
    if code == -1:
        return GitProvenance(
            git_available=False,
            is_repo=False,
            warnings=["Git executable is not installed or not in PATH."],
        )
    if code != 0 or out != "true":
        detail = err or "Not a git repo"
        return GitProvenance(
            git_available=True,
            is_repo=False,
            warnings=[f"Directory is not inside a git repository: {detail}"],
        )

    # 2. Get full commit SHA
    code, commit_sha, _ = exec_git(["rev-parse", "HEAD"], cwd, timeout)
    if code != 0 or not commit_sha:
        commit_sha = None
        warnings.append("Could not determine git commit SHA.")

    # 3. Get short SHA
    code, short_sha, _ = exec_git(["rev-parse", "--short", "HEAD"], cwd, timeout)
    if code != 0 or not short_sha:
        short_sha = None

    # 4. Get active branch
    code, branch, _ = exec_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd, timeout)
    if code != 0 or not branch:
        branch = None

    # 5. Check working tree cleanliness (only tracked files: -uno)
    code, status_out, _ = exec_git(["status", "--porcelain", "-uno"], cwd, timeout)
    is_dirty = False
    modified_files: list[str] = []
    if code == 0 and status_out:
        is_dirty = True
        for line in status_out.splitlines():
            line = line.strip()
            if len(line) >= 3:
                file_path = line[2:].strip()
                if file_path:
                    modified_files.append(file_path)

    # 6. Get repository remote URL
    code, repo_url, _ = exec_git(["config", "--get", "remote.origin.url"], cwd, timeout)
    if code != 0 or not repo_url:
        repo_url = None

    return GitProvenance(
        git_available=True,
        is_repo=True,
        commit_sha=commit_sha,
        short_sha=short_sha,
        branch=branch,
        is_dirty=is_dirty,
        repository_url=repo_url,
        modified_files=modified_files,
        warnings=warnings,
    )
