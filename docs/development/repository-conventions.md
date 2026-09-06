# PRISM Repository Conventions

## Coding Standards & Architectural Guidelines

### Python Code Style
- **Formatter & Linter**: [Ruff](https://docs.astral.sh/ruff/) with standard 88-character line length limit.
- **Type Annotations**: Strict static typing enforced by `mypy`. All public module interfaces, function signatures, and class attributes must be fully typed.
- **Data Contracts**: Use Pydantic models for structured configurations and domain entities.
- **Docstrings**: All modules, classes, and public functions must include standard docstrings detailing parameter semantics and research invariants.

---

### Git & Commit Conventions
- **Commit Messages**: Write natural, concise, professional, and human-written sentence-style commit messages (e.g. `Initialize the PRISM project foundation`, `Add PRISM research documentation and methodology`).
- **No Prefixes**: Avoid conventional commit prefixes (such as `feat:`, `fix:`, `chore:`, `docs:`, `ci:`).
- **Atomic Commits**: Each commit must represent a coherent, buildable, and reviewable milestone.
- **Clean Staging**: Always verify with `git status` before committing. Never stage generated cache files, secrets, node modules, virtual environments, or heavy checkpoints.

---

### Testing Standards
- **Smoke Tests (`tests/smoke/`)**: Fast sanity tests verifying that all modules import cleanly without side effects.
- **Unit Tests (`tests/unit/`)**: Isolated unit tests for core math algorithms (e.g., CKA, metric evaluators, config validators).
- **Integration Tests (`tests/integration/`)**: Multi-module workflows verifying end-to-end configuration parsing and pipeline orchestration.

---

### Data and Storage Management
- Follow the [Research Contract](../methodology/research-contract.md) data policy at all times.
- Never commit binary checkpoints (`*.pt`, `*.pth`, `*.ckpt`) or dataset images to Git.
