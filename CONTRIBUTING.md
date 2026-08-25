# Contributing to PRISM

Thank you for your interest in contributing to PRISM (Probing the Evolution of Visual Representations).

As an open-source, research-oriented computer vision platform, we hold our codebase to rigorous scientific and software engineering standards.

---

## Code of Conduct
Please read and abide by our [Code of Conduct](CODE_OF_CONDUCT.md) in all project interactions.

---

## Research Integrity & Guidelines
Before proposing or implementing new models, datasets, or evaluation protocols, please review the [PRISM Research Contract](docs/methodology/research-contract.md). All experiments must adhere to:
1. **Fair Comparison**: Controlled, equal-budget baselines.
2. **Reproducibility**: Deterministic seeding, fixed dataset fingerprints, and recorded execution parameters.
3. **No Silent Comparisons**: Transparent documentation of all augmentations, preprocessing, and training splits.

---

## Development Workflow

1. **Fork and Clone** the repository:
   ```bash
   git clone https://github.com/mzayan-bit/prism-visual-representations.git
   cd prism-visual-representations
   ```

2. **Set up the environment**:
   ```bash
   uv venv --python 3.11 .venv
   source .venv/bin/activate
   uv pip install -e ".[dev]"
   ```

3. **Verify local checks**:
   ```bash
   make check
   ```

4. **Branching & Commits**:
   - Create a feature branch: `git checkout -b your-feature-name`
   - Write natural, concise, sentence-style commit messages (do not use conventional commit prefixes like `feat:` or `fix:`).
   - Ensure all tests pass, linting succeeds, and types validate before submitting a Pull Request.
