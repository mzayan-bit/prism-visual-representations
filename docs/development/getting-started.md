# Getting Started with PRISM

## System Requirements
- **Python**: 3.10, 3.11, 3.12, or 3.13
- **uv**: Modern Python package and environment manager (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- **Node.js**: 18.0.0 or higher (for frontend research observatory)
- **Git**: For version control and provenance tracking

---

## 1. Setting Up the Python Environment

PRISM uses `uv` for fast, deterministic dependency resolution.

```bash
# Clone repository
git clone https://github.com/mzayan-bit/prism-visual-representations.git
cd prism-visual-representations

# Create a virtual environment with uv
uv venv --python 3.11 .venv

# Activate the virtual environment
source .venv/bin/activate

# Install PRISM in editable mode with development dependencies
uv pip install -e ".[dev]"
```

Alternatively, if you use standard `make`:
```bash
make install
```

---

## 2. Setting Up the Frontend Interface

The frontend research observatory is located in `frontend/`.

```bash
cd frontend
npm install
npm run dev
```

The web interface will be available at [http://localhost:3000](http://localhost:3000).

---

## 3. Running Verification & Quality Checks

Run all checks across backend and tests:

```bash
# Run tests
pytest

# Run linting
ruff check .

# Run static type checking
mypy backend/src

# Run all checks via Makefile
make check
```

---

## 4. Configuration Template

Copy `.env.example` to `.env` to configure your local runtime environment:

```bash
cp .env.example .env
```
