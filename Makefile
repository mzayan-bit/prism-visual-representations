.PHONY: help setup install lint format typecheck test check dev demo clean frontend-install frontend-lint frontend-build

help:
	@echo "PRISM Research Platform Commands:"
	@echo "  make setup           Install backend & frontend dependencies"
	@echo "  make install         Install backend package in editable mode with dev dependencies"
	@echo "  make dev             Launch the PRISM Research Observatory frontend"
	@echo "  make demo            Generate deterministic demo campaign & research reports"
	@echo "  make lint            Run Ruff linter and format checking"
	@echo "  make format          Format code with Ruff"
	@echo "  make typecheck       Run static type checking with Mypy"
	@echo "  make test            Run test suite with Pytest"
	@echo "  make check           Run full quality gates (backend + frontend)"
	@echo "  make frontend-install Install frontend dependencies"
	@echo "  make frontend-lint   Run ESLint on frontend"
	@echo "  make frontend-build  Run Next.js production build on frontend"
	@echo "  make clean           Remove Python build and cache artifacts"

setup: install frontend-install

install:
	uv pip install -e ".[dev]"

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run mypy

test:
	uv run pytest

check: lint typecheck test frontend-lint frontend-build

dev:
	cd frontend && npm run dev

demo:
	uv run python scripts/generate_demo.py
	@echo ""
	@echo "========================================================"
	@echo " PRISM Demo Campaign Generated Successfully!            "
	@echo " To view the Research Observatory, run: make dev        "
	@echo " Open http://localhost:3000 in your browser.           "
	@echo "========================================================"

frontend-install:
	cd frontend && npm install

frontend-lint:
	cd frontend && npm run lint

frontend-build:
	cd frontend && npm run build

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
