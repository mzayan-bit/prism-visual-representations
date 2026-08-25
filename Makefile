.PHONY: help install lint format typecheck test check clean frontend-install frontend-lint frontend-build

help:
	@echo "PRISM Development Commands:"
	@echo "  make install         Install backend package in editable mode with dev dependencies"
	@echo "  make lint            Run Ruff linter"
	@echo "  make format          Format code with Ruff"
	@echo "  make typecheck       Run static type checking with Mypy"
	@echo "  make test            Run test suite with Pytest"
	@echo "  make check           Run full backend validation (lint, typecheck, test)"
	@echo "  make frontend-install Install frontend dependencies"
	@echo "  make frontend-lint   Run ESLint on frontend"
	@echo "  make frontend-build  Run Next.js build on frontend"
	@echo "  make clean           Remove Python build and cache artifacts"

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

check: lint typecheck test

frontend-install:
	cd frontend && npm install

frontend-lint:
	cd frontend && npm run lint

frontend-build:
	cd frontend && npm run build

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
