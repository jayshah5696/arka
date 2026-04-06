set shell := ["bash", "-cu"]

_default:
    @just --list

setup:
    uv sync

test:
    uv run pytest

lint:
    uv run ruff check .

format:
    uv run ruff format .

check:
    uv run ruff check .
    uv run ruff format --check .
    uv run pytest

matrix:
    uv run pytest -q tests/integration/test_supported_options_matrix.py

validate-examples:
    uv run python scripts/validate_examples.py

run:
    uv run arka

clean:
    rm -rf .pytest_cache .ruff_cache htmlcov .coverage
