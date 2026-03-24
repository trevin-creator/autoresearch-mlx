.PHONY: install install-dev test lint clean sync run

install:
	uv sync --no-dev

install-dev:
	uv sync --group dev

test:
	uv run pytest

lint:
	uv run ruff check .

clean:
	rm -rf .venv __pycache__ .pytest_cache .ruff_cache *.egg-info dist build
	find . -name '*.pyc' -delete
	find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

sync:
	uv sync

run:
	uv run python train.py
