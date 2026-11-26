.PHONY: check lint test licenses

check: lint test licenses

lint:
	uv run ruff check src tests scripts
	uv run ruff format --check src tests scripts

test:
	uv run pytest -q

licenses:
	@if [ -f scripts/check_licenses.py ]; then uv run python scripts/check_licenses.py; else echo "license gate non ancora presente"; fi
