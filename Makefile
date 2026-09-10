.PHONY: setup services dev ingest test eval lint fmt clean down

VENV := .venv
PY := $(VENV)/bin/python

setup:
	python3.11 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev,eval]"
	@echo "Done. Copy .env.example to .env and fill in your keys."

services:
	docker compose up -d
	@echo "Qdrant :6333  Postgres :5432  Redis :6379"

down:
	docker compose down

dev:
	$(VENV)/bin/uvicorn rag.api.main:app --reload --host 0.0.0.0 --port 8000

ingest:
	$(PY) -m rag.cli ingest $(DOCS)

test:
	$(VENV)/bin/pytest -v --cov=src/rag --cov-report=term-missing

eval:
	$(PY) -m rag.cli eval $(if $(TAGS),--tags $(TAGS),) $(if $(COMPARE),--compare $(COMPARE),)

lint:
	$(VENV)/bin/ruff check src tests
	$(VENV)/bin/mypy src

fmt:
	$(VENV)/bin/ruff format src tests
	$(VENV)/bin/ruff check --fix src tests

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
