.PHONY: up down reset-db verify index run eval test lint help

help:
	@echo "Available commands:"
	@echo "  make up        - Start database container (pgvector/pgvector:pg16)"
	@echo "  make down      - Stop database container"
	@echo "  make reset-db  - Wipe database volume and recreate schemas from scratch"
	@echo "  make verify    - Verify database row counts, data profile, and role security"
	@echo "  make index     - Run vector and hybrid document indexing"
	@echo "  make run       - Launch Streamlit application locally"
	@echo "  make eval      - Run evaluation benchmark (10 cases)"
	@echo "  make test      - Run full pytest test suite"
	@echo "  make lint      - Run ruff linter and code check"

up:
	docker compose up -d db

down:
	docker compose down

reset-db:
	docker compose down -v
	docker compose up -d db
	@echo "Waiting for PostgreSQL to initialize..."
	docker compose exec db sh -c 'until pg_isready -U $$POSTGRES_USER; do sleep 1; done'

verify:
	python db/verify.py

index:
	python -m src.retrieval.indexer

run:
	streamlit run app.py

eval:
	python -m evaluation.run_eval --limit 10

test:
	pytest

lint:
	ruff check .
	ruff format --check .
