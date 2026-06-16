# OpenResearch — convenience targets.
# Backend is Python (pytest), frontend is Node (in frontend/). These targets
# wrap the canonical commands documented in README.md / CLAUDE.md; they assume
# the venv at .venv/ and frontend/node_modules exist (see `make setup`).

.PHONY: help setup docs-check test test-backend test-frontend lint typecheck \
        check smoke docker-build dev-backend dev-frontend clean

help:
	@echo "make setup          Create .venv + install backend deps + npm ci (one-time)"
	@echo "make check          Everything CI runs: docs-check, backend tests, frontend lint+types+tests"
	@echo "make test           Backend test suite (alias: test-backend)"
	@echo "make test-frontend  Frontend vitest suite"
	@echo "make lint           Frontend eslint"
	@echo "make typecheck      Frontend tsc --noEmit"
	@echo "make smoke          Fast sanity: app factory boots, CLI parses, compose validates"
	@echo "make docs-check     Documentation freshness & consistency (docs/policies/documentation.md)"
	@echo "make docker-build   Build the production image"
	@echo "make dev-backend    Run the API with --reload on :8000 (preflight-aware)"
	@echo "make dev-frontend   Run the Next.js dev server on :3000"
	@echo "make clean          Remove local caches (never touches runs/ or .env)"

setup:
	python3 -m venv .venv
	.venv/bin/pip install -r backend/requirements.txt -r backend/requirements-dev.txt
	cd frontend && npm ci

# Documentation freshness / consistency gate. Mirrors .github/workflows/docs-freshness.yml.
docs-check:
	python3 scripts/docs_freshness_check.py

test: test-backend

test-backend:
	.venv/bin/python -m pytest tests/ -n auto

test-frontend:
	cd frontend && npm test

lint:
	cd frontend && npm run lint

typecheck:
	cd frontend && npx tsc --noEmit

check: docs-check test-backend lint typecheck test-frontend

# Fast, no-credentials sanity that a fresh checkout actually runs.
smoke:
	.venv/bin/python -c "from backend.app import create_app; create_app(); print('app factory OK')"
	.venv/bin/python -m backend.cli --help > /dev/null && echo "CLI OK"
	@command -v docker >/dev/null 2>&1 && docker compose config -q && echo "compose OK" \
		|| echo "compose SKIPPED (no docker on PATH)"

docker-build:
	docker build -t openresearch:dev .

dev-backend:
	./start.sh

dev-frontend:
	cd frontend && npm run dev

clean:
	rm -rf .pytest_cache .mypy_cache frontend/.next frontend/tsconfig.tsbuildinfo
	find . -name __pycache__ -type d -prune -not -path "./.venv/*" -not -path "./frontend/node_modules/*" -exec rm -rf {} + 2>/dev/null || true
