# OpenResearch / OpenResearch — convenience targets.
# The backend is Python (pytest) and the frontend is Node (in frontend/); this
# Makefile only wraps repo-wide helpers. See README.md / CLAUDE.md for the full
# command set.

.PHONY: docs-check test help

help:
	@echo "make docs-check   Verify documentation freshness & consistency (docs/policies/documentation.md)"
	@echo "make test         Run the backend test suite"

# Documentation freshness / consistency gate. Mirrors .github/workflows/docs-freshness.yml.
docs-check:
	python3 scripts/docs_freshness_check.py

test:
	.venv/bin/python -m pytest tests/
