#!/usr/bin/env bash
# Back-compat shim: e2e specs historically referenced this name.
set -euo pipefail
cd "$(dirname "$0")/.."
exec "${PYTHON:-.venv/bin/python}" scripts/seed_fake_run.py "$@"
