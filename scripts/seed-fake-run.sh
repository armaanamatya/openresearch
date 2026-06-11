#!/usr/bin/env bash
# Back-compat shim: e2e specs historically referenced this name.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -n "${PYTHON:-}" ]]; then PY="$PYTHON"
elif [[ -x .venv/bin/python ]]; then PY=.venv/bin/python
else PY=python3
fi
exec "$PY" scripts/seed_fake_run.py "$@"
