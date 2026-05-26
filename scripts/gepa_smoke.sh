#!/usr/bin/env bash
# Local GEPA smoke test — exercises the FULL gepa.optimize() loop against
# ReproLab's adapter without any API spend.
#
# Works around the brew Py3.14.5 pyexpat ABI bug (which blocks pip on the
# default .venv) by creating a parallel .venv-gepa using Py3.12 + uv.
#
# Usage:
#   bash scripts/gepa_smoke.sh
#
# Runtime: ~1s. Cost: $0.

set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null; then
    echo "ERROR: uv not installed. Install with 'brew install uv' or 'pip install uv'." >&2
    exit 2
fi

VENV=.venv-gepa
if [[ ! -d "$VENV" ]]; then
    echo "==> creating $VENV with Python 3.12"
    uv venv --python 3.12 "$VENV"
fi

echo "==> installing gepa + backend deps into $VENV"
uv pip install --python "$VENV/bin/python" -q gepa -r backend/requirements.txt pytest pytest-asyncio

echo "==> running full GEPA test suite (including real gepa.optimize end-to-end)"
"$VENV/bin/python" -m pytest \
    tests/test_gepa_end_to_end.py \
    tests/test_gepa_phase2.py \
    tests/test_gepa_prompt_overrides.py \
    tests/test_worker_reports.py \
    tests/test_rubric_verifier.py \
    -q "$@"
