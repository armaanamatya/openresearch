#!/usr/bin/env bash
# Docker-based GEPA smoke test — runs the full test suite inside the actual
# Docker python-deps stage (python:3.12-slim + backend/requirements.txt).
#
# Works with OrbStack or Docker Desktop. Cost: $0. Runtime: ~20s (cached
# build) or ~3 min (first build).
#
# Why python-deps only: the full image's `frontend` stage pulls in
# @rolldown/binding-linux-x64-gnu which is x64-only and breaks on arm64
# OrbStack. The frontend has nothing to do with GEPA — building just the
# Python stage is enough to prove the backend works end-to-end in container.
#
# Usage:
#   bash scripts/gepa_docker_smoke.sh
#
# To rebuild from scratch:
#   docker rmi openresearch-pydeps:gepa-test && bash scripts/gepa_docker_smoke.sh

set -euo pipefail
cd "$(dirname "$0")/.."

if ! docker info >/dev/null 2>&1; then
    echo "ERROR: Docker daemon not reachable. Start OrbStack ('orb start') or"
    echo "       Docker Desktop, then re-run." >&2
    exit 2
fi

IMAGE=openresearch-pydeps:gepa-test
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "==> building $IMAGE (python-deps stage only)"
    docker build --target python-deps -t "$IMAGE" .
else
    echo "==> using cached image $IMAGE (delete to rebuild)"
fi

echo "==> running GEPA test suite inside container"
docker run --rm -v "$PWD":/work -w /work "$IMAGE" bash -c '
    set -e
    pip install -q gepa pytest pytest-asyncio
    python -c "import gepa; print(\"gepa\", len([x for x in dir(gepa) if not x.startswith(\"_\")]), \"objects\")"
    python -c "from backend.app import create_app; app = create_app(); print(f\"FastAPI {len(app.routes)} routes\")"
    python -m pytest \
        tests/test_gepa_end_to_end.py \
        tests/test_gepa_phase2.py \
        tests/test_gepa_prompt_overrides.py \
        tests/test_gepa_reflective_shape.py \
        tests/test_worker_reports.py \
        tests/test_rubric_verifier.py \
        -q
'
