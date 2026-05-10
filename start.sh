#!/bin/bash
set -e
cd "$(dirname "$0")"
exec .venv/bin/uvicorn backend.app:create_app --factory --reload --port 8000
