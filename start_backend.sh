#!/bin/sh

.venv/bin/uvicorn backend.app:create_app --factory --reload --port 8000
