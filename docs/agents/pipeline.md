# `pipeline`

> **Module:** [`backend/agents/pipeline.py`](../../backend/agents/pipeline.py).

> **STUB** — infra, not a registered agent.

## Purpose
Top-level entry points that bootstrap a run end-to-end.

## Accepts
TODO — `project_id`, `runs_root`, workspace state.

Functions:
- `async run_pipeline_sdk(...)` — full LLM-driven pipeline.
- `run_pipeline_offline(...)` — deterministic offline pipeline (CI/tests).
- `_write_workspace_claim_map(...)` — prepares the `workspace_claim_map` consumed by `paper-understanding`.
- `_truncate_excerpt(text, max_chars=600)` — truncation rule for parsed-PDF excerpts.

## Emits
TODO — `FinalReport`; side effect: complete run tree under `<runs_root>/<project_id>/`.

## Source
- [`backend/agents/pipeline.py`](../../backend/agents/pipeline.py)
