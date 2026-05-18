# `orchestrator`

> **Module:** [`backend/agents/orchestrator.py`](../../backend/agents/orchestrator.py).

> **STUB** — infra, not a registered agent.

## Purpose
Drives the pipeline through its stages, maintains `PipelineState`, decides when to re-iterate / rebuild, and wraps each agent call with resilience.

## Accepts
TODO — `project_id`, `runs_root`, run config. Constructs/holds `PipelineState`.

## Emits
TODO — mutates `PipelineState`; emits stage transitions to the dashboard via `dashboard-emitter`. Final artifact: `FinalReport`.

Key types:
- `PipelineStage` (enum): stage names
- `PipelineState`: full per-run state object
- `AgentExecutionTrace`: per-agent invocation record
- `ReproLabOrchestrator`: top-level controller

## Source
- [`backend/agents/orchestrator.py`](../../backend/agents/orchestrator.py)
- Re-iteration logic: `_should_reiterate`, `_should_rebuild`
