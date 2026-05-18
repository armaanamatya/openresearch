# `telemetry`

> **Module:** [`backend/agents/telemetry.py`](../../backend/agents/telemetry.py).

> **STUB** — infra, not a registered agent.

## Purpose
Observability hooks around agent execution: timing, token usage, tool-call counts, run-level metadata. Companion to [`backend/observability/`](../../backend/observability/) (Tier 1 logging) and `docs/design/tier2-observability-plan.md` (Tier 2 plan).

## Accepts
TODO — agent invocation context (`agent_id`, `project_id`, `provider`, etc.).

## Emits
TODO — structured log records written to `<runs_root>/<project_id>/logs/` and forwarded to the dashboard emitter where relevant.

## Source
- [`backend/agents/telemetry.py`](../../backend/agents/telemetry.py)
- Related: [`backend/observability/`](../../backend/observability/), [`docs/design/unified-logging-launcher.md`](../design/unified-logging-launcher.md), [`docs/design/tier2-observability-plan.md`](../design/tier2-observability-plan.md).
