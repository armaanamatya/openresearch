# `dashboard-emitter`

> **Module:** [`backend/agents/dashboard_emitter.py`](../../backend/agents/dashboard_emitter.py).

> **STUB** — infra, not a registered agent.

## Purpose
Single funnel for dashboard events. Every agent stage transition, progress update, and completion goes through the emitter, which serializes to the event store and (via SSE) to the frontend.

## Accepts
TODO — typed event payloads. See [events.md](events.md) for the catalog of event types and shapes.

## Emits
TODO — JSON events on the SSE channel; persisted to the event store (see `backend/eventstore/`).

## Source
- [`backend/agents/dashboard_emitter.py`](../../backend/agents/dashboard_emitter.py)
- Event store: [`backend/eventstore/`](../../backend/eventstore/)
- Frontend bridge: [`docs/lab-ui-pipeline-bridge.md`](../lab-ui-pipeline-bridge.md)
