# Error & Resilience Envelopes

> **STUB** — central catalog of failure-mode shapes. Companion to the existing [resilience.md](resilience.md) (policy) — this file documents the *wire shapes*.

## Failed-agent envelope

When an agent raises and the resilience wrapper catches it, the orchestrator records an `AgentOutput` with `status="failed"`:

```json
{
  "agent_id": "experiment-runner",
  "status": "failed",
  "structured_outputs": {},
  "summary": "Docker build failed: missing apt package 'libosmesa6-dev'",
  "exploration_log": {
    "error_type": "DockerBuildError",
    "error_message": "...",
    "traceback": "...",
    "retry_attempts": 2
  }
}
```

TODO — confirm exact field names against [`backend/agents/resilience/`](../../backend/agents/resilience/) and [`backend/agents/orchestrator.py`](../../backend/agents/orchestrator.py).

## Per-stage failure markers

`PipelineState` records per-stage failure on `AgentExecutionTrace`. TODO — document the trace shape.

## Verifier failure surface

A verifier never "fails" in the exception sense; it returns a `VerifierScore` with a low score and populated `mismatches`. Aggregation by [`supervisor-verifier`](supervisor-verifier.md) may yield a `GateStatus` of `failed_reproduction` or `blocked_requires_human`.

## Retry semantics

See [resilience.md](resilience.md).

## Common failure modes (per agent)

TODO — populate from per-agent docs as they get filled in. Currently documented:

- [paper-understanding](paper-understanding.md#failure-modes)
