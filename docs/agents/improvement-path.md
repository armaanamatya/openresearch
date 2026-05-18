# `improvement-path`

> **Stage:** improvement (one parallel path per hypothesis).
> **Registry entry:** [`AGENT_REGISTRY["improvement-path"]`](../../backend/agents/registry.py).
> **Implementation:** [`backend/agents/improvement.py`](../../backend/agents/improvement.py).

> **STUB**

## Purpose
Implement one `ImprovementHypothesis`, run it, and report its measured `metrics` vs the round's baseline.

## Accepts
TODO — one [`ImprovementHypothesis`](../../backend/agents/schemas.py), baseline code path, environment.

## Emits
TODO — [`PathResult`](../../backend/agents/schemas.py).

## Source
- [`backend/agents/improvement.py`](../../backend/agents/improvement.py)
- Model: `PathResult` in [`backend/agents/schemas.py`](../../backend/agents/schemas.py).
