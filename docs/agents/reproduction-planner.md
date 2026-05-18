# `reproduction-planner`

> **Stage:** builder (4 of 6).
> **Registry entry:** [`AGENT_REGISTRY["reproduction-planner"]`](../../backend/agents/registry.py).

> **STUB**

## Purpose
Define what counts as reproduction: smoke-test plan, full-run plan, expected outputs, verification checklist.

## Accepts
TODO — `PaperClaimMap` + `EnvironmentSpec`.

## Emits
TODO — [`ReproductionContract`](../../backend/agents/schemas.py).

## Source
- Models: `ReproductionContract` in [`backend/agents/schemas.py`](../../backend/agents/schemas.py).
- Implementation: check registry — may live in `orchestrator.py` or a dedicated module.
