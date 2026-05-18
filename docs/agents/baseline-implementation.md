# `baseline-implementation`

> **Stage:** builder (5 of 6).
> **Registry entry:** [`AGENT_REGISTRY["baseline-implementation"]`](../../backend/agents/registry.py).
> **Implementation:** [`backend/agents/baseline_implementation.py`](../../backend/agents/baseline_implementation.py).

> **STUB**

## Purpose
Adapt the official code (if discovered) or implement from the paper. Produces a runnable training/eval entry point.

## Accepts
TODO — `PaperClaimMap`, `EnvironmentSpec`, `ReproductionContract`, discovered artifacts.

## Emits
TODO — [`BaselineResult`](../../backend/agents/schemas.py) (`mode` ∈ `{"adapt", "implement_from_paper"}`, `code_path`, `dockerfile_path`, `commands_to_run`).

## Source
- [`backend/agents/baseline_implementation.py`](../../backend/agents/baseline_implementation.py)
- Model: `BaselineResult` in [`backend/agents/schemas.py`](../../backend/agents/schemas.py).
