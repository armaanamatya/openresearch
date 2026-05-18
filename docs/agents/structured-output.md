# `structured-output`

> **Module:** [`backend/agents/structured_output.py`](../../backend/agents/structured_output.py).

> **STUB** — infra, not a registered agent.

## Purpose
Parsing/validation helpers that turn raw LLM text into a validated Pydantic instance. Handles JSON extraction from mixed text, repair on minor errors, `model_config = {"extra": "ignore"}` semantics.

## Accepts
TODO — raw text + target Pydantic class.

## Emits
TODO — validated Pydantic instance, or a structured failure result captured into the agent's `exploration_log`.

## Source
- [`backend/agents/structured_output.py`](../../backend/agents/structured_output.py)
- Pattern used by every agent in [`backend/agents/`](../../backend/agents/) when receiving LLM output.
