# `registry`

> **Module:** [`backend/agents/registry.py`](../../backend/agents/registry.py).

> **STUB** — infra, not a registered agent.

## Purpose
The single source of truth for which agents exist. Defines `AgentSpec` (frozen dataclass) and the `AGENT_REGISTRY` dict keyed by `agent_id`. Converts registry entries into provider-neutral `AgentRuntimeSpec` instances.

## Accepts
TODO — `AgentSpec` constructor: `agent_id`, `role`, `description`, `prompt`, `tools`, `spawn_permissions`, `max_turns`, `default_model_anthropic`, `default_model_openai`, `thinking_budget_tokens`.

`AgentSpec.to_runtime_spec(provider, *, model_override=None, max_turns=None, working_directory=None, sub_agents=())` — converts to the runtime contract.

## Emits
TODO — `AgentRuntimeSpec` (see [runtime.md](runtime.md)).

## The 14 registered agent_ids
Builder: `paper-understanding`, `artifact-discovery`, `environment-detective`, `reproduction-planner`, `baseline-implementation`, `experiment-runner`.
Verifier: `method-fidelity-verifier`, `environment-verifier`, `data-metrics-verifier`, `artifact-diff-verifier`, `rubric-verifier`, `supervisor-verifier`.
Improvement: `improvement-orchestrator`, `improvement-path`.

## Source
- [`backend/agents/registry.py`](../../backend/agents/registry.py)
- Helpers: `get_agent_definitions()`, `_default_model_for_provider`, `_model_override_from_settings`, `_TOOL_DESCRIPTIONS`.
