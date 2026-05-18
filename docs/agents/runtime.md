# `runtime`

> **Module:** [`backend/agents/runtime/`](../../backend/agents/runtime/).

> **STUB** — infra, not a registered agent.

## Purpose
Provider-neutral abstraction over LLM agent SDKs (Anthropic, OpenAI). Decouples `AgentSpec` (registry) from a concrete model client.

## Accepts
TODO — typed shapes:
- `AgentRuntime` (abstract base): the runtime contract.
- `AgentRuntimeSpec`: the provider-neutral spec produced by `AgentSpec.to_runtime_spec`.
- `ToolSpec`: name + description for each tool a runtime exposes.
- `ProviderName`: literal `"anthropic" | "openai"`.

## Emits
TODO — agent output text (streamed/collected via `collect_agent_text`).

## Source
- [`backend/agents/runtime/base.py`](../../backend/agents/runtime/base.py)
- [`backend/agents/runtime/invoke.py`](../../backend/agents/runtime/invoke.py) (`collect_agent_text`)
- Concrete impls under `backend/agents/runtime/` (Anthropic, OpenAI providers).
