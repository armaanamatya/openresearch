# Multi-Provider Agent Runtime — Plan (Anthropic Claude Agent SDK + OpenAI Agents SDK)

> Goal: Drive the ReproLab agent pipeline through **either** the Claude Agent SDK (`claude-agent-sdk`) or the OpenAI Agents SDK (`openai-agents`) via one Protocol. Both providers are first-class. Hackathon-provided OpenAI keys can drive the pipeline without losing Anthropic functionality. Switching is a single env var. **No quality compromise:** each provider uses its native best-practice path.

---

## Pre-flight: pydantic ModuleNotFoundError fix (DONE)

**Cause:** `frontend/src/lib/demo/node-runner.ts:291` shelled out to system `python3`, not the venv. System Python has no `pydantic`.

**Fix landed:** A `pythonBinary()` helper in `node-runner.ts` resolves in this order:
1. `REPROLAB_PYTHON_BIN` env var (override).
2. `<repo>/.venv/bin/python` (Linux/macOS) or `<repo>/.venv/Scripts/python.exe` (Windows).
3. System `python3` / `py` (last resort).

The dashboard run will now use the venv and find pydantic + claude-agent-sdk + deepeval. Re-test: open the lab dashboard, click **Start a run**, watch `runs/<project_id>/runner.stderr.log` — no more `ModuleNotFoundError`.

---

## Decision: Two SDKs, One Abstraction

The user decision is **dual-SDK first-class support**. We use both vendor-supplied agent SDKs natively and abstract them at the **agent-runtime layer**.

### Why not the alternatives

| Alternative | Why rejected |
|---|---|
| **Claude Agent SDK only, redirect to OpenAI** | Impossible — the SDK is hardwired to the Anthropic Messages API; no provider plug-in. |
| **Raw `openai` SDK (Responses API) on the OpenAI side, claude-agent-sdk on Anthropic** | We'd build the agent loop ourselves on OpenAI side; lose `openai-agents`' native handoffs, guardrails, and tracing — features we want symmetrically. |
| **`litellm` or `any-llm` proxy** | Translation layer adds bugs; loses provider-specific features (Anthropic prompt caching, OpenAI Structured Outputs `strict` mode); harder to debug. |
| **OpenAI Agents SDK for both** | Same impossibility on the Anthropic side. |

The right level is "an agent that can run with tools and dispatch sub-agents" — both SDKs natively model this. We abstract there.

---

## Current State (5 SDK call sites, one canonical pattern)

Claude Agent SDK is imported at exactly 5 places in `backend/agents/`:

| File | Line | Imports |
|---|---|---|
| `orchestrator.py` | 25-32 | `AgentDefinition, AssistantMessage, ClaudeAgentOptions, ResultMessage, ToolUseBlock, query` |
| `registry.py` | 159 | `AgentDefinition` |
| `paper_understanding.py` | 89 | `AssistantMessage, ClaudeAgentOptions, ResultMessage, query` |
| `environment_detective.py` | 110 | `AssistantMessage, ClaudeAgentOptions, ResultMessage, query` |
| `baseline_implementation.py` | 418 | `AssistantMessage, ClaudeAgentOptions, ResultMessage, query` |
| `experiment_runner.py` | 305 | `AssistantMessage, ClaudeAgentOptions, ResultMessage, query` |

All five sites use the same call shape:

```python
options = ClaudeAgentOptions(model=..., max_turns=..., agents=registry, permission_mode="bypassPermissions")
async for message in query(prompt=task_prompt, options=options):
    if isinstance(message, AssistantMessage):
        for block in message.content:
            if hasattr(block, "text"):
                collected_text.append(block.text)
            elif isinstance(block, ToolUseBlock):
                tool_calls.append(block)
    elif isinstance(message, ResultMessage):
        usage = message.usage
```

This narrow surface is what makes the dual-SDK adapter feasible — one Protocol, two implementations, five replacements.

---

## Design: `AgentRuntime` Protocol with Per-SDK Adapters

```
backend/agents/runtime/
├── __init__.py              # public API: make_runtime, AgentRuntime, AgentSpec, ToolSpec, StreamEvent
├── base.py                  # Protocol + ADTs
├── claude_runtime.py        # wraps claude_agent_sdk
├── openai_runtime.py        # wraps openai_agents
└── factory.py               # make_runtime(provider) → AgentRuntime
```

```
                       ┌────────────────────────────┐
                       │  ReproLabOrchestrator       │   ← provider-agnostic
                       │  (our code; minimal change) │
                       └─────────────┬───────────────┘
                                     │ uses
                       ┌─────────────▼──────────────┐
                       │  AgentRuntime Protocol      │   ← OUR abstraction
                       │  + AgentSpec / StreamEvent  │
                       └──────┬───────────┬──────────┘
                              │           │
              ┌───────────────┘           └────────────────┐
              ▼                                            ▼
   ┌─────────────────────┐                     ┌──────────────────────┐
   │ ClaudeAgentRuntime  │                     │ OpenAiAgentRuntime   │
   │ wraps               │                     │ wraps                │
   │   claude-agent-sdk  │                     │   openai-agents      │
   │   (query, agents=,  │                     │   (Agent, Runner,    │
   │    AgentDefinition) │                     │    handoffs=,        │
   │                     │                     │    function_tool)    │
   └─────────────────────┘                     └──────────────────────┘
              │                                            │
              ▼                                            ▼
   Anthropic Messages API                       OpenAI Responses API
                                                (driven by openai-agents)
```

### The Protocol (frozen contract)

```python
# backend/agents/runtime/base.py
from typing import Any, AsyncIterator, Protocol
from dataclasses import dataclass, field

@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]                    # JSON Schema; provider-agnostic

@dataclass(frozen=True)
class AgentSpec:
    name: str                                       # e.g. "paper-understanding"
    instructions: str                               # system prompt / persona
    model: str                                      # provider-specific id
    tools: tuple[ToolSpec, ...] = ()
    sub_agents: tuple["AgentSpec", ...] = ()        # mapped to agents= (Claude) / handoffs= (OpenAI)
    max_turns: int = 15
    thinking_budget_tokens: int | None = None
    cache_static_blocks: bool = True
    permission_mode: str = "bypassPermissions"      # claude-only; ignored elsewhere

# StreamEvent ADT — the orchestrator only ever pattern-matches on these.
@dataclass(frozen=True)
class StreamText:
    text: str

@dataclass(frozen=True)
class StreamToolCall:
    tool_id: str
    tool_name: str
    tool_input: dict[str, Any]

@dataclass(frozen=True)
class StreamUsage:
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    reasoning_tokens: int = 0                       # openai o-series; 0 elsewhere

StreamEvent = StreamText | StreamToolCall | StreamUsage

class AgentRuntime(Protocol):
    """Provider-agnostic agent runtime.

    Each implementation wraps its provider's official agent SDK
    (claude-agent-sdk or openai-agents) and yields StreamEvents.
    """

    @property
    def provider_name(self) -> str: ...

    async def run_agent(
        self,
        *,
        agent: AgentSpec,
        user_input: str,
    ) -> AsyncIterator[StreamEvent]: ...
```

**Why this shape:**
- The orchestrator only sees `StreamText`, `StreamToolCall`, `StreamUsage`. Never imports SDK types.
- `ToolSpec.input_schema` is **JSON Schema** — both SDKs natively understand it.
- `AgentSpec.sub_agents` maps to Claude's `agents=` and OpenAI's `handoffs=` natively. No translation tax.
- Provider-specific options (`permission_mode`, `thinking_budget_tokens`, `cache_static_blocks`) are honored when supported, logged-as-mapped when not, never silently ignored.

### `ClaudeAgentRuntime` — wraps `claude-agent-sdk`

```python
# backend/agents/runtime/claude_runtime.py
from claude_agent_sdk import (
    AgentDefinition,
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    ToolUseBlock,
    query,
)

class ClaudeAgentRuntime:
    provider_name = "anthropic"

    async def run_agent(self, *, agent, user_input):
        sub_agent_defs = {
            sa.name: AgentDefinition(
                description=sa.instructions[:200],
                prompt=sa.instructions,
                model=sa.model,
                tools=[t.name for t in sa.tools],
            )
            for sa in agent.sub_agents
        }

        options = ClaudeAgentOptions(
            model=agent.model,
            max_turns=agent.max_turns,
            agents=sub_agent_defs,
            permission_mode=agent.permission_mode,
            system=[
                {
                    "type": "text",
                    "text": agent.instructions,
                    "cache_control": {"type": "ephemeral"} if agent.cache_static_blocks else None,
                }
            ],
            thinking={"type": "enabled", "budget_tokens": agent.thinking_budget_tokens}
                     if agent.thinking_budget_tokens else None,
        )

        async for message in query(prompt=user_input, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if hasattr(block, "text"):
                        yield StreamText(block.text)
                    elif isinstance(block, ToolUseBlock):
                        yield StreamToolCall(
                            tool_id=block.id,
                            tool_name=block.name,
                            tool_input=block.input,
                        )
            elif isinstance(message, ResultMessage):
                u = message.usage or {}
                yield StreamUsage(
                    input_tokens=u.get("input_tokens", 0),
                    output_tokens=u.get("output_tokens", 0),
                    cache_read_input_tokens=u.get("cache_read_input_tokens", 0),
                    cache_creation_input_tokens=u.get("cache_creation_input_tokens", 0),
                )
```

This is a **mechanical adapter** — nothing observable changes for Anthropic users.

### `OpenAiAgentRuntime` — wraps `openai-agents`

```python
# backend/agents/runtime/openai_runtime.py
from openai_agents import Agent, Runner, function_tool

class OpenAiAgentRuntime:
    provider_name = "openai"

    async def run_agent(self, *, agent, user_input):
        # Build OpenAI handoff Agents from sub_agents recursively.
        oa_handoffs = [self._build_oa_agent(sa) for sa in agent.sub_agents]
        oa_tools = [self._build_oa_tool(t) for t in agent.tools]

        oa_agent = Agent(
            name=agent.name,
            instructions=agent.instructions,
            model=agent.model,
            tools=oa_tools,
            handoffs=oa_handoffs,
            model_settings=self._model_settings(agent),
        )

        result_stream = Runner.run_streamed(
            oa_agent,
            input=user_input,
            max_turns=agent.max_turns,
        )
        async for event in result_stream.stream_events():
            translated = self._translate_event(event)
            if translated is not None:
                yield translated

    def _build_oa_agent(self, spec) -> Agent:
        return Agent(
            name=spec.name,
            instructions=spec.instructions,
            model=spec.model,
            tools=[self._build_oa_tool(t) for t in spec.tools],
            handoffs=[self._build_oa_agent(sa) for sa in spec.sub_agents],
            model_settings=self._model_settings(spec),
        )

    def _build_oa_tool(self, t):
        # openai-agents' function_tool builds from JSON Schema directly.
        return function_tool(
            name=t.name,
            description=t.description,
            params_json_schema=t.input_schema,
            strict_mode=True,                       # OpenAI Structured Outputs guarantee
        )

    def _model_settings(self, spec):
        settings: dict[str, Any] = {}
        if spec.thinking_budget_tokens:
            settings["reasoning_effort"] = self._map_thinking(spec.thinking_budget_tokens)
        return settings or None

    def _map_thinking(self, budget_tokens: int) -> str:
        if budget_tokens >= 16_000: return "high"
        if budget_tokens >= 4_000:  return "medium"
        return "low"

    def _translate_event(self, event):
        # openai-agents emits typed events; translate to our StreamEvent ADT.
        if event.type == "run_item_stream_event":
            item = event.item
            if item.type == "message_output_item":
                return StreamText(item.text)
            if item.type == "tool_call_item":
                return StreamToolCall(
                    tool_id=item.tool_call_id,
                    tool_name=item.tool_name,
                    tool_input=item.arguments_json,
                )
        if event.type == "raw_response_event":
            r = event.data
            if r.type == "response.completed" and r.response.usage:
                u = r.response.usage
                return StreamUsage(
                    input_tokens=u.input_tokens,
                    output_tokens=u.output_tokens,
                    cache_read_input_tokens=getattr(u.input_tokens_details, "cached_tokens", 0) or 0,
                    reasoning_tokens=getattr(u.output_tokens_details, "reasoning_tokens", 0) or 0,
                )
        return None
```

### Factory

```python
# backend/agents/runtime/factory.py
import os
from typing import Literal

def make_runtime(provider: Literal["anthropic", "openai"] | None = None) -> AgentRuntime:
    chosen = (provider or os.environ.get("REPROLAB_LLM_PROVIDER") or "anthropic").lower()
    if chosen == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ProviderConfigurationError(provider="anthropic", reason="api_key_missing")
        return ClaudeAgentRuntime()
    if chosen == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise ProviderConfigurationError(provider="openai", reason="api_key_missing")
        return OpenAiAgentRuntime()
    raise ValueError(
        f"Unknown LLM provider {chosen!r}. Set REPROLAB_LLM_PROVIDER=anthropic|openai."
    )
```

---

## Migration: Five Call Sites → One Path

### Step 1 (P-Multi-1): Land the abstraction without using it (1 day)

- Create `backend/agents/runtime/{base,claude_runtime,openai_runtime,factory,__init__}.py`.
- `pyproject.toml` + `backend/requirements.txt` — add `openai-agents>=0.1,<0.2`. Keep `claude-agent-sdk>=0.1.80`.
- New tests:
  - `tests/test_runtime_claude_adapter.py` — fake `claude_agent_sdk.query` async-generator returning canned messages → assert StreamEvent translation correct.
  - `tests/test_runtime_openai_adapter.py` — fake `Runner.run_streamed` → same.
- **Acceptance:** `pytest tests/test_runtime_*.py -v` passes; **no production behaviour change**.

### Step 2 (P-Multi-2): Migrate the orchestrator (½ day)

- `backend/agents/orchestrator.py:_invoke_agent` — replace direct `claude_agent_sdk.query()` with `self._runtime.run_agent(...)` where `self._runtime: AgentRuntime` is constructor-injected.
- `ReproLabOrchestrator.__init__` — accept `runtime: AgentRuntime | None = None`; default to `make_runtime()`.
- The `AssistantMessage / ResultMessage / ToolUseBlock` pattern-match becomes `StreamText / StreamToolCall / StreamUsage`.
- Telemetry (cost via `usage` event) keeps working — usage now arrives via `StreamUsage`.
- **Frozen contract:** orchestrator's existing public surface (`PipelineState`, gate decisions, checkpoint behavior) stays byte-identical for Anthropic users.

### Step 3 (P-Multi-3): Migrate the four `run_with_sdk` per-agent functions (½ day)

`paper_understanding.run_with_sdk`, `environment_detective.run_with_sdk`, `baseline_implementation.run_with_sdk`, `experiment_runner.run_with_sdk` each currently does its own local `claude_agent_sdk` import. Replace each with `make_runtime()` + the same Protocol pattern.

### Step 4 (P-Multi-4): Sub-agent semantics (½ day, Opus design)

Both SDKs natively support sub-agents but with different context-passing semantics:

| SDK | Mechanism | Context passing |
|---|---|---|
| Claude Agent SDK | `agents={name: AgentDefinition(...)}` | Sub-agent shares parent's conversation context |
| OpenAI Agents SDK | `handoffs=[agent_object]` | Full context transferred to handoff target |

Document this and normalize: both adapters serialize the parent agent's accumulated context into the sub-agent's `user_input`. Tests assert decision-log identity across providers. The `sub_agent_router.py` mentioned in the production roadmap becomes a thin shim — both SDKs handle the heavy lifting natively.

### Step 5 (P-Multi-5): Configuration & ergonomics (¼ day)

- `backend/config.py`:
  ```python
  llm_provider: Literal["anthropic", "openai"] = "anthropic"
  anthropic_default_model: str = "claude-opus-4-7"
  openai_default_model: str = "gpt-4o"
  openai_reasoning_model: str = "o4-mini"          # for verifier / improvement agents
  agent_provider_overrides: dict[str, str] = {}   # per-agent override map
  ```
- `backend/cli.py:reproduce` — add `--provider {anthropic,openai}` flag (default = env var).
- `frontend/src/lib/demo/node-runner.ts` — provider chip in the dashboard "Mode" panel; passes through as `REPROLAB_LLM_PROVIDER` env var to the spawned Python.
- `.env.example` (new, checked-in template):
  ```
  ANTHROPIC_API_KEY=sk-ant-...
  OPENAI_API_KEY=sk-...
  REPROLAB_LLM_PROVIDER=anthropic    # or "openai"
  ```

### Step 6 (P-Multi-6): Tests + parity matrix (1 day)

- Existing `tests/test_issue22_orchestrator.py` and friends — add `pytest.mark.parametrize("provider", ["anthropic", "openai"])` so every existing test runs against both adapters with **fake SDKs**. No live API calls in unit tests.
- New `tests/test_runtime_parity.py` — given identical canned model output via both fakes, both adapters produce byte-identical `StreamEvent` sequences.
- New `tests/test_runtime_provider_smoke.py` — gated on `RUN_PROVIDER_SMOKE=1` and *both* `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` set; runs a tiny prompt through each adapter; asserts non-empty `StreamUsage`. Skipped in CI by default.

### Step 7 (P-Multi-7): Replay determinism cross-provider (¼ day)

The `RecordingClaudeClient` from P1-T5 in the production roadmap becomes `RecordingAgentRuntime` (provider-agnostic). The `request_hash` includes the provider name so a recording made on Claude is not silently replayed against OpenAI. But the recording **format** is the same `StreamEvent` JSONL on both sides — useful for "did we behaviorally regress when we switched providers" diff testing.

---

## Feature Parity Matrix

| Feature | Claude Agent SDK | OpenAI Agents SDK | Strategy |
|---|---|---|---|
| Streaming | ✅ `query()` async iter | ✅ `Runner.run_streamed(...)` | Both yield `StreamEvent`s |
| Tool use | ✅ `ToolUseBlock` | ✅ `tool_call_item` event | Common JSON Schema → both |
| Structured outputs (`strict: true`) | ✅ via tool use | ✅ `function_tool(strict_mode=True)` | Same Pydantic-derived JSON Schema both ways |
| Sub-agent dispatch | ✅ `agents={...}` | ✅ `handoffs=[...]` | Both first-class |
| Prompt caching | ✅ `cache_control: ephemeral` (5min/1h) | ✅ Automatic, no fine-grained control | Adapter applies if supported; no silent skip |
| Reasoning / thinking | ✅ `thinking.budget_tokens` | ✅ `model_settings.reasoning_effort` (o-series) | Map budget → effort via `_map_thinking` |
| Citations | ✅ Anthropic Citations API | ❌ DIY | Claude-only; OpenAI emits `ProviderFeatureUnsupported` |
| Files API | ✅ Anthropic Files | ✅ OpenAI Files | Out of scope for v1 |
| Tracing | ✅ Native | ✅ Native (built-in) | Both wired into `AgentTelemetryRecorder` |
| Guardrails | ⚠️ Manual | ✅ `Guardrails` first-class | OpenAI-only enhancement; opt-in per agent |

Where parity is imperfect, the adapter logs a `ProviderFeatureUnsupported` event with `feature_name` + `provider_name` — orchestrator behavior unchanged but audit trail records the gap. **/ml-code-quality** "failure modes explicit" rule applied to capability mismatch.

---

## Eight Quality / Reliability Guardrails (mandatory)

A multi-provider system fails badly if the abstraction is leaky. These invariants are non-negotiable:

### 1. Schemas are the contract, not provider features

Every agent's output schema is a Pydantic model in `backend/agents/schemas.py`. Tool calls — both SDKs — consume `model.model_json_schema()`. **Neither provider can return data that fails Pydantic validation.** OpenAI's `strict_mode=True` and Anthropic's tool-use schema both enforce required fields.

### 2. Telemetry is provider-agnostic

`AgentInvocationRecord.usage: dict` (already in `backend/agents/telemetry.py`) carries normalized keys:

```python
{
  "provider": "anthropic" | "openai",
  "model": "<model_name>",
  "input_tokens": int,
  "output_tokens": int,
  "cache_read_input_tokens": int,    # 0 if provider doesn't support
  "reasoning_tokens": int,           # 0 unless openai o-series
  "cost_usd": float,                 # computed from per-provider price table
}
```

Cost analysis dashboards work uniformly across providers.

### 3. Determinism via replay, not seeds

Neither SDK exposes deterministic seeds for production. `RecordingAgentRuntime` (Step 7) records `StreamEvent` JSONL — identical format across providers; replay-misses fail loudly with typed `ReplayMiss(missing_hash, provider)`.

### 4. Per-agent provider override

Heavy code-generation agents (`baseline-implementation`, `improvement-path`) often work better on Claude Opus; lighter analysis agents may prefer OpenAI o4-mini for cost. Per-agent overrides:

```python
agent_provider_overrides: dict[str, str] = {
    "baseline-implementation": "anthropic",
    "improvement-path": "anthropic",
    "method-fidelity-verifier": "openai",   # cheaper, comparable fidelity
}
```

The orchestrator consults the override map before calling `make_runtime()`. Default: global provider.

### 5. Fail-closed on missing key

If `REPROLAB_LLM_PROVIDER=openai` and `OPENAI_API_KEY` is unset, `make_runtime()` raises `ProviderConfigurationError(provider="openai", reason="api_key_missing")` at orchestrator init — never mid-pipeline. The dashboard surfaces this immediately.

### 6. No silent feature degradation

If a user sets `thinking_budget_tokens=8000` and the runtime is OpenAI, the budget maps to `reasoning_effort="medium"` and a `ProviderFeatureMapped` event is emitted. The user is **never silently downgraded** — the decision log records the mapping.

### 7. Per-agent model selection per provider

`AGENT_REGISTRY` carries `default_model_anthropic` and `default_model_openai`:

```python
"baseline-implementation": AgentDefinitionSpec(
    default_model_anthropic="claude-opus-4-7",
    default_model_openai="gpt-4o",
    ...
),
"method-fidelity-verifier": AgentDefinitionSpec(
    default_model_anthropic="claude-sonnet-4-6",
    default_model_openai="o4-mini",
    ...
),
```

Cost / latency tradeoffs tuned per role per provider, not flattened.

### 8. Mandatory test parametrization

Every test exercising an agent runs `pytest.mark.parametrize("provider", ["anthropic", "openai"])`. CI fails if a test skips one provider without a documented `xfail` reason. This catches "behavior diverged on OpenAI" regressions at PR time.

---

## What This Does Not Change

- Event store, citation invariant, eventstore subscription cursor, intake/parser/discovery/indexer/workspace pipeline, sandbox runtime — none touch the LLM. **Zero risk to those layers.**
- Pydantic schemas (`backend/agents/schemas.py`), prompts (`backend/agents/prompts/`), gate-decision binding rule (PRD §788), assumption ledger, research map — provider-agnostic.
- Dashboard, eval system, CLI commands at the user-visible level. `reproduce` gains a `--provider` flag.

---

## Effort & Sequencing

| Step | Effort | Dependencies |
|---|---|---|
| P-Multi-1 (abstraction) | 1 day | — |
| P-Multi-2 (orchestrator migration) | ½ day | P-Multi-1 |
| P-Multi-3 (per-agent migration) | ½ day | P-Multi-2 |
| P-Multi-4 (sub-agent normalization) | ½ day | P-Multi-2 |
| P-Multi-5 (config + CLI + frontend) | ¼ day | P-Multi-2 |
| P-Multi-6 (parametrized tests) | 1 day | P-Multi-3 |
| P-Multi-7 (cross-provider replay) | ¼ day | P-Multi-1 |

**Total: ~4 days.** Hackathon-fast minimum: P-Multi-1 + P-Multi-2 + P-Multi-3 + P-Multi-5 (~2 days). Provider toggle works end-to-end, sub-agent normalization + parametrized tests defer to post-demo.

---

## Verification (after all 7 steps)

```bash
# 1. Both adapter unit tests pass
pytest tests/test_runtime_claude_adapter.py tests/test_runtime_openai_adapter.py -v

# 2. Parametrized tests cover both
pytest -q -k "agent or orchestrator" --collect-only | grep -E "anthropic|openai" | wc -l
# Each agent/orchestrator test listed twice (once per provider)

# 3. Live smoke (real keys)
ANTHROPIC_API_KEY=... OPENAI_API_KEY=... RUN_PROVIDER_SMOKE=1 \
  pytest tests/test_runtime_provider_smoke.py -v

# 4. End-to-end: same paper, both providers, compare research maps
REPROLAB_LLM_PROVIDER=anthropic python -m backend.cli reproduce demo/papers/ppo.pdf --runs-root /tmp/anthropic
REPROLAB_LLM_PROVIDER=openai    python -m backend.cli reproduce demo/papers/ppo.pdf --runs-root /tmp/openai
diff <(jq -S .baseline_summary /tmp/anthropic/<id>/research_map.json) \
     <(jq -S .baseline_summary /tmp/openai/<id>/research_map.json)
# Structurally similar; gate decisions reach the same status, same number of paths

# 5. Dashboard
# Open http://localhost:3000 → "Start a run" → toggle provider in Mode panel
```

---

## Risk Register

| Risk | Mitigation |
|---|---|
| `openai-agents` SDK API drift | Pin `openai-agents>=0.1,<0.2`; adapter is the only consumer; bump deliberately |
| Sub-agent context semantics diverge | Both adapters serialize parent context into sub-agent input; tests assert decision-log identity |
| Schema validation passes Claude / fails OpenAI | Pydantic `model_json_schema()` is the only source; `strict_mode=True` enforced |
| Cost runaway on o-series reasoning tokens | `BudgetEnforcer` (P1-T4) tracks `reasoning_tokens` separately; cap independently |
| Provider-specific features (Citations API) | Adapter emits `ProviderFeatureUnsupported`; dashboard surfaces "Citations not used" advisory |
| Claude Agent SDK migrates non-async | Wrapper means we change `ClaudeAgentRuntime` only; orchestrator unchanged |
| Hackathon deadline | Ship P-Multi-1 + P-Multi-2 + P-Multi-3 + P-Multi-5 (~2 days minimum end-to-end) |

---

## Bottom Line

The current code has a **narrow, regular SDK surface** (5 sites, one pattern). That's the ideal precondition for an adapter pattern at the **agent-runtime layer**. The plan is:

1. Define `AgentRuntime` Protocol + `AgentSpec` + `StreamEvent` ADT.
2. Two adapters — `ClaudeAgentRuntime` (wraps `claude-agent-sdk`) and `OpenAiAgentRuntime` (wraps `openai-agents`).
3. Migrate 5 call sites + factory.
4. Parametrize every existing test with `provider ∈ {anthropic, openai}`.
5. Frontend toggle + CLI flag.

**Total effort: ~4 days. Quality preservation: complete** — schemas are the contract, both SDKs honor `strict: true`, telemetry normalized, decisions binding regardless of provider, replay-format cross-provider compatible.

Hackathon-ready in 2 days if we ship the minimum (steps 1+2+3+5; sub-agent normalization + parametrized tests post-demo polish).

— Claude Opus 4.7 (1M context), 2026-05-09
