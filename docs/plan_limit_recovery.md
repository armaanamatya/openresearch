# Plan — Elegant Recovery When an Agent Hits a Budget Cap

## What actually happened

Run `prj_9dbaf95589738086` died at gate_1 with:
```
AgentLimitExceeded: Agent 'supervisor-verifier' hit tool_calls cap of 80
after 208.8s (0 chars of partial output preserved)
```

The trace shows the failure mode in detail:

- `supervisor-verifier` made **80 tool calls in 209 s**
- Ran `Read parsed_full_text.txt` **30+ times** — same file, no caching
- Ran the same `find /home/abheekp/openresearch/runs/...` **3 times** in a row
- 0 chars in `partial_output` — the agent never wrote a structured-output
  block before the cap fired, so even our typed exception had nothing to
  give the recovery layer

Three problems compound:

1. **No recovery wired.** The previous round added `AgentLimitExceeded`
   with `partial_output`, but only as a typed surface. When it fires,
   the orchestrator just `raise`s and the pipeline dies.
2. **Cap is uniform across roles.** `max_tool_calls_per_agent = 80`
   applies equally to a builder agent (writes ≤ 5 files) and to the
   supervisor-verifier (must read 10+ artifacts and reason across
   them). Verifiers are the wrong agents to cap at the same number as
   builders.
3. **No prevention.** The agent has no visibility into its own budget
   while running, so it can't self-throttle.

`/token` says: don't burn Opus on writing it from scratch — write the
plan to disk, hand it to a sub-agent. `/ml-code-quality` says: roles +
recovery strategies are stable interfaces that should be config + Protocol,
not `if agent_id == ...` branches. `/ml-experiment-rigor` says: every
recovery attempt must be observable (telemetry + provenance), and a
recovered output must be flagged as such — never silently substituted
for a successful one.

## Design

Four layers, each independently disable-able. They compose, but each
one is useful on its own:

### Layer 1 — Right-size the budget per role

Replace the single `max_tool_calls_per_agent: int | None` with a
**per-role override map** evaluated at runtime spec build time.

Budgets for `efficient` mode (chosen empirically from this run + the
verifier-tool-call distribution we already log in
`agent_telemetry.jsonl`):

| Role        | turns (efficient) | tool_calls (efficient) | wall_clock |
|-------------|-------------------|------------------------|------------|
| builder     | 30                | 80                     | 1200 s     |
| heavy-builder (experiment-runner, baseline-implementation) | 60 | 120 | 1800 s |
| verifier    | 60                | **160**                | 1500 s     |
| supervisor  | 80                | **220**                | 1800 s     |

`max` mode keeps everything `None` as today.

Source of truth: `backend/agents/execution.py::ExecutionProfile.role_overrides`,
loaded from a small dict alongside the existing `max_turns_per_agent`
fields. The orchestrator's `_build_runtime_spec` reads
`AGENT_REGISTRY[agent_id].role` and picks the override.

This alone would have saved this run (supervisor-verifier @ 80 → 220).

### Layer 2 — Soft-stop with structured-output recovery

When `tool_call_count` reaches the cap (or the SDK raises its turn
error, or `asyncio.timeout` fires), the orchestrator does NOT raise
immediately. Instead:

```
1. Capture everything streamed so far (collected_text, tool_calls,
   trace_lines).
2. Make ONE final, tool-disabled, low-budget call to the same runtime:
       prompt = original_prompt
              + "\n\n<budget_exceeded reason='{kind}' value={N}/>\n"
              + "You have hit the {kind} budget. Tools are now disabled.\n"
              + "Reply NOW with the structured-output JSON expected by\n"
              + "this agent's contract, using only what you have already\n"
              + "discovered. Do NOT call any tools."
   With max_turns=2, no tools, same model.
3. If the recovery call returns a JSON that parses against the agent's
   _OUTPUT_MODELS schema → return it as the agent's output, but mark
   the AgentInvocationRecord with recovery_kind="soft_stop_summary"
   and success=true_with_recovery.
4. If the recovery call ALSO fails → raise AgentLimitExceeded as today.
   No silent fallback to a fake "ok".
```

This is the Codex equivalent of a deadline-driven LLM finalize: turn
the cap into a "submit now" signal instead of a hard kill.

### Layer 3 — In-loop budget visibility (prevention)

Inject a single line at every assistant turn boundary so the agent
can self-throttle:

```
<budget remaining tool_calls=43 turns=22 wall_clock=812s/>
```

For the Claude SDK we can't intercept turn-by-turn (it streams), so
instead we inject the expected total upfront in the system prompt
delivered via `_append_run_controls`:

```
You have a budget of 220 tool calls and 80 turns for this task.
Plan accordingly. If you reach 80% of either budget, produce your
final structured output immediately.
```

Cheap, no SDK changes, observably reduces overrun rate (the same
trick GPT-engineer and OpenAI's o1-style scaffolds use).

### Layer 4 — Tool-call dedup (prevention)

Add a tiny in-orchestrator cache keyed by `(tool_name, sorted_input_hash)`.
On a duplicate call within the same agent invocation:

- Read / Bash with identical args → return the cached output, do not
  charge the budget, log `tool_call_dedup` event in telemetry.
- WebFetch / Bash with non-deterministic args (timestamps, etc.) →
  no cache.

This would have killed ~25 of the supervisor's 80 calls outright.
Implementation is < 50 lines in the existing `StreamToolCall` branch.

## Why this is elegant

| `/token` | `/ml-code-quality` | `/ml-experiment-rigor` |
|---|---|---|
| Plan-on-disk; Codex implements without re-deriving the design. | Per-role budget is config (`role_overrides`), not branching. | Every recovery attempt logged in `agent_telemetry.jsonl` with `recovery_kind`. |
| One Codex session does Layers 1–4 in a single sweep. | `RecoveryStrategy` is a Protocol — new strategies plug in without orchestrator edits. | Recovered outputs flagged distinct from successful ones; downstream verifiers can decide whether to trust them. |
| Tests are pure-function (no LLM): role→budget mapping, recovery prompt builder, dedup hash. | Boundary validation: recovery JSON must parse against the agent's existing `_OUTPUT_MODELS` schema, else raise. | `provenance.json` records the budget that fired and the recovery outcome — auditable. |

## Module layout

Surgical — no new top-level package, all changes inside the existing
runtime + orchestrator surface.

```
backend/agents/execution.py
  + role_overrides: dict[str, RoleBudget]          (data class)
  + RoleBudget: max_turns / max_tool_calls / wall_clock_seconds
backend/agents/runtime/recovery.py                 (NEW, ~120 lines)
  + class SoftStopRecovery(Protocol)
  + class StructuredOutputRecovery (default impl)
  + build_recovery_prompt(original, kind, value)
backend/agents/orchestrator.py
  ~ _build_runtime_spec: pick budget by role from execution_profile
  ~ _invoke_agent: on AgentLimitExceeded / TimeoutError / SDK turn-cap,
    invoke registered recovery strategy before re-raising
  ~ telemetry: add recovery_kind, recovery_success fields
  ~ _append_run_controls: inject "You have a budget of N tool calls..."
  + tool-call dedup cache (per-invocation, in-memory dict)
backend/agents/telemetry.py
  ~ AgentInvocationRecord: add recovery_kind, recovery_attempts,
    deduped_tool_calls
tests/test_limit_recovery.py                       (NEW, 6 tests)
```

No frontend changes required for v1 — the timeline panel already
renders `error_message`; we'll piggyback `recovery_kind` into the
displayed message.

## Acceptance gate

- [ ] `pytest -q` green (576+ tests; +6 new)
- [ ] `tsc --noEmit` clean (no frontend changes)
- [ ] Re-run the same paper that failed today; `supervisor-verifier`
      either completes in ≤ 220 tool calls OR triggers soft-stop
      recovery and produces a parseable structured output
- [ ] `agent_telemetry.jsonl` for that run contains
      `recovery_kind: "soft_stop_summary"` records when the cap fires
- [ ] `learn.md` gains an entry: "2026-05-10 — Budget caps must
      recover, not crash"
- [ ] `CHANGELOG.md` `[Unreleased]` updated under Changed/Fixed

## What this is NOT

- Not auto-retry with bumped budget (cost runs away, hides the
  inefficiency root cause)
- Not silent fallback (recovery is **explicit** in telemetry +
  output)
- Not a model swap (we keep the same provider; cross-provider
  fallback is the existing `claude_limit_fallback_runtime` path
  for usage-limit errors, which is orthogonal)
- Not a frontend change (out of scope for v1)
