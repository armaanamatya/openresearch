# Subscription Cost Reduction — Sub-Agent Burn Design (2026-05-28)

> Companion to the brief in chat. Source of truth for the design; consumed by an
> executing-plans pass once approved.

## TL;DR

Three-phase fix to cut sub-agent Sonnet token burn per run by an estimated 30-50%,
in priority order:

- **Phase 0 (Measurement, ~15-line surgical change to `runtime/invoke.py`):** the
  StreamUsage events emitted by `claude_runtime.py:113-122` are silently dropped
  in `runtime/invoke.py:46-57` — the async-for loop handles `StreamText` and
  `StreamToolCall` but has no `StreamUsage` branch. Add it, route into the
  existing `cost_ledger.jsonl`, emit a `subagent_usage` SSE event. Zero behavior
  change; you simply see what you are spending.
- **Phase 1 (Retry-burst):** snapshot partial code state during `implement_baseline`
  so the B-001 retry resumes from the partial files instead of re-rolling the
  sub-agent from scratch. Drop the pre-emit stall threshold from 240s to 60s.
- **Phase 2 (Bound per-call growth):** cache-tag the stable sub-agent prefix
  (system prompt + tool defs) with `cache_control: ephemeral` + 1h extended TTL.
  Add `PlateauPolicy` as a CEILING above the existing `ForcedIterationPolicy`
  FLOOR. Reduce Hermes audit cadence to {iter 0, every Nth, last}, gated on
  audit-chain owner sign-off.

All changes preserve the rediscovery loop — paper-level interpretations remain
uncached cross-run.

## Code evidence

Verified by reading:

- `backend/agents/runtime/claude_runtime.py:113-122` — yields `StreamUsage` with
  full token counts (input, output, cache_read, cache_creation). Data already
  correct.
- `backend/agents/runtime/invoke.py:46-57` — `collect_agent_text` consumes the
  stream but **does not branch on `StreamUsage`**. Events dropped silently. This
  is the path `baseline_implementation.run_with_sdk:1842` uses for
  `implement_baseline` — the single biggest token sink.
- `backend/agents/resilience/cost.py:30-33, 59-86` — `CostLedgerEntry` already
  has the four token fields and `from_usage()` accepts exactly the StreamUsage
  shape. No schema change needed.
- `backend/agents/resilience/engine.py:304-308, 428-435` — `ResilientEngine`
  DOES correctly wire StreamUsage → cost_ledger. But `collect_agent_text` does
  not route through the engine — it calls `selected_runtime.run_agent()`
  directly. The engine path is correct; the invoke path is the bug.
- `backend/agents/rlm/primitives.py:89` — `_DEFAULT_PRE_EMIT_STALL_S = 240.0`,
  override-able via `REPROLAB_PRE_EMIT_STALL_S`.
- `backend/agents/rlm/forced_iteration.py` — 494-line module; provides the
  pattern (thread-local stack + `LocalREPL._final_var` patch + context manager)
  that `PlateauPolicy` will mirror.

## Architecture

### Pillar A: Measurement (Phase 0 — unblocks everything)

Without this, no fix can be validated. Three components:

1. **`StreamUsage` wire-up in `collect_agent_text`.** Add the branch that
   accumulates per-call usage; on stream end, build a `CostLedgerEntry.from_usage`
   and append to a ledger discovered from `project_dir` (path-based, mirroring
   how `_emit_dashboard_event_to_path` works in primitives.py:675).
2. **Per-primitive `agent_id` tagging.** `collect_agent_text` already takes
   `agent_id` ("baseline-implementation" today); use it as the ledger
   `agent_id` so per-primitive burn is attributable.
3. **`subagent_usage` SSE event** through the existing `dashboard_events.jsonl`
   chokepoint. No SSE allowlist change needed — events flow generically.

**Why this is the gating fix:** every subsequent change is invisible without
it. Once it lands, the dark side of the cost ledger becomes visible and we can
prioritize the rest of the work by actual measured impact rather than suspicion.

### Pillar B: Eliminate retry-burst (Phase 1)

The B-001 pre-emit stall (claude-agent-sdk aclose deadlock, documented in
`docs/superpowers/specs/2026-05-22-sdk-aclose-investigation.md`) currently
costs a full sub-agent rollout per hang. Two changes:

1. **Checkpoint partial state.** `implement_baseline`'s sub-agent writes files
   incrementally to `runs/<id>/code/`. After every successful tool-call write
   event (visible at the StreamToolCall layer), snapshot the directory listing
   + sizes to `runs/<id>/_baseline_checkpoint/<attempt_idx>/manifest.json`. The
   files themselves stay where they are — the manifest just records what was
   reached.
2. **Resume from partial.** On retry, `implement_baseline` builds the sub-agent
   prompt with a prefix:
   > You started this implementation in a previous attempt and reached the
   > following state: `<file listing with sizes>`. Read these files first, do
   > not rewrite them unless they are wrong, and complete the missing pieces.

The token cost of a resumed attempt scales with the SIZE OF THE GAP, not the
size of the full implementation. Empirically expected: <30% of fresh-rollout
tokens.

3. **Drop pre-emit stall threshold to 60s.** `_DEFAULT_PRE_EMIT_STALL_S =
   240` is excessive once we resume from partial. Set to 60. Tune via
   `REPROLAB_PRE_EMIT_STALL_S` if false positives appear.

Existing retry cap (`_MAX_TRANSIENT_RETRIES = 3` at primitives.py:85) is
unchanged — beyond 3 attempts we still fail to user.

### Pillar C: Bound per-call growth (Phase 2)

Three independent sub-changes, each landable separately.

**C1: Cache-tag stable sub-agent prefix.**

The expensive part of every sub-agent call is the prefix: AGENT_REGISTRY's
system prompt + tool definitions + guardrail boilerplate (claude_runtime.py:146
`_with_guard_prompt`). These are paper-AGNOSTIC and stable across the run.

Investigation FIRST: does `claude_agent_sdk.ClaudeAgentOptions` propagate
`cache_control` markers to the underlying Anthropic API? Run

```python
from claude_agent_sdk import ClaudeAgentOptions
import inspect
print(inspect.signature(ClaudeAgentOptions.__init__))
```

If yes, set `cache_control={"type": "ephemeral", "ttl": "1h"}` at the END of the
stable system prompt segment and on the tool-def block.

If no, two options:
- **Option A (preferred):** patch the SDK to plumb cache_control through. Small
  PR upstream.
- **Option B (fallback):** bypass the SDK for hot sub-agent calls; use the raw
  `anthropic.Anthropic` client with explicit cache_control. Loses MCP/sub-agent
  abstractions; only worth it if Option A is blocked.

Use the **1h extended cache TTL** (beta header
`anthropic-beta: extended-cache-ttl-2025-04-11`). The 5-min default blows the
cache during long `implement_baseline` runs; 1h survives even an 8-iteration
SDAR. Cost: 2× standard cache write per refresh, but only ONE write per hour vs
one per ~5min of work = net win.

**Caveat on `_with_guard_prompt`:** if `agent.guard.blocked_terms` varies
per-call (paper-specific blocked PaperBench resources), the appended guardrail
breaks cache stability. Move the guardrail to a SEPARATE block (so the stable
prefix-cache boundary precedes it), or hoist common terms into a static prefix
and only append per-call deltas after the cache boundary.

**C2: PlateauPolicy ceiling.**

New file: `backend/agents/rlm/plateau.py`. Mirrors `forced_iteration.py`'s
pattern (thread-local stack, context manager, REPL `_final_var` interceptor).

Logic:

```
if iteration_count < forced_iteration_floor: defer to forced_iteration policy
elif len(rubric_history) >= REPROLAB_PLATEAU_WINDOW:
    delta = max(history[-WINDOW:]) - min(history[-WINDOW:])
    if delta < REPROLAB_PLATEAU_EPSILON:
        emit run_warning code="plateau_detected"
        AUTHORIZE next FINAL_VAR  # hint, not force
```

The policy is **hint-based**: it emits a warning that the root model sees at the
start of its next iteration (via the existing `check_user_messages` plumbing).
The root chooses to FINAL_VAR or continue. This respects root autonomy and
avoids cutting off legitimately-active exploration.

Composition with forced_iteration: floor is unchanged; plateau lives above it.
Both push onto the same thread-local policy stack.

Defaults: `REPROLAB_PLATEAU_EPSILON=0.02`, `REPROLAB_PLATEAU_WINDOW=2`.

**C3: Hermes audit cadence.**

Add `should_run_audit(iteration_n, last_audit_iter, mode)` gate in
`backend/hermes_audit/providers.py`. Default cadence: iter 0, every Nth iter
(N=`REPROLAB_HERMES_AUDIT_CADENCE`, default 3), and final iter (triggered when
a FINAL_VAR is accepted).

**GATED ON OWNER SIGN-OFF.** The audit chain may rely on per-iteration audit for
its tamper-evidence properties. If owner confirms periodic auditing is
acceptable, ship. If not, skip C3 (still get token savings from A + B + C1 + C2).

### Pillar D (deferred): Workflow-scaffolding archetype cache

Out of scope for v1. Logged here so it does not get lost. Idea: maintain a
library of paper-archetype templates (RL agentic, diffusion, transformer
pretraining…). When a new paper matches an archetype, `implement_baseline`
starts from the scaffold skeleton (loader patterns, eval harness shapes,
DataLoader idioms — NOT paper-specific behavior). Would amortize "writing the
same DataLoader from scratch" across runs without leaking paper interpretation.

Defer until Phases 0-2 measurements show whether it would meaningfully help.

## What is NOT changing

- The 12 domain primitives' signatures or the REPL `context`-variable shape.
- The SSE bridge sanitization layer (`backend/agents/rlm/sse_bridge.py`).
- The `primitive_cache` WITHIN-run memoization (already cached intra-run for
  understand_section, extract_hyperparameters, detect_environment,
  plan_reproduction; cross-run caching off-limits per discovery constraint).
- Root model auth path. `claude-oauth` continues to ride the existing 98% cache
  hit rate.
- RunPod sandbox or dynamic GPU resolution.
- `REPROLAB_MIN_RUBRIC_ITERATIONS` semantics (still a floor).
- `_MAX_TRANSIENT_RETRIES = 3` retry cap.

## Code-touchpoint diff plan

### Phase 0 (Measurement)

**`backend/agents/runtime/invoke.py` (the ~15-line surgical fix)**

Add a `StreamUsage` branch to the async-for at line 46-57. Accumulate usage
totals; on completion, write to cost_ledger via a path-based helper. Pseudo-diff:

```python
from backend.agents.runtime.base import StreamUsage
from backend.agents.resilience.cost_path_writer import record_subagent_usage_to_path

_inner_usage = {"input_tokens": 0, "output_tokens": 0,
                "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
async for event in selected_runtime.run_agent(...):
    if isinstance(event, StreamText): ...
    elif isinstance(event, StreamToolCall): ...
    elif isinstance(event, StreamUsage):
        for k in _inner_usage:
            _inner_usage[k] += getattr(event, k, 0) or 0
return _inner_collected, _inner_tool_calls, _inner_usage
```

After `run_isolated`, call `record_subagent_usage_to_path(project_dir, agent_id,
spec.model, selected_runtime.provider_name, _inner_usage)`.

**`backend/agents/resilience/cost.py`**

Add a new module-level helper (or a sibling `cost_path_writer.py`):

```python
def record_subagent_usage_to_path(
    project_dir: Path,
    agent_id: str,
    model: str,
    provider: ProviderName,
    usage: dict[str, int],
    *,
    attempt_index: int = 0,
) -> None:
    """Path-based ledger write — usable from contexts without a RunCostLedger."""
    ledger_path = project_dir / "cost_ledger.jsonl"
    entry = CostLedgerEntry.from_usage(
        agent_id=agent_id, attempt_index=attempt_index,
        provider=provider, model=model, usage=usage,
    )
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry.to_json(), sort_keys=True) + "\n")
        fh.flush(); os.fsync(fh.fileno())
    # Also emit subagent_usage SSE event for live observability:
    _emit_dashboard_event_to_path(
        project_dir,
        event_type="subagent_usage",
        payload={"agent_id": agent_id, "model": model, **usage,
                 "estimated_usd": entry.estimated_usd or 0.0},
    )
```

**`backend/agents/resilience/pricing.py`**

Verify the active Sonnet model string (likely `claude-sonnet-4-5` per CLAUDE.md
"most recent Claude model family is Claude 4.X") is in the pricing table so
`estimated_usd` is non-null. Pricing under subscription mode is a sanity check,
not the metric — token counts are what matter.

### Phase 1 (Retry-burst)

**New file: `backend/agents/rlm/baseline_checkpoint.py`**

```python
def snapshot_partial(code_dir: Path, attempt_idx: int) -> Path:
    """Write manifest of files present in code_dir at this point."""
    ...

def load_partial(code_dir: Path) -> dict | None:
    """Return latest manifest for prompt construction."""
    ...

def build_resume_prefix(manifest: dict) -> str:
    """Construct the 'you've already written X' prompt prefix."""
    ...
```

**`backend/agents/baseline_implementation.py`**

At `run_with_sdk` (line 1718), accept an `attempt_idx` parameter. Before the
prompt construction at line 1816, if `attempt_idx > 0`, call
`load_partial(code_dir)` and prepend the resume prefix to `prompt`. After the
`await collect_agent_text(...)` call (line 1842), call `snapshot_partial(code_dir,
attempt_idx)` so the NEXT retry has a manifest.

**`backend/agents/rlm/primitives.py`**

At `implement_baseline` (line 1202), thread an `attempt_idx` counter through the
warm-retry cache loop into `_run_baseline_with_sdk`. Drop
`_DEFAULT_PRE_EMIT_STALL_S` to `60.0`.

### Phase 2 (Cache + plateau + cadence)

**`backend/agents/runtime/claude_runtime.py` (C1)**

Investigation: confirm `ClaudeAgentOptions` accepts cache_control. If yes, at
line 82-91 where `options = ClaudeAgentOptions(...)` is constructed, split
`system_prompt` into stable_prefix + per-call suffix; apply cache_control to the
prefix.

If SDK does not support cache_control, hoist hot sub-agent calls to a parallel
path that uses raw `anthropic.Anthropic.messages.create(...)` with explicit
cache_control on the system and tool blocks. Keep the SDK for everything else.

**`backend/agents/runtime/claude_runtime.py:146-154` (`_with_guard_prompt`)**

Audit `agent.guard.blocked_terms` stability. If terms are per-call (unlikely but
possible), restructure so guardrails do not break the cache boundary.

**`backend/agents/rlm/system_prompt.py`**

If root system prompt construction already achieves 98% cache hit rate, this
file is untouched. The SUB-AGENT system prompts live in `backend/agents/registry/`
(`AGENT_REGISTRY[agent_id].to_runtime_spec(...)`); audit there for stable vs
dynamic content separation.

**New file: `backend/agents/rlm/plateau.py` (C2)**

Mirror `forced_iteration.py`. ~150-200 lines. Exposes:

```python
@dataclass
class PlateauPolicy:
    min_window: int
    epsilon: float
    rubric_history: Callable[[], list[float]]
    current_iteration: Callable[[], int]
    forced_iteration_floor: int
    on_plateau: Callable[[str], None] | None = None

@contextmanager
def plateau_policy(policy: PlateauPolicy) -> Iterator[PlateauPolicy]: ...
```

**`backend/agents/rlm/run.py`**

Push `plateau_policy(...)` onto the same scope as `forced_iteration_policy(...)`.

**`backend/agents/rlm/rubric_guard.py`**

Bound `repair_context` to `REPROLAB_REPAIR_CONTEXT_MAX_BYTES` (default 2048).
Truncate-with-marker.

**`backend/hermes_audit/providers.py` (C3 — gated)**

Add `should_run_audit(iteration_n, last_audit_iter, mode) -> bool`. Default
cadence: iter == 0, (iter - last_audit_iter) >= REPROLAB_HERMES_AUDIT_CADENCE
(default 3), or mode == "final". Wire the gate at every audit call site.

### Files not touched

- `backend/agents/resilience/budget.py` — Phase 2 may add a `WeeklyTokenBudget`
  later, but not in this design. Weekly-quota tracking is heuristic-only; defer
  until measurement shows whether it is worth the noise.

## Measurement plan

### Pre-fix baseline (capture BEFORE Phase 0)

Run 3 SDAR full reproductions on `main`. For each run, record:

1. Wall clock to FINAL_VAR (or to budget exhaustion).
2. Iteration count.
3. Per-primitive call counts (count `primitive_call` events in
   `dashboard_events.jsonl` grouped by `data.name`).
4. Anthropic dashboard Sonnet spend (manual screenshot, if dashboard accessible
   for the subscription tier).
5. cost_ledger.jsonl sub-agent row count — expect ~0 today.

Save to `docs/runbooks/2026-05-28-baseline-burn/`.

### Phase 0 gate — Measurement landed

**PASS condition 1:** `cost_ledger.jsonl` after one SDAR iteration contains
non-zero `tokens_in` and `tokens_out` rows for each of: `baseline-implementation`,
`plan_reproduction` (if it routes through `collect_agent_text`),
`propose_improvements`, `verify_against_rubric`, and any `hermes_audit_*` agents.

**PASS condition 2:** Sum of all `input_tokens + cache_read_input_tokens +
cache_creation_input_tokens` ≈ Anthropic dashboard spend (within 10%).

**PASS condition 3:** Each row has a non-empty `agent_id` matching the primitive
name.

**FAIL diagnostic:** if rows are still zero, the StreamUsage branch is correct
but `record_subagent_usage_to_path` is not being reached. Add a `logger.debug`
line and trace.

### Phase 1 gate — Retry-burst eliminated

**Setup:** add `REPROLAB_FORCE_PRE_EMIT_HANG_AT_ATTEMPT=1` (test-only env var)
that simulates an aclose hang on the first implement_baseline attempt.

**PASS condition 1:** the second attempt's `tokens_in` is < 30% of the first
attempt's `tokens_in`.

**PASS condition 2:** `runs/<id>/_baseline_checkpoint/0/manifest.json` exists
and lists ≥1 file present after the hung attempt.

**PASS condition 3:** the resume prompt (DEBUG-logged) contains "you've already
written" and lists the snapshot files.

**E2E:** run SDAR with 5 forced hangs interspersed; total cost is < 2× a clean
run.

### Phase 2 gate — Cache + plateau + cadence

**C1 cache hit rate:** `sum(cache_read_input_tokens) / sum(input_tokens) >= 0.6`
across sub-agent rows in a single 10-iteration run.

  - Rationale: each iteration's FIRST sub-agent call is a cache miss; intra-iteration
    calls (which fire seconds apart) hit cache. ~6 hits per 10 misses → 60% floor.
  - If SDK does not support cache_control and we do not bypass, record the
    result and skip this gate.

**C2 plateau early exit:** SDAR run with `REPROLAB_PLATEAU_EPSILON=0.02`,
`REPROLAB_PLATEAU_WINDOW=2` terminates 3-5 iterations earlier than baseline AND
a `plateau_detected` warning event fires before FINAL_VAR.

**C3 audit cadence:** `hermes_audit_*` row count ≤ `ceil(iteration_count / 3)
+ 1` (vs == iteration_count today).

### Post-deployment KPI (1 month)

Track per-week:

1. Mean total `tokens_in + cache_read + cache_creation` per SDAR reproduction.
2. Number of successful runs per week.
3. Subscription cap exhaustion day (target: day 7+, current: day 5).
4. Failure-class distribution (B-001 hangs as % of total).

**Target:** day-5 exhaustion → day-7+ exhaustion. 1.5-2× runs per week.

## Risks and open questions

1. **claude_agent_sdk cache_control support** is unverified. Phase 2 starts
   with an investigation deliverable. If unsupported AND SDK patch is blocked,
   fallback Option B (raw SDK bypass) loses MCP and AgentDefinition abstractions
   — acceptable for hot sub-agent calls only.
2. **Audit chain integrity vs cadence change.** C3 is gated on audit-chain
   owner confirmation. If every-iteration audit is required for tamper-evidence,
   skip C3; still get the other savings.
3. **Phase 0 may reveal a different bottleneck.** If root model burn dominates
   sub-agent burn (despite 98% cache hit, the absolute token volume of 374K
   per run × N runs may exceed sub-agent volume), the plan stays valid but
   priorities shift — measurement is what tells us.
4. **Pre-emit stall threshold drop 240→60s** may produce false-positive hang
   triggers on slow-but-honest sub-agents. Tune via
   `REPROLAB_PRE_EMIT_STALL_S` if observed.
5. **PlateauPolicy false positives** — two zero-delta iterations during active
   exploration would prematurely hint at exit. Mitigation: the hint is advisory,
   not forced; root chooses.
6. **WeeklyTokenBudget (deferred)** — Anthropic does not expose remaining
   subscription quota via API. Any local tracker is conservative-only. Not in v1.
7. **Workflow scaffolding (Pillar D, deferred)** — defer until Phases 0-2
   measurements justify the effort.

## Sequencing

| Phase | Days | Deliverable | Behavior change? |
| --- | --- | --- | --- |
| 0 — Measurement | 2-3 | Sub-agent rows in cost_ledger; `subagent_usage` SSE event. | None. |
| 1 — Retry-burst | 4-5 | Checkpoint + resume + stall threshold drop. | Yes (retries differ). |
| 2 — Cache + plateau + cadence | 5-7 | cache_control + PlateauPolicy + Hermes cadence. | Yes (each piece). |

Phase 0 is the gate for all subsequent work — without it the changes cannot be
verified. Each phase ships independently.

## Composition with existing systems

- **Anthropic prompt cache** — C1 explicitly leverages it via `cache_control:
  ephemeral` markers. 1h extended TTL via beta header.
- **RunPod sandbox** — untouched; sub-agents do not call RunPod.
- **`forced_iteration` floor** — preserved. PlateauPolicy lives above it as
  a hint, not a force.
- **`primitive_cache` (intra-run memoization)** — preserved. Distinct from
  cross-run caching that the user ruled out.
- **`dashboard_events.jsonl`** — new `subagent_usage` event flows through it
  generically. No SSE allowlist entry needed.
- **`cost_ledger.jsonl`** — schema unchanged; only consumer-side wire-up changes.
- **`REPROLAB_FORCE_SANDBOX`, `REPROLAB_DYNAMIC_GPU`** — orthogonal; no
  interaction.
- **`claude-oauth` subscription auth path** — preserved; this is what the
  measurement counts AGAINST.

## What addresses each stated requirement

| Requirement | Pillar |
| --- | --- |
| 1. Reduce sub-agent token burn per run | C1 (cache prefix), C2 (early exit), C3 (audit cadence), and B (no retry-burst). |
| 2. Eliminate retry-burst cost on B-001 stall | Phase 1 (checkpoint + resume + 60s stall). |
| 3. Measure sub-agent token usage | Phase 0 (the 15-line invoke.py fix). |
| 4. Respect that paper-level interpretation cannot be cached | Only paper-AGNOSTIC prefixes are cache-tagged; primitive_cache stays intra-run. |
| 5. Compose with Anthropic cache, RunPod, forced_iteration | Explicit composition section above. |

---

## Live Run Issues — SDAR prj_e2e07ed9fd6cad78 (2026-05-28)

Observed during a live test run of `arxiv:2605.15155` in `rlm` mode, `claude-oauth` root, `local` sandbox, `efficient` + `minimize_compute=true`.

### BUG-LR-001: Root model returned empty response in iteration 1 (CRITICAL)

**Observed:** `rlm_state/iterations.jsonl` shows `{"iteration":1,"response":"","code_blocks":[],"sub_calls":0,"timing":66.4}` — the claude-oauth root produced zero Python code, zero primitive calls after 66 seconds.

**Root cause hypothesis:** Two aclose-pre-result SDK isolation retries fired before iteration 1 (PIDs 19028 and 20553 in `runner.stderr.log`). The third attempt succeeded at the SDK level (no exception) but the claude CLI returned empty text. The RLM library treated empty-response as a valid (but empty) iteration and continued to iteration 2.

**Impact:** Zero productive work in the first iteration, and potentially in subsequent iterations if the same empty-response pattern repeats. Token budget consumed for nothing.

**Workaround (not yet implemented):** The `claude-oauth` path's `ClaudeLlmClient.complete()` should validate that the returned text is non-empty and re-raise as a retriable error if empty. This would surface the failure instead of silently burning an iteration.

**Files:** `backend/agents/rlm/claude_oauth_client.py` (add non-empty validation on the completion result).

### BUG-LR-002: aclose-pre-result errors on startup (HIGH)

**Observed:** Two consecutive `RuntimeError: aclose(): asynchronous generator is already running` errors logged at startup, each from a different subprocess PID (19028, 20553). Both are `Loop ... that handles pid X is closed` variants — the asyncio event loop in the sdk_isolation worker thread closed while the async generator cleanup was in progress.

**This is B-001 from the design doc** — confirmed triggering on the FIRST primitive call (rubric generation or the initial claude-oauth root turn), not just `implement_baseline`. The stall watchdog (240s threshold) was NOT the mechanism here; the errors happen during cleanup, not during a pre-emit hang.

**Impact:** Wasted SDK call retries at the start of every run. With `_DEFAULT_MAX_RETRIES=2` in `sdk_isolation.py`, up to 3 total attempts fire before one succeeds (and may still return empty content per BUG-LR-001).

**Files:** `backend/agents/runtime/sdk_isolation.py`, `backend/agents/rlm/claude_oauth_client.py`.

### BUG-LR-003: UI shows blank main panel after iteration 1 (MEDIUM)

**Observed:** After the run's first iteration completed (empty response), the Playwright snapshot shows `main > generic [ref=e79]` with no children — completely blank. The status says "running" but the UI gives no indication of progress, iteration count, or current phase.

**Expected:** The UI should at minimum show "Iteration 1 complete — no code generated, retrying" or the iteration counter, even when the RLM root emits no code. The blank main panel makes debugging very difficult.

**Files:** `frontend/src/components/lab/rlm/` — the `repl_iteration` SSE event (or equivalent) never fired because no `dashboard_events.jsonl` was created; the SSE stream has nothing to show.

### BUG-LR-004: leaf-scores endpoint polled during active run (LOW)

**Observed:** Frontend calls `GET /api/demo/runs/{id}/leaf-scores` repeatedly (every ~20s) while the run is in `running` state. Backend returns 404 ("Run not found or scoring not complete"). Each call generates a console error in the browser.

**Fix:** Gate the `leaf-scores` fetch on `status === "completed"` or on the existence of a rubric score in the run state. No fetch should be made during an active run.

**Files:** Likely `frontend/src/components/lab/rlm/rubric-breakdown.tsx` or wherever `leaf-scores` is fetched — add a `status !== "completed"` early-return guard.

### BUG-LR-005: /reports polled at very high frequency (LOW)

**Observed:** Backend access log shows `GET /runs/{id}/reports` firing multiple times per second from the frontend. This is polling churn that produces zero value during an active run (reports only exist post-completion).

**Fix:** Add a minimum polling interval (e.g. 10s) or gate on run status, similar to leaf-scores.

### Phase 0 validation status

Phase 0 changes landed (`invoke.py` + `cost.py`), tests green. However, **the live run did not reach any sub-agent primitive calls** (BUG-LR-001/002 caused the RLM to stall before `implement_baseline`), so the `subagent_usage` rows in `cost_ledger.jsonl` were never written. Phase 0 correctness is confirmed by unit tests only; live validation pending a run that actually reaches `implement_baseline`.

### BUG-LR-006: UI blank on fresh page load during active run (MEDIUM)

**Observed:** Playwright snapshot at cycle 3 (elapsed ~6m) shows `main > generic` with no children — blank panel — despite 36 live dashboard events and active sub-RLMs. Cycle 2 loaded correctly and showed the pipeline nav, sub-RLM status, and 12 primitive calls. The difference: cycle 2 was navigated to a page that was already open; cycle 3 did a fresh `page.goto()`.

**Root cause hypothesis:** The SSE stream (`/runs/{id}/events`) sends events as they arrive; it does NOT replay `dashboard_events.jsonl` history on a new connection. A fresh page load that misses the initial `repl_iteration` event has nothing to render in the main panel.

**Expected:** On fresh load of `/lab?projectId=X` for a running run, the frontend should hydrate from the snapshot in `demo_status.json` (or a dedicated `/runs/{id}/snapshot` endpoint) so the pipeline phase, iteration count, and primitive call history are visible immediately.

**Files:** `frontend/src/components/lab/rlm/rlm-lab.tsx` — the initial hydration path on mount; `backend/services/events/live_runs.py` — could expose a replay endpoint.

### BUG-LR-007: Zero-token cost ledger rows for heuristic primitives (INFO)

**Observed:** `cost_ledger.jsonl` has 7 rows all with `tokens_in=0, tokens_out=0` for `understand_section` (×3), `extract_hyperparameters`, `check_user_messages`, and `heartbeat`. These all use local heuristics (no LLM call), so zero tokens is **correct**.

**Impact:** None. But these zero-row entries add noise to the ledger and inflate "sub-agent call count" metrics. Consider filtering them out of the cost summary display, or only writing ledger rows when `tokens_in > 0`.

---

## Feature Request: Node Click Breakdown (2026-05-28)

**Screenshot context:** At ~7m27s elapsed the canvas shows: `Paper` (box), `Comprehension` (box), three `Understand` circles (kind=`llm_primitive` from `understand_section`), and one `Extract Hyp...` circle (kind=`llm_primitive` from `extract_hyperparameters`). The right sidebar shows aggregate counters only — no per-node detail.

**Request:** Clicking any node in the canvas should show a breakdown of that specific node's activity in the detail sidebar.

### What "breakdown" means per node kind

| Kind | Data to show |
|------|-------------|
| `primitive` | Primitive name (friendly + raw), status badge, iteration #, timestamp, args_summary key:type pairs, result_summary in full |
| `llm_primitive` | Same as primitive; additionally flag as LLM-using; show full result_summary (not truncated to 200 chars as in the list) |
| `subrlm` | Full query question (model field), full prompt_preview (not truncated), timing, completion status, error if any |
| `work` | Already works — shows all primitive calls |
| `paper` | Already works — shows paper metadata |
| `candidate` | Already works — shows category, description, rubric delta |
| `baseline` | Already works — shows rubric score |

### Current gap

`SidebarBody` in `node-detail-sidebar.tsx` has no case for `kind === "primitive"` or `kind === "llm_primitive"` — they fall through to the `declined-group` catch-all and render either `nowBlock` or "no iteration detail".

### Implementation notes

- Node id format: `prim-${primitiveName}-${N}` where `N = primitiveCalls.length + 1` at call time → call index = `N - 1`
- The last hyphen-delimited token is always the numeric suffix since primitive names use underscores
- `PrimitiveCallView` carries: `primitive`, `status`, `args_summary`, `result_summary`, `iteration`, `rubric_delta`, `timestamp`
- For `llm_primitive`, also check if there's a sub-RLM whose `spawnedAt` timestamp closely follows the call's `timestamp` — if so, surface its `model` (the query question) and `duration_ms`
- Reuse existing `styles.metaList/metaRow/metaKey/metaVal` and `styles.primitiveResult` CSS — no new tokens needed

### Files changed

- `frontend/src/components/lab/rlm/node-detail-sidebar.tsx` — added `PrimitivePanelDetail` component + cases for `primitive` and `llm_primitive` in `SidebarBody`
- `frontend/src/components/lab/rlm/node-detail-sidebar.module.css` — added `.primitiveDetail`, `.primitiveSection`, `.primitiveSectionLabel`

**Status: IMPLEMENTED and verified via Playwright.** Clicking an "Understand" (llm_primitive) node shows: primitive name, status, called-at, phase, type=LLM call, result structure, and sub-RLM query (question + duration + prompt preview). Example confirmed: `understand_section` → `dict[ambiguities, datasets, hardware_clues, metrics, outcome, training_recipe]`, sub-RLM question = "What are the main results...", duration = 56.6s.

---

### BUG-LR-008: cost_ledger.jsonl written to code/ subdirectory for implement_baseline (HIGH)

**Observed:** `implement_baseline` row in `runs/<id>/cost_ledger.jsonl` shows `tokens_in=0, tokens_out=0`. Actual token data (`in=3880, out=24209, cache_create=66455`) was written to `runs/<id>/code/cost_ledger.jsonl` because `baseline_implementation.run_with_sdk` calls `collect_agent_text(project_dir=code_dir)` and Phase 0 wrote the ledger entry to that directory.

**Root cause:** `collect_agent_text` uses `project_dir` for both the agent's working directory AND the ledger write path. When the working directory is a subdirectory (code/), ledger entries land there instead of the run root.

**Fix (2026-05-28):** Added `ledger_dir: Path | None` parameter to `collect_agent_text`. When not provided, defaults to `project_dir`. The two `collect_agent_text` call sites in `baseline_implementation.py` now pass `ledger_dir=project_dir` (the run root) while keeping `project_dir=code_dir` for the agent's working directory.

**Files changed:** `backend/agents/runtime/invoke.py`, `backend/agents/baseline_implementation.py`

---

## Live Run Completion — 2026-05-28 16:03 CDT

**Run `prj_e2e07ed9fd6cad78` completed** at 21:00:47 UTC (1h 11m 34s total).

**Final score: 0.0% overall, verdict=partial, meetsTarget=False, iterations=2.**

All 3 `run_experiment` attempts failed:
1. Attempt 1: Docker build timeout (CUDA wheel, BUG-LR-009)
2. Attempt 2: Pre-flight blocked — surrogate model, no real Qwen weights (BUG-LR-010)
3. Attempt 3: Pre-flight blocked again — 3rd `implement_baseline` pass still used surrogate (BUG-LR-010)

RLM exhausted its repair budget (`max_repair_iterations=2`) and emitted `run_complete` with 0 score.

### Phase 0 Measurement Results

**Sub-agent costs** (`code/cost_ledger.jsonl` — old subprocess, pre-BUG-LR-008-fix):

| Pass | agent_id | tokens_in | tokens_out | cache_create | cache_read | cost_usd |
|------|----------|-----------|------------|--------------|------------|----------|
| 1 | baseline-implementation | 3880 | 24209 | 66455 | 630285 | $0.813 |
| 2 (repair) | baseline-implementation | 9 | 2296 | 47655 | 366158 | $0.323 |
| **Total** | | **3889** | **26505** | **114110** | **996443** | **$1.136** |

**Root model costs** (run root `cost_ledger.jsonl`):

| Primitive | tokens_in | tokens_out | cache_create | cache_read |
|-----------|-----------|------------|--------------|------------|
| plan_reproduction | 6 | 1072 | 60442 | 0 |
| run_experiment (×1) | 6 | 609 | 11083 | 0 |
| run_experiment (×2) | 6 | 430 | 8316 | 50130 |
| propose_improvements | 6 | 1012 | 60373 | 0 |

**Phase 0 fix validation:**
- ✅ Root model primitives: tokens captured in run root `cost_ledger.jsonl`
- ❌ Sub-agent tokens: still in `code/cost_ledger.jsonl` — BUG-LR-008 fix requires a NEW run (old subprocess had pre-fix code loaded)
- ✅ `StreamUsage` events captured and aggregated correctly
- ✅ `subagent_usage` SSE events emitted (visible in `code/dashboard_events.jsonl`)

**UI final state:** 0 console errors on completed run (BUG-LR-004 404 polling stops after run completes).

### Next Steps
1. Start a new run to validate BUG-LR-008 fix (sub-agent tokens should land in run root)
2. Fix BUG-LR-010 permanently: add `from_pretrained` validation earlier so implement_baseline generates real Qwen weight loading, not surrogates — the pre-flight catches it but wastes one `run_experiment` round-trip per iteration
3. Fix BUG-LR-009 permanently: generate `FROM runpod/pytorch:2.1.0-...` or CPU-only torch in the Dockerfile from the very first `implement_baseline` pass, not the repair

---

## Live Run Observations — 2026-05-28 ~15:58 CDT (monitoring check #7–8)

**Status:** Rapid iteration — 3 failures resolved, now at `implement_baseline` pass 3. 164 events.

### BUG-LR-010: Pre-flight blocked on surrogate model instead of real Qwen weights (HIGH)

**Observed:** `run_experiment` attempt 2 returned `success=False, failure_class=preflight_blocked`. Hard violation from `validate_code_pre_flight`: "no `AutoModelForCausalLM.from_pretrained(...)` call references any of the canonical model paths (Qwen/Qwen3-1.7B-Instruct, Qwen/Qwen2.5-3B-Instruct, Qwen/Qwen2.5-7B-Instruct)". The generated `train.py` used `nn.Linear` / `nn.Embedding` surrogate LM heads.

**Root cause:** The first two `implement_baseline` passes generated lightweight surrogate code instead of loading real Qwen weights. This is the expected SDAR paper invariant — the leaf scorer is supposed to verify real weights are loaded.

**Status:** RLM root received the violation, called `verify_against_rubric` (instant, heuristic) → `propose_improvements` (in=6, out=1012, cc=60373) → `implement_baseline` pass 3 started at 20:57:57 UTC. Root model should now generate code with `AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-1.7B-Instruct')`.

### Cost summary so far (run root cost_ledger.jsonl)

| Primitive | tokens_in | tokens_out | cache_create | cache_read |
|---|---|---|---|---|
| check_user_messages | 0 | 0 | 0 | 0 |
| understand_section ×3 | 0 | 0 | 0 | 0 |
| extract_hyperparameters | 0 | 0 | 0 | 0 |
| plan_reproduction | 6 | 1072 | 60442 | 0 |
| implement_baseline (pass 1 preamble) | 0 | 0 | 0 | 0 |
| run_experiment (attempt 1, timeout) | 6 | 609 | 11083 | 0 |
| implement_baseline (pass 2 preamble) | 0 | 0 | 0 | 0 |
| run_experiment (attempt 2, preflight) | ? | ? | ? | ? |
| verify_against_rubric | 0 | 0 | 0 | 0 |
| propose_improvements | 6 | 1012 | 60373 | 0 |
| record_candidate_outcome | 0 | 0 | 0 | 0 |

Sub-agent tokens (in `code/cost_ledger.jsonl`, old subprocess path):
- `baseline-implementation` pass 1: in=3880, out=24209, cc=66455, cr=630285, **$0.813**
- `baseline-implementation` pass 2 (repair): in=9, out=2296, cc=47655, cr=366158

**UI check:** Lab page loads, running. Only console error: BUG-LR-004. No new bugs.

---

## Live Run Observations — 2026-05-28 ~15:46 CDT (monitoring check #5)

**Status:** Repair pass complete. New `run_experiment` underway with fixed Dockerfile.

**Repair `implement_baseline` (20:42:18–20:43:22 UTC, 64s):** Sub-agent tokens in `code/cost_ledger.jsonl` (old subprocess, pre-BUG-LR-008-fix path): `in=9, out=2296, cache_create=47655, cache_read=366158`. Extremely cheap repair due to high cache hit rate — vs original `in=3880, out=24209, cache_create=66455, cache_read=630285`.

**Dockerfile fix (BUG-LR-009 addressed by RLM root):**
- Changed base image to `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-runtime-ubuntu22.04` (torch pre-installed)
- Removed `pip install torch==2.2.0 --index-url cu118` from Dockerfile entirely
- Lightweight remaining installs: `transformers==4.44.0`, `accelerate`, `datasets`, `sentencepiece`, `protobuf`, `tokenizers`, `huggingface_hub`, numpy, matplotlib, tqdm, requests
- alfworld/webshop deferred to runtime inside train.py

**New build container `pedantic_poitras`:** Running `pip install transformers...` — only 2.27MB network vs 2.28GB for the failed build. Should complete in minutes.

**Stale container:** `awesome_kepler` still Up 31 min (the timed-out CUDA wheel build). It's a zombie that will eventually be cleaned up; not blocking the new run.

**UI check:** Lab page loads normally. Only console error: BUG-LR-004 (leaf-scores 404). No new bugs.

**Next:** Build completes → new image tagged → `python train.py` runs (5 steps, CPU fallback) → `experiment_runs.jsonl` updated with attempt 2 → `verify_against_rubric` → Phase 0 cost row in run root ledger.

---

## Live Run Observations — 2026-05-28 ~15:48 CDT (monitoring check #4)

**Status:** `run_experiment` FAILED (exec_timeout). RLM repair loop started: `implement_baseline` re-invoked at 20:42:18 UTC.

### BUG-LR-009: Docker build times out downloading CUDA PyTorch wheel (HIGH)

**Observed:** `run_experiment` returned `success=False, failure_class=exec_timeout, error="build_environment: Docker build timed out after 1800 s (attempt 2)"`. The timeout occurred inside `build_environment` which tries to rebuild the Docker image from the run's Dockerfile before executing commands. The Dockerfile contains a `RUN pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu118` layer, which downloads a ~2.5GB CUDA wheel. On a Mac (Docker via Rosetta x86_64 emulation), this takes >30 min and exceeds the 1800s build timeout.

**Root cause:** The generated Dockerfile uses a slim base image (`python:3.10-slim`) and installs PyTorch from scratch inside the build. The `run_experiment` primitive calls `build_environment` to rebuild from Dockerfile before each run. In local/Docker sandbox on Mac, large CUDA wheels cause build timeout.

**Suggested fix options:**
1. `train.py` should import CPU-only torch (no CUDA URL) in local sandbox mode, or the generated Dockerfile should detect local vs GPU sandbox and use `torch` (CPU wheel, ~200MB) instead of `torch+cu118` (~2.5GB).
2. Alternatively, set `REPROLAB_RUN_EXPERIMENT_TIMEOUT_S` to > 7200 for local dev, or pass `--no-cache-dir` caching hint.
3. The pre-flight validator could detect Dockerfiles with CUDA wheel downloads and warn before wasting build time.

**Status:** Run is continuing in repair loop — RLM root received `suggested_fix: "reduce train.py epochs OR set REPROLAB_RUN_EXPERIMENT_TIMEOUT_S higher"` and started a new `implement_baseline` pass.

### Phase 0 validation update (BUG-LR-008 fix scope note)

**`code/cost_ledger.jsonl`** still holds the original `baseline-implementation` sub-agent entry: `in=3880, out=24209, cache_create=66455, cache_read=630285, cost_usd=$0.813`. This is the sub-agent call from the OLD subprocess (PID 18994 was spawned before the `ledger_dir` fix landed). Hot-reload only updated the uvicorn process, not the running subprocess.

**The BUG-LR-008 fix (`ledger_dir=project_dir` in `collect_agent_text`) will only take effect in new runs** (new subprocess). This run's repair `implement_baseline` will still write to `code/cost_ledger.jsonl`.

**Phase 0 measurement IS working** — `run_experiment` cost entry appeared in run root `cost_ledger.jsonl` with `cache_creation_input_tokens=11083, input_tokens=6, output_tokens=609`. These are the RLM root tokens for the run_experiment call.

**Dashboard events:** 110 (was 97), 13 new events from run_experiment timeout + repair dispatch.

---

## Live Run Observations — 2026-05-28 ~15:38 CDT (monitoring check #3)

**Status:** `run_experiment` phase in progress (started 20:11:58 UTC, ~27 min elapsed).

**Docker container:** `awesome_kepler` running `pip install torch==2.2.0 torchvision --index-url https://download.pytorch.org/whl/cu118` inside image `6c04cc73377d`. Consumed **2.12GB network / 1.8GB block reads / 2GB block writes** — torch CUDA wheel download nearly complete. CPU at 3.72%, actively working. Not stuck.

**Process status:** PID 18994 (RLM run) still alive. `cost_summary.updated_at` being updated every heartbeat (last: 20:36 UTC). No `experiment_runs.jsonl` yet — will be created when run_experiment resolves.

**Dashboard events:** 97 events, no new events since 20:11:58 UTC. Last event was `iteration_heartbeat` counter=7, note="running experiment - this may take several minutes".

**Known risk:** Container is downloading `torch+cu118` on a Mac with no NVIDIA GPU. Once pip install completes, `train.py` runs in the container. The generated `commands.log` mentions "adaptive: 5 steps on CPU, 150 on GPU" — `train.py` has a CPU fallback path, so training may complete in degraded mode. If `train.py` fails with CUDA error instead of falling back, `run_experiment` will return `success=False` and the RLM root will receive an error result.

**UI check:** Lab page loads, shows the run. Only console error: BUG-LR-004 (leaf-scores 404 during active run). No new bugs observed.

**Next check ETA:** ~15:50 CDT — to see if pip install completed and train.py started.

---

## Monitoring check — 2026-05-28 ~16:40 CDT (post-context-compaction)

**Run status:** `completed` (sandboxMode: `local`). Started 19:49 UTC, completed 21:00 UTC. No active runs.

**Cost ledger (run root):** 0 `subagent_usage` rows — expected. BUG-LR-008 fix requires a NEW run; old subprocess never used the patched code.

**Dashboard events:** 198 total. No `subagent_usage` events (same reason).

**Experiment runs:** 3 rows in `experiment_runs.jsonl`, all `success=False`. UI shows 6 experiments (01–06) — 01/03/05 labelled "RUNNING" (stale status; run is done), 02/04/06 labelled "FAILED". Stale RUNNING labels are a display artifact — the subprocess exited before writing a final status to those rows.

**UI check:**
- ✅ Lab page loads at `/lab?projectId=prj_e2e07ed9fd6cad78`
- ✅ Run shows: 0.00 rubric score, 6/6 areas failing, 27 leaf scores all `degraded_no_metrics`
- ✅ Worker reports: 10 total / 3 failed / 7 completed
- ✅ **0 console errors** — BUG-LR-004 (leaf-scores 404 spam) no longer firing on completed run
- ⚠️  3 error alerts shown in UI (2× `run_experiment`, 1× `record_candidate_outcome`) — pre-existing, from the failed run

**New bugs detected:** None.

**Action needed:** Start a new run with `sandbox=runpod` to validate BUG-LR-008 fix and test the permanent BUG-LR-009/010 fixes.

---

## Root Cause Analysis — SDAR prj_e2e07ed9fd6cad78 final score 0.0% (2026-05-28 post-run)

### Summary

Three independent failures combined to produce 0/6 areas, all `degraded_no_metrics`:

| # | Root cause | Effect |
|---|-----------|--------|
| 1 | Run submitted with `sandbox=local` instead of `sandbox=runpod` | No GPU; Qwen 3B/7B physically unable to load on Mac |
| 2 | BUG-LR-009: generated Dockerfile installed CUDA torch wheel from scratch (~2.5GB, >1800s build) | Attempt 1 `run_experiment` timed out |
| 3 | BUG-LR-010: implement_baseline passes 2 & 3 generated surrogate models; pre-flight AST check blocked them | Attempts 2 & 3 blocked before training |

**The fundamental blocker is #1.** Even with fixes 2 & 3, training 1.7B/3B Qwen models is impossible without a GPU. The run must be re-submitted with `sandbox=runpod`.

### Why RunPod wasn't used

- `.env` has `REPROLAB_DEFAULT_SANDBOX=runpod` ✓ (correct)
- `frontend/src/app/lab/page.tsx:38` reads `process.env.REPROLAB_DEFAULT_SANDBOX` → passes as `serverDefaultSandbox="runpod"` prop ✓
- `lab-shell.tsx:181`: `readUserPrefs().sandbox ?? serverDefaultSandbox ?? "docker"`
- **User's browser localStorage** had `sandbox: "local"` saved from a previous manual selection → that took precedence over `serverDefaultSandbox`

**Fix**: For the next run, use the RunPod sandbox selector in the lab UI to select RunPod before submitting.

---

## Permanent Fixes Applied (2026-05-28 post-run)

### Fix 1: BUG-LR-009 — RunPod Dockerfile uses pre-built pytorch base (permanent)

**Files changed:**
- `backend/agents/environment_detective.py` — `run_offline()` accepts `sandbox_mode` param; `_generate_dockerfile()` accepts `sandbox_mode` — when `"runpod"`, generates `FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-runtime-ubuntu22.04` base with no torch install
- `backend/agents/rlm/primitives.py` — `detect_environment` threads `ctx.sandbox_mode` into `run_offline()` and into the primitive cache key

**Effect:** On all future RunPod runs, the initial Dockerfile will use the pre-built pytorch image. Torch is already installed; no 2.5GB CUDA wheel download. Build time drops from ~30 min (timeout) to ~2 min.

**Tests:** 4 new tests in `tests/test_issue24_environment_detective.py` — all pass.

### Fix 2: BUG-LR-010 — BASELINE_EXTRA_GUIDANCE enforces real Qwen weights (permanent)

**Files changed:**
- `.env` — added `REPROLAB_BASELINE_EXTRA_GUIDANCE` with SDAR-specific instructions:
  - Load `Qwen/Qwen3-1.7B-Instruct` + `Qwen/Qwen2.5-3B-Instruct` via `AutoModelForCausalLM.from_pretrained` (no surrogate)
  - Use `FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-runtime-ubuntu22.04` base in Dockerfile
  - Only install: transformers, accelerate, datasets, trl, peft, sentencepiece, alfworld
  - Skip 7B model — stay within 24GB GPU

**Effect:** `implement_baseline` prompt now explicitly prohibits surrogate models and specifies exact Qwen paths. Pre-flight validator should pass on attempt 1.

### Fix 3: `.env` RunPod image — devel → runtime (minor)

- `REPROLAB_RUNPOD_IMAGE` changed from `devel-ubuntu22.04` to `runtime-ubuntu22.04`
- The `devel` variant includes NVCC + full CUDA SDK (~18GB pod pull); SDAR never compiles CUDA kernels
- Saves 5–10 min pod provisioning time and ~$0.50–1.50 per run

---

## Next run checklist

Before starting the next SDAR reproduction:

- [ ] In the lab UI, verify the **sandbox selector shows "RunPod"** (not "local" or "docker") before clicking Submit
- [ ] Confirm `.env` `REPROLAB_BASELINE_EXTRA_GUIDANCE` is set (it is after this session's fix)
- [ ] Confirm `.env` `REPROLAB_RUNPOD_IMAGE` is `runtime` not `devel` (fixed)
- [ ] Confirm `REPROLAB_DEFAULT_SANDBOX=runpod` in `.env` (already correct)
- [ ] Watch for `detect_environment` in the run artifacts — Dockerfile should start with `FROM runpod/pytorch:...`
- [ ] Watch for `implement_baseline` pre-flight PASS on attempt 1 (no more surrogate model blocks)
- [ ] Watch `cost_ledger.jsonl` in run ROOT (not `code/`) for `subagent_usage` rows (BUG-LR-008 fix validation)
