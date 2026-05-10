# learn.md — bugs we shipped and what we changed so we don't ship them again

This is a runbook of post-mortem entries for production-shaped bugs in the
OpenResearch agent stack. Each entry is short and follows the same shape:

> **Symptom → Root cause → Fix → Lesson → Guardrail (test or pattern)**

Add a new entry to the top of the list. Keep entries surgical: one bug per
section, no broad essays. If a class of bug recurs, escalate it to a section
in **Cross-cutting principles** below.

---

## 2026-05-10 — Hermes Agent oversight silently no-oped on every run

**Symptom.** `hermes_step_reports` and `hermes_checkpoint_reports` in pipeline
state always showed `status=unavailable` with `summary="Nous Hermes runtime
unavailable"`.  The oversight layer was integrated into the orchestrator but
never actually audited anything.

**Root cause.** Two compounding issues:

1. `NousHermesClient._run_agent()` called `importlib.import_module("run_agent")`
   to load the Nous Hermes Agent runtime, but the `hermes-agent` package was
   never installed.  Every call raised `ModuleNotFoundError`.
2. The constructor hardcoded `model="anthropic/claude-sonnet-4"` without
   passing `api_key` or `provider` to `AIAgent`.  Even after installing the
   package, Hermes Agent's provider resolver could not find credentials because
   `ANTHROPIC_API_KEY` was empty in `.env` — only `OPENAI_API_KEY` was set.

The `audit()` method caught all exceptions and returned an `unavailable`
report, so the pipeline never crashed — but oversight was entirely dead.

**Fix.**

1. Installed `hermes-agent` (`pip install git+https://github.com/NousResearch/hermes-agent.git`).
2. Rewrote `NousHermesClient` (`backend/hermes_audit/client.py`) with:
   - `_resolve_hermes_config()` — auto-detects available API keys
     (`ANTHROPIC_API_KEY` preferred, `OPENAI_API_KEY` fallback) and returns
     the correct `(model, api_key, provider)` triple.
   - Explicit `api_key=` and `provider=` passed to `AIAgent()` so Hermes
     doesn't rely on its own config wizard / env-var discovery.
   - **Fallback chain:** Hermes Agent → Claude Code SDK (`claude_agent_sdk.query()`)
     → unavailable report.  The Claude SDK is already installed for the main
     pipeline, so it serves as a zero-config fallback.

**Lesson.** **A graceful degradation path that is always active is
indistinguishable from a missing feature.**  The original code's
`try/except → unavailable` was correct for resilience, but without any
logging, alerting, or test that asserts the *happy* path works, the feature
shipped dead.  When you add a `try/except → soft fallback`, always pair it
with:
- A log line at WARNING level so the fallback is visible in stderr
- A test that exercises the primary path with a mock
- A test that exercises the fallback path with the primary disabled

**Guardrail.**
- `tests/test_hermes_audit_service.py::test_client_uses_hermes_agent_when_available`
  asserts the primary Hermes Agent path produces a valid report.
- `tests/test_hermes_audit_service.py::test_client_falls_back_to_claude_sdk_when_hermes_unavailable`
  asserts the Claude SDK fallback activates when Hermes fails.
- `tests/test_hermes_audit_service.py::test_client_returns_unavailable_when_both_backends_fail`
  asserts the final unavailable fallback with error details.
- `tests/test_hermes_audit_service.py::test_client_resolve_config_prefers_anthropic_key`
  and `test_client_resolve_config_falls_back_to_openai_key` lock in the
  credential resolution order.

---

## 2026-05-09 — `database disk image is malformed` on `reprolab.db`

**Symptom.** `reprolab reproduce …` boot crashes:
```
File "backend/eventstore/sqlite_store.py", line 132
    boot = _new_connection(self._path)
sqlite3.DatabaseError: database disk image is malformed
```
`sqlite3 reprolab.db "PRAGMA integrity_check"` confirms the file is corrupt.

**Root cause.** `SqliteEventStore` ran in WAL mode with the SQLite default
`synchronous=NORMAL`. NORMAL is fast — it doesn't fsync the WAL on every
commit — but if the writer process is `SIGKILL`'d at exactly the wrong
moment between a WAL write and the next checkpoint, the main DB file can
be left referring to pages that the WAL never committed. The most recent
killed `backend.cli reproduce` subprocess (mid-pipeline crash on the IPv6
URL bug) was the proximate trigger.

**Fix.** `backend/eventstore/sqlite_store.py:_new_connection` now sets
`PRAGMA synchronous=FULL`. The throughput cost is negligible at our write
rate (≈ a few hundred events per pipeline run); the durability win is the
whole point of an event store. The corrupt DB was quarantined to
`reprolab.db.corrupt-<timestamp>` and the offline backup restored.

**Lesson.** **Default SQLite settings are tuned for read-heavy app caches,
not for event stores.** Any code path where a SIGKILL'd process must leave
the DB in a recoverable state needs `synchronous=FULL` (or at minimum
`synchronous=NORMAL` with explicit `PRAGMA wal_checkpoint(TRUNCATE)` after
each commit batch). NORMAL + WAL is a fine combination for a process you
control the lifecycle of, but pipelines crash and dev servers get
`Ctrl+C`'d — assume the worst.

**Guardrail.**
- Inline comment in `_new_connection` cites this entry.
- `learn.md` cross-cutting principle #9 (added below) generalises the
  "configure for the failure mode you actually have" rule to any local
  store.

---

## 2026-05-09 — Per-agent budget caps must be elegant, not silent

**Symptom.** Two related complaints from the same root cause:
1. Agents would silently fail at turn 16 with the SDK's opaque
   `"Reached maximum number of turns (15)"` exception bubbling out — no
   structured signal, no partial-output preservation, no remediation
   hint. The lab UI just showed the run as `failed` with no actionable
   detail.
2. With turn caps removed entirely, runaway agents (infinite tool-call
   loops, model-side hallucinated retries) had no stop condition other
   than killing the dev server.

**Root cause.** The original implementation conflated two concerns:
"how do we bound a misbehaving agent" and "how do we surface that
boundary being hit". The fix-by-removal made the second worse; the
fix-by-numerical-cap made the first worse.

**Fix.** Three independent governors per agent invocation, each with a
typed exception:

| Governor | Efficient | Max | Enforced by |
|---|---|---|---|
| `max_turns_per_agent` | 30 (60 heavy) | None | SDK `--max-turns` flag |
| `max_tool_calls_per_agent` | 80 | None | orchestrator counter |
| `agent_wall_clock_seconds` | 1200 (20 min) | 3600 (1 hr) | `asyncio.timeout` wrapping `runtime.run_agent` |

All three raise the same typed exception:
```python
class AgentLimitExceeded(RuntimeError):
    agent_id: str
    kind: Literal["turns", "tool_calls", "wall_clock"]
    limit_value: int
    elapsed_seconds: float
    partial_output: str   # preserved for retry / logging / display
```

The orchestrator additionally **converts the SDK's untyped
`Reached maximum number of turns (N)` exception** into the same
`AgentLimitExceeded(kind="turns")` via a regex match, so callers
never have to string-match exception text. The frontend timeline panel
+ `agent_telemetry.jsonl` already render `error_message`, so partial
output and the kind/value of the limit hit surface in the UI for free.

**Lesson.** **Bounded resources are a product surface, not an
implementation detail.** When a budget cap fires, the system must:
1. Preserve partial work (don't blow away the `collected_text` buffer)
2. Tell the operator *which* budget fired and *what value* it was at
3. Suggest remediation (`--execution-mode max` raises all caps)
4. Be programmatically inspectable so retry / fallback logic can branch
   on `kind`, not on string-matched English

**Guardrail.**
- `tests/test_execution_modes.py::test_execution_profile_efficient_caps_at_30_turns_and_80_tool_calls`
  locks in the numerical contract.
- `tests/test_agent_runtime_orchestrator.py::test_orchestrator_converts_sdk_turn_cap_message_to_typed_exception`
  asserts the SDK-error → typed-exception conversion path.
- `tests/test_agent_runtime_orchestrator.py::test_orchestrator_uses_efficient_default_caps_for_heavy_agents`
  asserts that heavy agents see the heavy-agent caps end-to-end.

---

## 2026-05-09 — `Reached maximum number of turns (15)` aborts every real run

**Symptom.** Frontend lab page shows the run failing in `paper_understood`.
Stderr trace ends with:
```
Exception: Claude Code returned an error result:
  Reached maximum number of turns (15)
```
Even though commit `42aa8f5 fix: remove agent turn caps` had previously
removed the caps, real runs (e.g. `paperbench1.pdf`) still aborted at turn 16.

**Root cause.** `backend/agents/execution.py::ExecutionProfile.from_mode`
silently re-introduced `max_turns_per_agent=15` (efficient) / `25` (max) and
added a `max_tool_calls_per_agent=250` cap. The values were carried through:
`ExecutionProfile → orchestrator.max_turns_per_agent → AgentRuntimeSpec.max_turns →
ClaudeAgentOptions.max_turns → claude CLI --max-turns 15`. The Claude CLI then
threw the SDK exception at turn 16. None of the existing tests caught the
regression because they had been **updated** to assert the cap rather than the
absence of one.

**Fix.** `backend/agents/execution.py:69-102` — both `efficient` and `max`
profiles now set `max_turns_per_agent=None`, `heavy_agent_max_turns=None`,
`max_tool_calls_per_agent=None`. The orchestrator continues to forward
`max_turns=None` through the SDK, so neither it nor the CLI imposes a cap.
Bounding is delegated to:
- `command_timeout_seconds` (per shell command, currently 1 h / 2 h)
- The agent's submit-when-done contract (system prompts instruct the agent
  to call the submit tool when finished)

**Lesson.** **A removed limit is a contract.** If you decide a cap should not
exist, the test must assert `is None`, not `== <new_higher_value>`. Otherwise
the next refactor will silently re-introduce a cap that survives review
because the test still passes against the new number. We had a regression
because tests said "30 is the cap for heavy agents" — true at one point, but
the right invariant was "no cap".

**Guardrail.**
- `tests/test_execution_modes.py::test_execution_profile_efficient_does_not_cap_agent_turns`
  asserts `max_turns_per_agent is None` for both modes. The test docstring
  explicitly cites the bug so a future engineer raising the cap reads why.
- `tests/test_agent_runtime_orchestrator.py::test_orchestrator_does_not_cap_heavy_agents_by_default`
  asserts the propagation: orchestrator → AgentRuntimeSpec → SDK call.

---

## 2026-05-09 — `ValueError: Invalid IPv6 URL` crashes the runtime guard on bracketed agent text

**Symptom.** `paper_understood` agent died after ~60 s with:
```
File "backend/agents/runtime/base.py", line 154, in _normalize_guard_text
    parsed = urlparse("https://" + text if "://" not in text else text)
ValueError: Invalid IPv6 URL
```
Stack trace originated in `RuntimeGuard.find_blocked_term`, called by
`claude_runtime.run_agent` on every assistant text block.

**Root cause.** `_normalize_guard_text` was applied to **arbitrary agent
output** (the `text` of the assistant's narration, not a URL). Python 3.12
tightened `urllib.parse.urlsplit` to validate bracketed netlocs as IPv6
literals; any text containing `[...]` (including narration like *"build the
comprehensive PaperClaimMap [for FTRL]"*) caused `urlparse` to raise. The
exception bubbled up out of the SDK transport and aborted the agent loop.

**Fix.** `backend/agents/runtime/base.py:151-178` — split the function:
- `_canonicalize_url_term(value)` — used **only** on configured blocked
  terms (which are documented to be URL-like). Wraps `urlparse` in
  `try/except ValueError`; falls back to the lowercased input when parsing
  fails so a malformed blocked term still substring-matches.
- `find_blocked_term(text)` — now does lowercase substring matching against
  the canonicalised terms. **Never URL-parses arbitrary text.**

Same defensive try/except added at
`backend/services/ingestion/discovery/adapters/regex.py:46-53`, the second
site that fed `urlparse` text it did not control (regex-extracted URLs in
paper text).

**Lesson.** **Never URL-parse data you do not control.** Validators that
were safe on Python 3.11 became hazardous on 3.12+ because the standard
library's permissiveness changed. Any layer that calls `urlparse`,
`urlsplit`, or any other strict parser on adversarial / model-generated
strings must wrap it in `try/except ValueError`. The principle is broader:
**parse only at boundaries, not inside hot paths**, and treat input from
LLMs the same way you'd treat input from the network — possibly malformed,
always handled defensively.

**Guardrail.**
- `tests/test_runtime_guard.py::test_runtime_guard_handles_arbitrary_text_with_brackets`
  feeds the guard five flavours of bracketed text that previously crashed
  `urlparse` (including `[::1]:8080`, `[]`, `https://[malformed`, etc.).
- `tests/test_runtime_guard.py::test_runtime_guard_normalizes_blocked_term_with_brackets`
  asserts that even a malformed configured term (e.g.
  `github.com/foo[bar`) does not crash term normalisation.
- `tests/test_issue14_artifact_discovery.py::test_regex_adapter_skips_malformed_url_without_crashing`
  covers the second urlparse site.

---

## Cross-cutting principles (May 2026)

These are the practices we follow because we've now been bitten by violating
them. Read this section before adding a new agent, runtime, or boundary.

### 1. Type the cap, not the value.

If a constraint should be opt-in (e.g. "no turn cap by default"), encode that
in the type system: `int | None` with default `None`, **not** `int = 999`.
Tests must assert the type-level invariant (`is None`), not a placeholder
value, otherwise a refactor that swaps `999` for `42` will pass review.

### 2. Parse at boundaries, never in hot paths on adversarial input.

URL/JSON/YAML/regex/etc. parsers raise on hostile input. The boundary where
the parser runs determines the blast radius. Apply this rule:

| Caller | Input source | Parser? |
|---|---|---|
| Intake | User-uploaded PDF / arXiv ID | yes — wrap in try/except, surface a typed error |
| Config loader | `.env` / `config.yaml` | yes — fail loud at import time |
| Runtime guard / agent middleware | LLM output, agent narration | **no** — substring match or fall back to a permissive heuristic |
| Discovery / link extraction | Paper body text | yes — wrap in try/except, **skip** the bad match, do not crash |

### 3. Default to `None`, not to a magic number.

Every cap (`max_turns`, `max_tool_calls`, `command_timeout_seconds`,
`sandbox_memory_limit`, …) should default to `None` unless a concrete bound
is genuinely required for safety. When a bound is required, write the
constant once in `execution.py` and reference it everywhere — never inline
the literal.

### 4. Failures must be observable from the lab UI.

Backend exceptions used to disappear into `runner.stderr.log`. The frontend
now exposes:

- `ProgressStrip` — current stage, elapsed time, **stall warning** if no
  activity for ≥ 90 s (`frontend/src/lib/demo/progress.ts::STALL_THRESHOLD_SECONDS`)
- `TimelinePanel` — per-agent invocation card with success/failure dot,
  duration, error message
- `Copy debug bundle` button — `GET /api/lab/debug-bundle?projectId=...`
  returns a compact JSON (status, last 24 KB stderr, last 30 telemetry
  records, pipeline state preview, latest error) for paste-into-Claude-Code
  triage

If you add a new failure mode, make sure one of these surfaces shows it. A
silent failure is a missing UI element, not a missing log line.

### 5. Regression tests cite the bug.

Every fix in this file is locked in by a test whose **docstring** names the
symptom. The next engineer who tries to revert the fix should read why it
exists from the test alone. Convention:

```python
def test_execution_profile_efficient_does_not_cap_agent_turns() -> None:
    """Regression: capping max_turns_per_agent caused the SDK to abort
    runs at turn 16 with 'Reached maximum number of turns (15)'. ..."""
```

### 6. Don't trust auto-generated `.gitignore` exclusions.

Build artifacts (`.next/`, `tsconfig.tsbuildinfo`, `_test_logs/`, local DB
backups, sample PDFs) have repeatedly crept into `git status`. Run
`git status --porcelain | grep -E '^\?\?'` before each commit and add
patterns to `.gitignore` as you discover them. Do this **once per noise**,
not once per commit.

### 7. Hardware-conditional code paths must not be the only path.

PaperBench's SAPG paper requires a GPU for Isaac Gym. Our system has no GPU.
We deliberately built the PaperBench integration so that:

- `dry` mode validates the bundle and submission shape with no LLM call
- `--with-pipeline` mode runs the agent stack
- Code-Development rubric nodes (≈ 60 % of weight) are scored from source
  files alone, no execution required

Whenever you add a feature whose happy path needs hardware we don't have,
add a `dry` / `simulate` mode at the same time so CI and local dev can
exercise it. If the feature only works on prod hardware, it doesn't really
work.

### 9. Configure local stores for the failure mode you actually have.

A long-running pipeline can be killed at any instant — the OS killing it
for memory, the developer hitting Ctrl+C, an upstream crash leaving a
subprocess orphaned. Any local data store needs settings tuned for *that*
failure mode, not for the abstract "well-behaved process" case. For SQLite
that means `synchronous=FULL` in WAL mode, despite the small write
throughput hit. For file-backed JSON status (`runs/<project>/status.json`)
that means atomic write-and-rename, not in-place mutation. For any cache,
a `try/except` around the read with a one-shot rebuild path.

If your local store can't survive a `kill -9`, treat it the same as you'd
treat ephemeral memory and persist the source of truth elsewhere.

### 10. A removed cap is a contract — test for `is None`, not for the new number.

See learn.md 2026-05-09 ("Reached maximum number of turns") and the
follow-up "Per-agent budget caps must be elegant". When you decide a
constraint shouldn't exist, the test must assert the type-level
invariant (`assert max_turns is None`), not a placeholder value
(`assert max_turns == 999`). Otherwise the next refactor will silently
re-introduce a cap that survives review because the test still passes
against the new number.

### 11. A silent fallback needs a loud test.

When you write `try/except → return degraded_result`, you are creating a
feature that can ship dead without anyone noticing.  Pair every graceful
degradation path with: (1) a WARNING-level log so operators see it in
stderr, (2) a test that asserts the *primary* path works with a mock, and
(3) a test that asserts the *fallback* path activates when the primary is
broken.  If you only test the fallback, you'll never know the primary was
never invoked.  See learn.md 2026-05-10 (Hermes Agent no-op).

### 8. Auto-reload is your friend AND your enemy.

`uvicorn --reload` and `next dev` Turbopack both watch the working tree.
**Branch operations (`git switch -c <new-branch>` from HEAD) are safe —
tracked file contents don't change.** Operations that mutate tracked files
(`git checkout <other-branch>`, `git pull`, `git reset --hard`) will trigger
auto-reload storms in both processes and **will** kill an in-flight pipeline
run. Plan merges accordingly.

---

## Editing this file

- Add new entries at the **top** of the dated section.
- Keep each entry under ~250 words.
- Always include a regression test path.
- If a principle is violated more than twice, promote it from a per-bug
  lesson to a numbered item under **Cross-cutting principles**.
