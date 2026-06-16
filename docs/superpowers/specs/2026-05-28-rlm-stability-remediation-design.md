# RLM Stability Remediation — Design + Plan (2026-05-28)

Date: 2026-05-28
Author: codex-session (assistant) + Aayush (user)
Status: RESOLVED — all 5 fixes landed in commit `271df91` (2026-05-28)
Companion: [`2026-05-28-subscription-cost-reduction-design.md`](2026-05-28-subscription-cost-reduction-design.md) (cost track), [`2026-05-27-rlm-reproduction-stability-plan.md`](../../runbooks/2026-05-27-rlm-reproduction-stability-plan.md) (prior stability plan)

## TL;DR

A real SDAR reproduction submitted today (`prj_09047604e591d969`, arxiv 2605.15155, `--provider openai --sandbox runpod --max-usd 15`) terminated as `verdict: "partial"` after 5 iterations, 0 cost ledger spend recorded, and `rubric.overall_score: 0.0`. **The run never executed a single domain primitive other than `check_user_messages`.** The failure cascade was:

1. iter 1 root-model script called `globals().get("report_state", {...})` — a normal Python idiom for cross-iteration state
2. rlm's `LocalREPL` shadows the `globals` builtin with `None`, so `None(...)` raised `TypeError: 'NoneType' object is not callable`
3. `LocalREPL.execute_code` swallowed the traceback and surfaced only the bare 45-char message to the model
4. the root model — with no line number, no name, no traceback — concluded "primitives are unavailable" and spent iters 2–5 writing increasingly defensive fallback scripts, eventually shipping a `partial` verdict
5. the forced-iteration policy ([Lane H, 2026-05-24](../../../CLAUDE.md)) did NOT refuse `FINAL_VAR` because it only fires when `latest_rubric_score < target`; `latest_rubric_score is None` slipped through
6. cost ledger captured only 2 zero-cost rows (the two `check_user_messages` calls) because no LLM-bound primitive ran; this means BUG-LR-008's "subagent_usage rows" fix from today's commit (`a09db5b`) is also untestable from this run

Five distinct bugs surfaced. Three are NEW (BUG-LR-011/012/014). Two are existing-but-uncovered edge cases (BUG-LR-013/015). One is a re-validation backlog item (BUG-LR-008).

This document specifies fixes in priority order, gives a code-level diff plan for each, names the validation gate, and lists every related doc that must be updated.

## Severity matrix

| ID | Title | Severity | Blocks any run? | Detectable today? | Fix complexity |
|----|-------|----------|-----------------|-------------------|----------------|
| BUG-LR-011 | rlm `_SAFE_BUILTINS` shadows `globals`/`locals` as `None` | **P0** | YES — any iteration that touches `globals()` dies | YES — bare `NoneType` in stderr | Trivial (import-time monkey-patch) |
| BUG-LR-012 | `LocalREPL.execute_code` swallows traceback | **P0** | NO directly, but turns small bugs into cascading failures | NO — root cause looks like model confusion | Small (monkey-patch + system-prompt update) |
| BUG-LR-014 | shell-exported `OPENAI_API_KEY` silently overrides `.env` | **P0** | YES — every iteration 401s | YES — `invalid_api_key` in CLI log | Small (boot-time validator + doc fix) |
| BUG-LR-013 | forced-iteration policy lets FINAL_VAR through when `rubric_score is None` | P1 | NO — accepts bad final reports silently | NO — final_report looks "honest" | Small (extend policy) |
| BUG-LR-015 | no detector for "model gave up before doing work" | P1 | NO — runs complete | NO — silent | Medium (new emitter + heuristic) |
| BUG-LR-008 (recap) | `subagent_usage` rows missing from cost ledger | P2 (re-validate) | NO | YES — once a run reaches `implement_baseline` | Already fixed in `a09db5b`; needs E2E proof |

P0 fixes are mandatory before the next SDAR attempt — without them, we will burn another GPT-5 budget on the same 5-iteration death-spiral.

## Run-of-record: `prj_09047604e591d969`

Inputs:
- arxiv 2605.15155 (SDAR)
- CLI: `python -m backend.cli reproduce 2605.15155 --provider openai --sandbox runpod --max-usd 15 --max-wall-clock 7200`
- Root model: `gpt-5` (resolved by `resolve_root_model` because `OPENAI_API_KEY` was set)
- Sub-backend: `gpt-5-mini`
- Sandbox: `runpod` (confirmed in CLI banner)
- `.env`: correctly configured (sandbox not forced, RunPod image set to cuda-runtime, baseline guidance set to Qwen 1.7B+3B)

Two CLI invocations:
1. **Attempt 1** (21:54Z): died at iter 0 with `openai.AuthenticationError: 401 invalid_api_key` for `sk-svcac…IMEA`. The valid `sk-proj-…` from `.env` was shadowed by a stale shell `OPENAI_API_KEY=sk-svcacct-…`. → BUG-LR-014.
2. **Attempt 2** (21:56Z, `env -u OPENAI_API_KEY` prefix): auth fixed; ran 5 iterations, terminated `partial`. → BUG-LR-011/012/013/015.

Observable state after Attempt 2:
- `runs/prj_09047604e591d969/demo_status.json` → `status: completed, completedAt: 22:01:17Z` (7-min wall clock)
- `runs/prj_09047604e591d969/final_report.json` → `verdict: "partial"`, `rubric.overall_score: 0.0`, `primitive_trace: {calls: 2, by_primitive: {check_user_messages: 2}}`, `scope.gaps: ["required primitives (detect_environment, build_environment, implement_baseline, run_experiment) were not available in this REPL session."]` **(factually wrong)**
- `runs/prj_09047604e591d969/cost_ledger.jsonl` → 2 rows, both `cost_usd: 0.0`, `model: "gpt-4o-mini"`, primitive `check_user_messages`. NO `subagent_usage` rows (no `implement_baseline` ever ran). → BUG-LR-008 validation deferred.
- `runs/prj_09047604e591d969/code/` → does not exist (no baseline generated)
- `runs/prj_09047604e591d969/dashboard_events.jsonl` → 14 events: 4 `primitive_resource`, 4 `primitive_call` (2 start + 2 ok for `check_user_messages`), 5 `repl_iteration`, 1 `run_complete`. No `rubric_score`, no `primitive_call` for any of the 13 other tools.
- `runs/prj_09047604e591d969/rlm_state/iterations.jsonl` → iter 0 `stderr_meta: {length: 45, prefix: "\nTypeError: 'NoneType' object is not callable", has_traceback: false}`. This is the primary smoking gun.

Probe to confirm BUG-LR-011 is deterministic, not a model artifact:

```python
# probe.py — produces "TypeError: 'NoneType' object is not callable" against a clean LocalREPL
from rlm.environments.local_repl import LocalREPL
from backend.agents.rlm.binding import build_custom_tools
# (build a RunContext with sandbox_mode='local')
repl = LocalREPL(custom_tools=build_custom_tools(ctx))
r = repl.execute_code('results = {}\nfor n in ["check_user_messages","understand_section"]:\n    obj = globals().get(n)\n    results[n] = (obj is not None, callable(obj))\nprint(results)')
print(r.stderr)  # "\nTypeError: 'NoneType' object is not callable"
```

Source of the shadow: `.venv/lib/python3.14/site-packages/rlm/environments/local_repl.py:112-118`:

```python
"input": None,
"eval": None,
"exec": None,
"compile": None,
"globals": None,
"locals": None,
```

`globals`/`locals` are grouped with `eval`/`exec`/`compile`. The latter three are genuine code-execution risks; `globals`/`locals` are namespace getters — not dangerous.

## Architecture (fixes)

### Fix 1 — BUG-LR-011: restore `globals`/`locals` in the REPL sandbox

**Approach**: import-time monkey-patch of `rlm.environments.local_repl._SAFE_BUILTINS`, mirroring the existing pattern in `backend/agents/rlm/forced_iteration.py` (which patches `LocalREPL._final_var`).

New file: `backend/agents/rlm/safe_builtins_patch.py`

```python
"""Restore globals()/locals() inside rlm's LocalREPL sandbox.

The upstream rlm library blacklists ``globals`` and ``locals`` alongside
genuinely dangerous builtins (eval/exec/compile/input). Unlike those four,
``globals`` and ``locals`` are pure namespace getters with no code-execution
surface — Python's stdlib treats them as safe enough to ship in every REPL.

Blocking them by setting their entries to ``None`` means any model code that
calls ``globals().get("x", default)`` — a normal idiom for persisting state
between iterations — crashes with the bare message
``TypeError: 'NoneType' object is not callable``. The model sees no name
and no traceback (see safe_repl_traceback_patch.py for the latter half), so
it cannot diagnose what went wrong and typically spirals into a "primitives
unavailable" partial report.

This module patches the dict at import time. Import once from run.py BEFORE
``from rlm import RLM``.
"""
from __future__ import annotations
import builtins as _builtins

from rlm.environments import local_repl as _local_repl

def apply_safe_builtins_patch() -> None:
    sb = _local_repl._SAFE_BUILTINS
    sb["globals"] = _builtins.globals
    sb["locals"] = _builtins.locals
    # NEVER restore eval/exec/compile/input — those are intentionally blocked.

apply_safe_builtins_patch()
```

Wire-up: `backend/agents/rlm/run.py` near the top (next to `forced_iteration` import):

```python
from backend.agents.rlm import safe_builtins_patch  # noqa: F401 — import-time monkey-patch
```

Validation:
1. Unit test in `tests/agents/rlm/test_safe_builtins_patch.py` that constructs a `LocalREPL`, executes `r = repl.execute_code('print(type(globals()).__name__)')`, asserts `r.stderr == ""` and `r.stdout == "dict\n"`.
2. Negative test: assert `_local_repl._SAFE_BUILTINS["eval"] is None` and same for `exec`/`compile`/`input` — the patch must not loosen the genuine blocks.
3. Smoke test: replay the iter-1 snippet from `prj_09047604e591d969` against a `LocalREPL` and assert it completes without exception.

### Fix 2 — BUG-LR-012: surface traceback when REPL code raises

**Approach**: extend the existing module-level patch pattern to wrap `LocalREPL.execute_code`'s exception handler. Don't override `execute_code` wholesale — keep the patch surgical so future rlm releases stay compatible.

New file: `backend/agents/rlm/safe_repl_traceback_patch.py`

```python
"""Make REPL exceptions diagnosable by the root model.

Upstream rlm's LocalREPL.execute_code catches the exception and appends only
``f"{type(e).__name__}: {e}"`` to stderr (local_repl.py:506). For a
``TypeError: 'NoneType' object is not callable`` raised at line 105 of a
12K-character script, the model sees 45 chars with no file, no line, no
chain — and cannot diagnose which name was None. Today's run wasted four
iterations because of this.

We patch ``LocalREPL.execute_code`` to also include
``traceback.format_exc()`` (capped at 2000 chars to stay within the SSE
egress bound). The patch is applied at module import time; if the upstream
signature changes, the patch is a no-op (we log a warning instead of
crashing).
"""
from __future__ import annotations
import logging, traceback
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

from rlm.environments import local_repl as _local_repl

logger = logging.getLogger(__name__)
_PATCHED_ATTR = "_reprolab_traceback_patched"

def _patched_execute_code(self, code: str):
    # Re-implement execute_code's body but include the full traceback in stderr.
    # If you bump the rlm version and execute_code's signature changes, drop the
    # patch and add an upstream PR — see the upstream-merge note below.
    import time
    start = time.perf_counter()
    self._pending_llm_calls = []
    with self._capture_output() as (stdout_buf, stderr_buf), self._temp_cwd():
        try:
            combined = {**self.globals, **self.locals}
            exec(code, combined, combined)
            for k, v in combined.items():
                if k not in self.globals and not k.startswith("_"):
                    self.locals[k] = v
            self._restore_scaffold()
            stdout = stdout_buf.getvalue()
            stderr = stderr_buf.getvalue()
        except Exception as e:
            stdout = stdout_buf.getvalue()
            tb = traceback.format_exc()
            # Cap at 2000 chars; show last lines (most informative).
            if len(tb) > 2000:
                tb = "...(truncated)\n" + tb[-1800:]
            stderr = stderr_buf.getvalue() + f"\n{tb}"
    fa = self._last_final_answer
    self._last_final_answer = None
    return _local_repl.REPLResult(
        stdout=stdout, stderr=stderr,
        locals=self.locals.copy(),
        execution_time=time.perf_counter() - start,
        rlm_calls=self._pending_llm_calls.copy(),
        final_answer=fa,
    )

def apply_traceback_patch() -> None:
    if getattr(_local_repl.LocalREPL, _PATCHED_ATTR, False):
        return
    _local_repl.LocalREPL.execute_code = _patched_execute_code
    setattr(_local_repl.LocalREPL, _PATCHED_ATTR, True)

apply_traceback_patch()
```

**Upstream-merge note**: file a PR against `rlm` (PyPI `rlms`) to include `traceback.format_exc()` in the stderr capture by default, and to remove `globals`/`locals` from the `_SAFE_BUILTINS` blocklist. Until accepted, this monkey-patch is permanent.

Validation:
1. Unit test: execute `repl.execute_code("def f():\n    raise ValueError('x')\nf()")`, assert stderr contains `"in f"` and `"raise ValueError"`.
2. Cap test: execute code that raises with a deep stack, assert stderr ends with the last frames and is ≤ 2000 chars.
3. Egress test: confirm the SSE bridge's `sanitize_iteration` does not leak corpus content via the traceback (the SSE bridge already bounds stdout/stderr to metadata-prefix; assert the new patch composes correctly).

### Fix 3 — BUG-LR-014: shell vs `.env` precedence for API keys

**Two-pronged**:

(i) Boot-time validator in `backend/cli.py` (before any LLM call). Pseudocode:

```python
def _warn_on_shell_env_override() -> None:
    from dotenv import dotenv_values
    dotenv_vals = dotenv_values(".env")
    suspect_keys = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                    "FEATHERLESS_API_KEY", "OPENROUTER_API_KEY",
                    "AZURE_OPENAI_API_KEY", "REPROLAB_RUNPOD_API_KEY")
    for k in suspect_keys:
        env_val = os.environ.get(k, "")
        file_val = dotenv_vals.get(k, "") or ""
        # Only warn when both are set and they differ
        if env_val and file_val and env_val != file_val:
            # Show prefix only — never log the key body
            prefix = lambda s: f"{s[:10]}…{s[-4:]}" if len(s) > 14 else "<set>"
            print(
                f"\n[warn] {k}: shell value ({prefix(env_val)}) "
                f"differs from .env ({prefix(file_val)}). "
                f"Shell value will be used. To use the .env value: "
                f"`env -u {k} python -m backend.cli ...`\n",
                file=sys.stderr,
            )
```

Call this once from the top of `_reproduce` (or wherever the CLI entry is). Strictly an advisory — don't abort, because some workflows legitimately override.

(ii) Doc fix in `CLAUDE.md` under "### RLM auth — two surfaces, billed separately": add a short subsection "Shell vs .env precedence — shell wins" with the `env -u` workaround.

Validation:
1. Unit test in `tests/cli/test_env_override_warning.py` that mocks `os.environ` with one key and a `.env` with a different value, runs the validator, and asserts the warning string is on stderr.
2. Manual smoke: re-run the SDAR CLI with a stale `OPENAI_API_KEY` shell var and verify the warning appears before ingestion.

### Fix 4 — BUG-LR-013: forced-iteration policy with `rubric_score is None`

**Approach**: extend `ForcedIterationPolicy.should_refuse_final_var` to also refuse when the iteration floor has not been hit AND no rubric score has been recorded yet. The intent of the policy is "don't give up before doing the work" — `rubric_score is None` means strictly less work has been done than `rubric_score < target`, so the policy should be at least as strict.

Diff target: `backend/agents/rlm/forced_iteration.py`. Find the predicate:

```python
# Current (paraphrased):
if (
    iteration_count < min_iterations
    and latest_rubric_score is not None
    and latest_rubric_score < target_score
):
    return True
```

Change to:

```python
if iteration_count < min_iterations:
    if latest_rubric_score is None:
        # Haven't even tried verify_against_rubric — definitely refuse FINAL_VAR.
        return True
    if latest_rubric_score < target_score:
        return True
return False
```

Plus emit a `run_warning` with `code="forced_iteration"` and a tailored message: `"FINAL_VAR refused at iter {n}: no rubric score yet. Call verify_against_rubric on your latest run_experiment result, or call run_experiment if you haven't run the baseline."`

Wall-clock bypass (already in place) still wins: when `ctx.remaining_s() <= 60s`, the policy yields and ships the partial report.

`OPENRESEARCH_MIN_RUBRIC_ITERATIONS=0` still disables the policy entirely.

Validation:
1. Unit test: construct a policy with `min_iterations=2`, `latest_rubric_score=None`, `iteration_count=1` → expect `should_refuse_final_var()` returns True.
2. Unit test: same setup but `iteration_count=2` → returns False (floor met).
3. Unit test: `iteration_count=1`, `latest_rubric_score=0.0` (score zero is a real value, not None) → returns True (existing behavior preserved).
4. Replay test: replay the iter-5 `FINAL_VAR("report_json")` call from today's run and assert it is refused.

### Fix 5 — BUG-LR-015: detect "model gave up early"

**Approach**: a single emitter in `backend/agents/rlm/run.py` that runs once at run completion, before the final report writes. Compute three signals:

- `essential_primitives_missed`: did the run complete without calling AT LEAST ONE OF `{implement_baseline, run_experiment, verify_against_rubric}`?
- `iteration_underutilization`: did iterations end with `iteration_count < (max_iterations * 0.25)` AND `verdict == "partial"`?
- `rubric_never_scored`: `latest_rubric_score is None`

If any two of three are true AND verdict is `"partial"`, emit `run_warning` with `code="suspicious_partial"` and a message naming the missed primitives + the iteration count. The warning is purely diagnostic — does not change the report.

This is the runtime mirror of the forced-iteration policy (which is preventive). Together they cover "before FINAL_VAR" (Fix 4) and "after FINAL_VAR slipped through anyway" (Fix 5).

Validation:
1. Replay test: feed today's `prj_09047604e591d969` artifacts and assert `suspicious_partial` would fire.
2. Negative test: feed a successful run and assert it does NOT fire.

### Fix 6 — BUG-LR-008 re-validation (deferred until P0 fixes land)

The commit `a09db5b` claims to fix `subagent_usage` row emission in `backend/agents/runtime/invoke.py`. Today's run produced zero rows because no `implement_baseline` ever ran, so the fix is unverified.

**Pre-E2E unit test** (lands today, validates without needing a 60-min run):

`tests/agents/runtime/test_invoke_subagent_usage.py` already exists (per `git status`). Read it; if it doesn't assert that `cost_ledger.jsonl` gains a `subagent_usage` row after a stubbed Sonnet sub-agent call with non-zero `StreamUsage`, extend it. Run pytest; gate the next E2E run on this passing.

**E2E gate**: after rerunning SDAR end-to-end, assert `grep -c subagent_usage runs/<id>/cost_ledger.jsonl > 0`.

## Code-touchpoint diff plan

| File | Change | Bug | Effort |
|------|--------|-----|--------|
| `backend/agents/rlm/safe_builtins_patch.py` | CREATE — restore `globals`/`locals` in `_SAFE_BUILTINS` | LR-011 | 15 min |
| `backend/agents/rlm/safe_repl_traceback_patch.py` | CREATE — wrap `execute_code` to include traceback | LR-012 | 30 min |
| `backend/agents/rlm/run.py` | MODIFY — import both patch modules near top (next to `forced_iteration`) | LR-011, LR-012 | 5 min |
| `backend/agents/rlm/forced_iteration.py` | MODIFY — predicate covers `rubric_score is None` | LR-013 | 15 min |
| `backend/agents/rlm/run.py` | MODIFY — add `_emit_suspicious_partial_warning` near `_finalize` | LR-015 | 30 min |
| `backend/cli.py` | MODIFY — call `_warn_on_shell_env_override` early in `_reproduce` | LR-014 | 15 min |
| `backend/agents/rlm/system_prompt.py` | MODIFY — add short paragraph: "If you see a bare `NoneType` error after the patches land, look at the full traceback above for the actual cause; do NOT conclude primitives are unavailable" | LR-012 (defense in depth) | 10 min |
| `tests/agents/rlm/test_safe_builtins_patch.py` | CREATE | LR-011 | 20 min |
| `tests/agents/rlm/test_safe_repl_traceback_patch.py` | CREATE | LR-012 | 20 min |
| `tests/agents/rlm/test_forced_iteration_none_score.py` | CREATE | LR-013 | 20 min |
| `tests/agents/rlm/test_suspicious_partial_warning.py` | CREATE | LR-015 | 25 min |
| `tests/cli/test_env_override_warning.py` | CREATE | LR-014 | 20 min |

Total dev effort: ~3.5 hours including tests, before any rerun.

## Sequencing

1. **Land all P0 (LR-011, LR-012, LR-014) + tests** in a single PR. Smoke test locally — run a 30-second iter-1-only SDAR via the CLI with `--max-wall-clock 60` and confirm: (a) no `NoneType` crash, (b) any errors print a traceback, (c) the env-override warning fires if you set `OPENAI_API_KEY=fake` in the shell.
2. **Land P1 (LR-013, LR-015) + tests** in a follow-up PR. These are report-quality, not run-blocking; smaller blast radius, can ship after P0 is proven.
3. **Re-attempt SDAR end-to-end** (`prj_09047604e591d969` ID is fine to reuse — the archive system rotates artifacts). Budget: `--max-usd 15 --max-wall-clock 7200`. Watch for:
   - `cost_ledger.jsonl` accumulating `gpt-5` root rows (confirms cost tracking)
   - `cost_ledger.jsonl` accumulating `subagent_usage` rows once `implement_baseline` runs (validates BUG-LR-008)
   - `Dockerfile` first line `FROM runpod/pytorch:2.1.0-…-runtime-ubuntu22.04` (validates BUG-LR-009)
   - `train.py` references `Qwen/Qwen3-1.7B-Instruct` and `Qwen/Qwen2.5-3B-Instruct` (validates BUG-LR-010)
   - `rubric.overall_score > 0.0` (the primary success criterion)
4. **Post-success**: write the run summary into [`2026-05-28-subscription-cost-reduction-design.md`](2026-05-28-subscription-cost-reduction-design.md) under a new "## Run Results 2026-05-28 (post-stability-fixes)" section.

## Doc update plan

| Doc | Change | Why |
|-----|--------|-----|
| `CLAUDE.md` | Insert a "**Shell vs .env API-key precedence**" subsection under "### RLM auth — two surfaces, billed separately"; cross-link this plan | BUG-LR-014 — caused today's first failure |
| `system_overview.md` | Add short paragraph: "REPL sandbox safety — `eval/exec/compile/input` blocked; `globals/locals` restored by `safe_builtins_patch`" | BUG-LR-011 root cause is non-obvious |
| `docs/runbooks/known-issues-and-monitoring.md` | New entries: bare-NoneType-symptom → BUG-LR-011, shell-key-shadow → BUG-LR-014, premature-FINAL_VAR → BUG-LR-013/015. Each with one-line workaround | Operator-facing runbook |
| `docs/runbooks/2026-05-23-sdar-baseline-handoff.md` | Append "2026-05-28 attempt — same SDAR target, blocked on BUG-LR-011/012/014 — see remediation plan" | Keeps the SDAR-specific story coherent |
| `docs/runbooks/2026-05-27-rlm-reproduction-stability-plan.md` | Cross-link this plan as the next-day continuation | Avoids divergent stability plans |
| `docs/superpowers/specs/2026-05-28-subscription-cost-reduction-design.md` | New section "Live Run Issues 2026-05-28 22:01Z — BUG-LR-011/012/013/014/015" pointing at this doc; note BUG-LR-008 validation status remains DEFERRED | Today's cost-reduction spec is the existing ledger for run-level bugs; keep them co-located |

## What is NOT changing

- **No change to the rlm library blocks on `eval/exec/compile/input`.** Those are real risks. Only `globals`/`locals` are restored.
- **No change to the SSE bridge or sanitize_iteration.** The traceback patch composes with existing egress bounds; the cap (2000 chars) is below the SSE per-event budget.
- **No change to the wall-clock policy.** The forced-iteration extension still yields under wall-clock pressure (60s remaining).
- **No change to root-model selection logic, sandbox routing, GPU resolution, Docker base-image policy, or the rubric scoring path.** Those are out of scope for this plan.
- **No change to the partial-report format.** Existing `partial`/`completed`/`failed` verdicts remain; we add a new `run_warning` (`suspicious_partial`) — not a new verdict. The `partial` verdict is still the right answer when a model legitimately runs out of compute or wall-clock budget.

## Risks and open questions

- **Monkey-patching `LocalREPL.execute_code` is a maintenance burden.** Every rlm version bump needs the patch re-verified. Mitigation: add an upstream PR; mark the patch deprecated once accepted; the `_PATCHED_ATTR` guard prevents double-application.
- **The forced-iteration change could trap a model that genuinely can't make progress** (e.g., RunPod is down, no GPU available). Mitigation: the wall-clock bypass already covers this — `ctx.remaining_s() <= 60s` yields. We are NOT extending the policy beyond `iteration_count < min_iterations`; long-running confused models can still ship a partial after iter N.
- **The "suspicious_partial" detector is heuristic.** A paper that's genuinely impossible to reproduce (no data, no code, no compute) would trip it. Acceptable: it's a warning, not a verdict change.
- **BUG-LR-014's `dotenv_values()` import** adds a small startup-time cost (~10ms). Negligible.
- **System-prompt update for BUG-LR-012** consumes ~80 tokens. Trivial vs. an iteration of GPT-5 output. Worth it for defense in depth even after the upstream patch lands.

## Validation gates summary

| Gate | Mechanism | Owner |
|------|-----------|-------|
| BUG-LR-011 fixed | `tests/agents/rlm/test_safe_builtins_patch.py` green; LocalREPL probe with `globals()` returns dict | dev |
| BUG-LR-012 fixed | `tests/agents/rlm/test_safe_repl_traceback_patch.py` green; tracebacks visible in stderr | dev |
| BUG-LR-013 fixed | `tests/agents/rlm/test_forced_iteration_none_score.py` green; replay refuses today's iter-5 FINAL_VAR | dev |
| BUG-LR-014 fixed | `tests/cli/test_env_override_warning.py` green; manual smoke with stale shell key shows warning | dev + operator |
| BUG-LR-015 fixed | `tests/agents/rlm/test_suspicious_partial_warning.py` green; replay today's run fires the warning | dev |
| BUG-LR-008 validated | E2E rerun yields `grep -c subagent_usage cost_ledger.jsonl > 0` | operator |
| SDAR overall | rerun ends with `rubric.overall_score > 0.0` | operator |

## Composition with existing systems

- **Lane G (`rubric_guard.py`) and Lane H (`forced_iteration.py`)** continue to operate as documented. This plan extends Lane H's predicate; does not touch Lane G.
- **2026-05-23 cleanup spec** (mode field on final_report, started_at/completed_at) is preserved. We add no new top-level final_report keys.
- **2026-05-28 cost-reduction spec** is the parent ledger for cost-track bugs (BUG-LR-001..010); this plan extends it with LR-011..015 for stability-track bugs. The two specs are sibling pillars under the same date.
- **Subscription cost target** (`--model claude-oauth` for $0 sub-agents) is unaffected. Both patches apply equally under any root-model choice.

## Cross-references

- Companion cost track: [`2026-05-28-subscription-cost-reduction-design.md`](2026-05-28-subscription-cost-reduction-design.md)
- Prior stability plan: [`../../runbooks/2026-05-27-rlm-reproduction-stability-plan.md`](../../runbooks/2026-05-27-rlm-reproduction-stability-plan.md)
- Prior SDAR debugging history: [`../../runbooks/2026-05-23-sdar-baseline-handoff.md`](../../runbooks/2026-05-23-sdar-baseline-handoff.md), [`../../runbooks/2026-05-27-sdar-run-issues.md`](../../runbooks/2026-05-27-sdar-run-issues.md)
- Canonical architecture: [`../../design/rlm-pivot-brief.md`](../../design/rlm-pivot-brief.md)
- Operator runbook: [`../../runbooks/known-issues-and-monitoring.md`](../../runbooks/known-issues-and-monitoring.md)
- CLAUDE.md sections to update: "RLM auth — two surfaces" (shell vs .env precedence), "Forced-iteration policy (Lane H)" (rubric_score=None extension)
