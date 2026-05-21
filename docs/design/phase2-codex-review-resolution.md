# Phase 2 (PR #68) — Codex Review Resolution

> Resolution of `phase2-codex-review-findings.md` (the independent adversarial
> review run against `phase2-codex-review-prompt.md`) **and** the broader
> 8-point review prompt that preceded it. Every finding was re-verified from
> primary source — the installed `rlm` 0.1.1 package, the real stage agents,
> the real schemas — before being fixed or refuted.

## Verdict

**Ship.** Every Important (I1–I7) and every actionable Minor (M1–M7) finding is
resolved in-branch; the over-reach findings are refuted with evidence. The one
finding that cannot be code-fixed in Phase 2 — **I1, the prompt-injection →
host-RCE surface** — is now a documented **hard Phase-3 gate**: Phase 2 ships
only the primitive layer and never constructs `rlm.RLM(...)`, so there is no
REPL here to harden; the gate stops Phase 3 from wiring the REPL until the
threat model is resolved.

13 findings fixed in code/docs, 1 documented as a Phase-3 gate, 2 documented as
tracked deferrals (#62 / #60), 3 refuted. Full suite green modulo 2 pre-existing
unrelated failures (see Verification).

---

## Confirmed findings — fixed

### Important

**I1 — `environment="local"` + open `__import__`/`open` + an attacker-controllable
paper is a prompt-injection → host-RCE path.** Confirmed from source:
`rlm/environments/local_repl.py:89-90` keeps `__import__` and `open` in
`_SAFE_BUILTINS`; `LocalREPL.execute_code` runs the root model's code via
`exec(code, combined, combined)` (`local_repl.py:492`) in the host process.
`environment="local"` is mandatory (`DockerREPL` drops `custom_tools`), and the
paper — offloaded as `context` — is an arXiv id / uploaded PDF, i.e.
attacker-influenceable. *Resolution:* Phase 2 has no `RLM(...)` construction to
harden, so this is a **documented Phase-3 gate**, not a code fix. A threat-model
block was added to `rlm-pivot-brief.md` §7; the stale §8 "the root model is
trusted — do not sink time into [sandboxing]" note (which conflated *model*
trust with *input* trust) is corrected. **Phase 3 (#60) must not construct
`rlm.RLM(environment="local")` until the threat model is resolved.**

**I2 — `build_environment` used a fixed, mutable Docker tag.** `primitives.py`
built `tag = f"reprolab/{ctx.project_id}:env-check"` — constant per run. A
Docker tag is a mutable pointer: two `build_environment` calls in one run
collided, after which `run_experiment` ran whichever image the tag last pointed
at, regardless of the `image_tag` value threaded in. *Fixed:* the tag is keyed
to a sha1 of the Dockerfile (`env-{digest}`) — distinct environments get
distinct images; rebuilding the same Dockerfile is stable/cache-friendly.

**I3 — `build_environment`'s D3 fail-soft contract had a hole.**
`SandboxRuntimeError` was imported *inside* the `try` block it is named in. If
the import line above it (`from backend.config import get_settings`) raised,
`SandboxRuntimeError` was unbound, so evaluating `except SandboxRuntimeError:`
raised `NameError` — which escapes the whole `try`, violating "the primitive
never raises [except `SandboxRuntimeError`]". *Fixed:* the import is hoisted
above the `try`, so the `except` name is always bound.

**I4 — `run_experiment`'s unbounded `logs` flowed verbatim into LLM calls.**
`_execute_in_sandbox` returned `"logs": "\n".join(r.stdout …)` — a training run
can emit megabytes — and `verify_against_rubric` / `propose_improvements`
`json.dumps` the whole result into an LLM prompt, re-creating the
context-window blow-up RLM exists to avoid (brief §8). *Fixed:* `_cap_logs`
bounds the log to a head+tail window (`_MAX_LOG_CHARS = 16000`) before it
leaves `run_experiment`.

**I5 — `plan_reproduction`'s prompt never named the `ReproductionContract` JSON
keys.** `ReproductionContract` is `extra="ignore"` and every field has a
default; the prompt described the fields in prose only, so a response keyed on
prose-guessed names (`"faithful_reproduction"` vs `reproduction_definition`)
was silently dropped to an all-defaults, near-empty contract — and the
one-key-match guard let it through. *Fixed:* the user prompt now enumerates the
exact keys, derived from `ReproductionContract.model_fields` so prompt and
schema cannot drift.

**I6 — nothing pinned the root REPL to `environment="local"`; `DockerREPL`
swallows `custom_tools` silently.** Confirmed: `DockerREPL.__init__` takes no
`custom_tools` param (absorbed by `**kwargs`). Phase 2 builds only the dict, so
nothing is broken today. *Resolution:* `build_custom_tools`' docstring now
states the consumer MUST use `rlm.RLM(environment="local")`; brief §7 records
that Phase 3 should centralise the `RLM(...)` construction in one helper that
hardcodes it.

**I7 — the verify → improve loop is structurally inert in Phase 2.**
`_execute_in_sandbox` returns `metrics={}` (real metric extraction is Phase 5,
#62); `verify_against_rubric` treats no-metrics as `degraded` and caps every
area at 0.35, so no Phase-2 run can score above 0.35. This is honest-by-design
(the cap is *correct* behavior with no metrics) — not a bug. *Resolution:* no
code change; **#62 is a hard tracked dependency** before any phase relies on
the loop's score or improvement signal.

### Minor

**M1 — error events leaked argument/return values.** `wrap_primitive`'s error
path emitted `result_summary = f"{type(exc).__name__}: {exc}"` — an exception
message can carry raw LLM output or paper text, and `result_summary` streams to
the UI. *Fixed:* the error path emits the exception *type* only; the full
exception still propagates via `raise` for server-side logs.

**M2 — `_extract_json` silently mis-parsed a truncated object.** Scanning from
each `{`, a truncated outer object `{"a":1,"b":{"c":2}` returned the intact
*inner* fragment `{"c":2}` — a silent wrong parse. *Fixed:* a `JSONDecodeError`
whose `.pos` reaches end-of-text means the structure ran off the end (truncated
response); `_extract_json` now raises instead of falling through to an inner
fragment. (Residual Minor: a top-level array still yields its first object —
left as-is; it needs a malformed response and the callers all expect an
object.)

**M3 — `verify_against_rubric` was not robust to a malformed `areas` payload.**
Iterating `parsed["areas"]` and calling `a.get(...)` raised `AttributeError` if
the LLM returned a non-list or non-dict items. *Fixed:* non-list `areas` → `[]`;
non-dict items skipped (consistent with `propose_improvements`' tolerance).

**M4 — primitive cost-ledger rows are always zero-usage.** `wrap_primitive`
writes a `CostLedgerEntry` with no token counts (D7 — real usage lands with
`run.py`, #60). *Resolution:* no code change — acknowledged tracked deferral.
A Phase-3 cost cap **must not** be built on the primitive ledger until #60
lands real token accounting; the LLM spend of `plan_reproduction` /
`verify_against_rubric` / `propose_improvements` / `build_environment` repair is
invisible to it for the whole of Phase 2.

**M5 — `propose_improvements`' `k` was unvalidated.** A non-positive `k`
produced an empty/surprising `out[:target]` slice. *Fixed:* `k` is clamped to
`>= 1`. (The fail-soft silent drop of a malformed hypothesis is left as-is —
surfacing a drop count would require changing the `list` return to a dict.)

**M6 — stale `__init__.py` docstring.** Said "these modules are stubs. Phase 2
will implement them". *Fixed* as part of #2 (package rewrite).

**M7 — `_execute_in_sandbox` hard-coded `timeout=3600`.** *Fixed:* lifted to a
named module constant `_EXEC_TIMEOUT_SECONDS` with a note that Phase 3 should
source it from settings / `ctx`. (Container resource bounds — network/memory/
cpu — are already enforced by `SandboxConfig` defaults; only the time bound was
a magic number.)

### From the 8-point review prompt

**#2 — the retired Phase-1 skeleton was still live and exported.**
`repl_host.py` / `root_loop.py` / `sub_call.py` / `system_prompt.py` were
`NotImplementedError` stubs that the `rlms`-library pivot retired
(`rlms-spike-report.md`, committed in Phase 2's own range as `88c64fe`), yet
`__init__.py` still re-exported `ReplHost` / `RootLoop` / `llm_query` /
`build_system_prompt` … as if public API — a Phase-3 landmine. *Fixed:* the
four stub files are deleted (verified: nothing outside `__init__.py` imported
them); `__init__.py` exports only the real Phase 2 surface (`RunContext`,
`build_custom_tools`, `PRIMITIVE_REGISTRY`, `PRIMITIVE_DESCRIPTIONS`).

**#6 — misleading deferred-import comment in `binding.py`.** The comment claimed
the deferred `primitives` import "keeps this module importable without loading
primitives.py" — false: importing `binding` runs the package `__init__`, which
imports `primitives`. *Fixed:* the comment now states the real benefit (an
explicit `registry=`/`descriptions=` call skips the `primitives` import).

**#8 — untracked review artifacts.** `phase2-codex-review-prompt.md` and
`phase2-codex-review-findings.md` were untracked. *Resolved:* both are committed
alongside this resolution doc, completing the review trail.

**Incidental** (found while editing brief §8 for I1): invariant #4 still said
termination is "the library's `answer` mechanism" — the same stale claim §7's
audit fix corrected. Now reads `FINAL_VAR` (there is no reserved `answer`
variable).

---

## Refuted suspicions

**#1 — "High blocker: environment dependency conflict."** Refuted as a blocker.
`pip install --dry-run --ignore-installed -r backend/requirements.txt` resolves
**cleanly** and pulls **no `litellm`** — `deepeval` does not depend on it. The
`pip check` warning (`litellm` pins `python-dotenv==1.0.1`) comes entirely from
`dspy`, which the RLM-engine spike (`tools/rlms_spike.py`) installed into the
`.venv` to evaluate `dspy.RLM` (rejected, issue #66); neither `dspy` nor
`litellm` is in any requirements file. *Valid sub-point, fixed:* the audit doc's
"pre-existing eval-path over-pin" wording was inaccurate (`deepeval` is not the
source) — `phase2-plan-audit.md` now names the real cause and cites the dry-run.

**#4 — "`implement_baseline(plan)` is a non-uniform signature / D4 violation."**
Refuted. `plan` is an aggregate of *structured specs* (`paper_claim_map`,
`environment_spec`, `reproduction_contract`) — not a whole-corpus argument, so
it does not violate the Algorithm-2 guard, and D4 concerns the injected `ctx`
kwarg, not flat-vs-aggregate arguments. The aggregate is deliberate, documented
in the docstring and `PRIMITIVE_DESCRIPTIONS`, and is the signature the brief
itself specifies (§7: `implement_baseline(plan)`); the independent findings
doc's own cross-primitive data-flow audit traced the `plan["…"]` hand-offs and
marked them ✅. No change.

**#7 — "`ctx` kwarg is silently discarded."** Refuted. `wrap_primitive` does
`kwargs.pop("ctx", None)` so a caller cannot double the `ctx` the wrapper
supplies. The root model is never told `ctx` exists (`PRIMITIVE_DESCRIPTIONS`
omits it), so this defends a near-impossible case; and the recovery is
*correct* — the wrapper's closed-in `ctx` is the right one. Raising instead
would convert a benign, correctly-recoverable contract slip into a run-killing
crash. The line is already commented. No change.

---

## Changes made

| File | Change |
|---|---|
| `backend/agents/rlm/repl_host.py`, `root_loop.py`, `sub_call.py`, `system_prompt.py` | **Deleted** — retired skeleton stubs (#2). |
| `backend/agents/rlm/__init__.py` | Rewritten — skeleton exports removed, docstring de-staled (#2, M6). |
| `backend/agents/rlm/primitives.py` | I2 (hashed image tag), I3 (import hoist), I4 (`_cap_logs`), I5 (named contract keys), M2 (`_extract_json` truncation guard), M3 (malformed `areas`), M5 (clamp `k`), M7 (`_EXEC_TIMEOUT_SECONDS`). |
| `backend/agents/rlm/binding.py` | M1 (value-free error events), #6 (corrected comment), I6 (`environment="local"` guard-rail in `build_custom_tools` docstring). |
| `tests/rlm/` | 6 new tests pinning I2, I4, I5, M2, M3, M5; M1 assertion added to `test_binding.py`. |
| `docs/design/rlm-pivot-brief.md` | §7 threat-model block (I1/I6); §8 note corrected (I1); §8 invariant #4 `answer` → `FINAL_VAR`. |
| `docs/design/phase2-plan-audit.md` | `pip check` / `litellm` note rewritten with the verified cause (#1). |
| `docs/design/phase2-codex-review-prompt.md`, `phase2-codex-review-findings.md`, `phase2-codex-review-resolution.md` | Committed — the review trail (#8). |

I3 has no dedicated test: forcing an import inside the `try` to fail is
artificial, the fix is correct by inspection, and
`test_build_environment_propagates_sandbox_runtime_error` still passing proves
the happy path through the `except SandboxRuntimeError` clause.

---

## Verification

```
pip install --dry-run --ignore-installed -r backend/requirements.txt
  → resolves cleanly; "Would install …" lists no litellm.
pip check
  → one line: litellm 1.83.7 requires python-dotenv==1.0.1 (you have 1.2.2).
    Spike-leftover dspy/litellm — not a backend/requirements.txt defect (see #1).
python -c "import backend.agents.rlm as r; print(sorted(r.__all__))"
  → ['PRIMITIVE_DESCRIPTIONS', 'PRIMITIVE_REGISTRY', 'RunContext', 'build_custom_tools']
pytest tests/rlm/ -q
  → 37 passed (31 pre-existing + 6 new pinning tests).
pytest tests/ -q
  → 915 passed, 2 failed, 2 skipped.
```

The **2 failures are pre-existing and unrelated to Phase 2** —
`test_issue17_runtime.py::test_local_process_backend_executes_commands_with_artifact_env`
and
`test_issue26_experiment_runner.py::TestRunWithRuntime::test_local_process_mode_runs_without_dockerfile`,
both failing with `/bin/sh: python: command not found` (a local-process-backend
PATH issue). Neither this branch's diff nor these fixes touch
`backend/services/runtime/` or those test files. The findings doc recorded the
same 2 failures at a 909-passed baseline; `915 = 909 + 6` new tests confirms
**zero regressions**. The 2 skips are the usual optional-`chromadb` skips.
