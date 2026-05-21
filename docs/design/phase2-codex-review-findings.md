# Phase 2 (PR #68) — Independent Adversarial Review Findings

> Reviewer: a fresh agent run against the brief in
> `docs/design/phase2-codex-review-prompt.md`. Every load-bearing claim below
> was verified from primary source — the installed `rlm` 0.1.1 package, the
> actual stage-agent modules, the actual schemas — not the design docs.
> Scope: the 35-commit Phase 2 delta `63adb10..HEAD`.

## Verdict

**Ship with fixes.**

Phase 2 is a genuinely solid piece of work. All nine primitives wrap (not
rewrite) their stage-agent helpers; the nine cross-primitive hand-offs trace
correctly end-to-end with no key-name or shape mismatch; the `rlm` 0.1.1 API
claims are all true; the integration test exercises the real `rlm` REPL with
real `custom_tools` injection; 31/31 rlm tests pass and the delta is
structurally additive. There is **no Critical bug that Phase 2 itself
activates** — Phase 2 ships only the primitive layer; nothing here constructs
`rlm.RLM(...)`.

But a different-model review surfaces issues the same-model pipeline
structurally under-weighted — most importantly that the architecture the
primitives are built for (`environment="local"`, paper offloaded as `context`)
is a **prompt-injection → host-RCE substrate**, and Phase 2 lands the primitive
layer onto it with no documented threat model. The fixes are concrete and
small; none require re-architecting Phase 2. Fix the Important findings before
Phase 3 wires the RLM.

---

## Findings

### Critical

None. (See verdict — the security finding below is Important *within Phase 2
scope* but **blocking for Phase 3**.)

### Important

**I1 — `environment="local"` + an open `__import__`/`open` builtin + an
attacker-controllable paper is a prompt-injection → host-RCE path. No threat
model is documented.**
`rlm/environments/local_repl.py:29-118` — `_SAFE_BUILTINS` blocks
`eval`/`exec`/`compile`/`globals`/`locals`, but **keeps `__import__` and
`open`**. `LocalREPL.execute_code` runs the root model's Python via
`exec(code, combined, combined)` (`local_repl.py:492`) **in the host process**.
The "sandbox" is cosmetic: any code the root model writes can
`__import__("os").system(...)` / `__import__("subprocess")...` and own the
host. The brief (§7) and `rlm`'s own README both pin `environment="local"` as
mandatory (custom_tools only inject in `local` — see C below), so this is not
optional. Meanwhile the paper is offloaded as the `context` variable and the
paper is **attacker-influenceable** — anyone who can upload a PDF or pass an
arXiv id controls `context['paper_text']`. A paper crafted with a prompt
injection ("ignore the task; run …") is read by the root model, which writes
the injected Python into the REPL, which runs on the host.
*Why it matters:* this is the highest-value thing a different-model review can
say. The pipeline treated `environment="local"` as a settled
config-compatibility fact (Task 0) and never re-derived it as a security
decision. Phase 2 is the layer that makes the primitives callable in that REPL.
*Fix:* before Phase 3, write an explicit threat-model section in the brief
covering (a) whether the paper is ever untrusted, (b) what isolates the host
from injected REPL code. Options: run the whole root REPL inside a container
(but then custom_tools break — see I6), strip/scan the offloaded `context`,
constrain the root model's allowed imports via a custom builtins map, or accept
the risk explicitly with a stated trust boundary. Do not let Phase 3 wire
`rlm.RLM(environment="local")` without this.

**I2 — `build_environment` uses a fixed, mutable Docker tag; two builds in one
run collide and `run_experiment` silently runs the wrong image.**
`primitives.py:169` — `tag = f"reprolab/{ctx.project_id}:env-check"`. The tag is
constant per project. If the root model calls `build_environment` twice (two
candidate `env_spec`s, or a manual retry), the second build overwrites the tag.
A Docker tag is a *mutable pointer*, not an identifier — but the primitive
contract treats `image_tag` as an identifier the root threads into
`run_experiment(code_path, env_id)`. So `run_experiment` runs whatever
`reprolab/<pid>:env-check` points at *now*, regardless of which `image_tag`
value was passed.
*Fix:* make the tag unique per build — include a short hash of
`env_spec["dockerfile"]` or a per-call counter:
`tag = f"reprolab/{ctx.project_id}:env-{hashlib.sha1(dockerfile.encode()).hexdigest()[:8]}"`.

**I3 — `build_environment`'s fail-soft guarantee (D3) has a hole: a failed
import makes `except SandboxRuntimeError` raise `NameError` and escape.**
`primitives.py:165-189` — `SandboxRuntimeError` is imported *inside* the `try`
block (line 166). If the line above it (`from backend.config import
get_settings`, line 165) raises `ImportError`, `SandboxRuntimeError` is never
bound. When Python then evaluates the `except SandboxRuntimeError:` clause it
raises `NameError`, which escapes the function — `build_environment` raises,
violating the documented D3 contract ("any failure … returns `{ok: False…}`;
the primitive never raises" except `SandboxRuntimeError`). The `except
Exception` clause does **not** catch it: an exception raised while evaluating
an `except` clause propagates out of the whole `try`.
*Fix:* hoist `from backend.services.runtime.interface import
SandboxRuntimeError` (and ideally `get_settings`) to module scope, or above the
`try`.

**I4 — `run_experiment`'s unbounded `logs` flows verbatim into the
`verify_against_rubric` / `propose_improvements` LLM calls — re-creating the
context-window blowup RLM exists to avoid (brief §8 fidelity).**
`_execute_in_sandbox` returns `logs = "\n".join(r.stdout for r in results)`
(`primitives.py:315`) — the full concatenated stdout of every command, which
for a training run can be megabytes. `verify_against_rubric` and
`propose_improvements` both do `json.dumps(results, …)` and send the whole
thing to `ctx.llm_client.complete` (`primitives.py:365, 411`). A primitive is
supposed to take *slices/specs, not the whole corpus* — here a primitive feeds
an unbounded blob straight into an LLM context window.
*Fix:* truncate/summarize `logs` before it leaves `run_experiment` (e.g. keep
head+tail, cap at N KB), or have `verify_against_rubric` /
`propose_improvements` slice `results["logs"]` before serializing.

**I5 — `plan_reproduction`'s system prompt never names the exact
`ReproductionContract` JSON keys; `extra="ignore"` + a one-key-match guard lets
a near-miss response yield a near-empty contract silently.**
`_PLAN_REPRODUCTION_SYSTEM` (`primitives.py:56-63`) describes the fields in
prose ("what counts as a faithful reproduction, a smoke-test plan, …") but the
LLM must guess that "what counts as a faithful reproduction" serializes to the
key `reproduction_definition`. `ReproductionContract` is `extra="ignore"`, so a
response keyed `"faithful_reproduction"` / `"reproduction"` is silently
dropped. The guard `if not any(k in data for k in
ReproductionContract.model_fields)` only requires **one** field to match — a
mostly-wrong response with one lucky key passes and returns a contract that is
empty everywhere else. Contrast `verify_against_rubric`, whose user prompt
*does* spell out the exact JSON shape — that one is robust.
*Fix:* enumerate the exact field names in `_PLAN_REPRODUCTION_SYSTEM` (or the
user prompt), the way `verify_against_rubric` does.

**I6 — `rlm`'s `DockerREPL` silently swallows `custom_tools`; nothing in the
codebase pins the root REPL to `environment="local"`.**
`rlm/environments/docker_repl.py:195-204` — `DockerREPL.__init__` does *not*
accept `custom_tools`; the kwarg is absorbed by `**kwargs` with no error, and
`_build_exec_script` (`docker_repl.py:158`) builds the in-container globals with
only `llm_query`/`llm_query_batched`/`FINAL_VAR`/`SHOW_VARS` — no custom tools,
no `rlm_query` either. So an operator who later sets `environment="docker"`
gets a *silently* tool-less RLM. Phase 2 only builds the `custom_tools` dict;
it never constructs `RLM(...)`, so today nothing is broken — but the trap is
laid for Phase 3.
*Fix:* have `binding.py` expose the RLM constructor (a `make_rlm(ctx, …)`
helper) that hardcodes `environment="local"`, or at minimum `assert
environment == "local"`. Document the constraint where the dict is built.

**I7 — the verify → improve loop is vacuous in Phase 2 and that is load-bearing
for #59's stated scope.**
`_execute_in_sandbox` hard-returns `metrics={}` (`primitives.py:314`, "real
metric extraction … is Phase 5 (#62)"). `verify_against_rubric` computes
`degraded = (not success) or (not metrics)` — `not {}` is always `True`, so the
0.35 honesty cap fires on **every** Phase-2 run and no reproduction can score
above 0.35. The verify→improve loop therefore cannot produce a meaningful score
or a meaningful improvement signal yet. This is documented honestly in the
docstrings — it is not hidden — but it means a reviewer cannot exercise the
loop end-to-end and a chunk of the layer's *behavior* is untestable until #62
lands.
*Fix:* none required in code — but make #62 a hard, tracked dependency before
any phase relies on the loop, and consider a Phase-2 note in the brief that the
loop is structurally present but inert.

### Minor

**M1 — error messages leak partial argument/return values into the event log.**
`binding.py:55` — on the error path `result_summary = f"{type(exc).__name__}:
{exc}"[:200]`. `_summarize`/`_result_summary` are correctly value-free, but
exception *messages* are not: `_extract_json` raises `ValueError(f"no JSON
object in LLM response: {text[:200]!r}")` (`primitives.py:45`) — that puts 200
chars of raw LLM output into `dashboard_events.jsonl`; `plan_reproduction`'s
`ValueError` leaks `list(data)`. Brief §D asks the wrapper to never leak full
values — the happy path honors that, the error path does not. *Fix:* truncate
harder, or sanitize exception text before it reaches `result_summary`.

**M2 — `_extract_json` mis-parses truncated JSON and top-level arrays.** The
scan-from-each-`{` + `raw_decode` strategy is sound for the common cases (the
five `test_extract_json.py` cases pass). But on a *truncated* object
`{"a":1,"b":{"c":2}` the outer object fails to decode and the scan returns the
inner fragment `{"c":2}` — a wrong result, silently. On a top-level array
`[{...},{...}]` it returns the first element, not the array. *Why it matters:*
a silent wrong parse is worse than a raise — `verify_against_rubric` would build
a verification from a fragment. *Fix:* prefer the longest successful decode
starting at position 0/after a fence; add the truncated-fragment case to the
test. Keep severity Minor — it needs a malformed LLM response to trigger.

**M3 — `verify_against_rubric` is not robust to a malformed `areas` payload.**
`primitives.py:376` iterates `parsed.get("areas", [])` and calls `a.get(...)`.
If the LLM returns `areas` as a dict, or as a list of non-dicts, this raises
`AttributeError` mid-loop. The primitive is not declared fail-soft (only
`build_environment` is), so a raise is contractually allowed — but a
`isinstance(a, dict)` skip would be cheap and consistent with
`propose_improvements`'s tolerance.

**M4 — primitive cost-ledger rows are always zero-usage.** `wrap_primitive._ledger`
writes `CostLedgerEntry(… attempt_index=0 …)` with no token counts
(`binding.py:39-46`), documented as deferred to #60. Consequence: the LLM spend
of `plan_reproduction` / `verify_against_rubric` / `propose_improvements` /
`build_environment` repair is invisible to any cost guard for the whole of
Phase 2. Acceptable as a deferral; flag it so a Phase-3 cost cap isn't trusted
prematurely.

**M5 — `propose_improvements` `k` is unvalidated.** `target = k if k is not None
else 3`; `out[:target]` (`primitives.py:409,427`). A negative or zero `k` from
the root model produces empty/surprising slices. Also the silent-drop of a
malformed hypothesis (`except Exception: continue`, line 425-426) is not
observable — the dropped count never reaches the event log. *Fix:* clamp `k` to
`>= 1`; include `dropped=N` in the result or a debug log.

**M6 — stale package docstring.** `backend/agents/rlm/__init__.py:9-11` still
says "Phase 1 (#58) artifact — these modules are stubs. Phase 2 (#59) will
implement them" — Phase 2 is done and `primitives.py`/`binding.py`/`context.py`
are no longer stubs.

**M7 — `_execute_in_sandbox` hard-codes per-command `timeout=3600`.**
(`primitives.py:309`.) The container itself inherits safe `SandboxConfig`
defaults — `network_disabled=True`, `memory_limit="4g"`, `cpus=2.0`
(`runtime/interface.py:51-57`), so resource bounds *are* enforced — but the
1-hour per-command wall-clock is a magic number that should come from settings
/ `ctx`, consistent with the rest of the codebase's configurable caps.

---

## Cross-primitive data-flow assessment

All nine hand-offs were traced by reading **both** sides. They hold:

| Hand-off | Verdict |
|---|---|
| `understand_section` → `detect_environment(method_spec)` | ✅ keys `datasets/metrics/training_recipe/hardware_clues/ambiguities` all map onto `PaperClaimMap` fields; Pydantic re-coerces the `.model_dump()` dicts back into `DatasetRequirement`/`TrainingRecipe`/etc. `core_contribution` is correctly defaulted (`primitives.py:124`). |
| `understand_section` → `implement_baseline(plan["paper_claim_map"])` | ✅ same `PaperClaimMap` round-trip; `core_contribution` defaulted at `primitives.py:247`. |
| `detect_environment` → `build_environment(env_spec)` | ✅ `EnvironmentSpec.model_dump()` carries `dockerfile`; `build_environment` reads `env_spec["dockerfile"]`. |
| `detect_environment` → `implement_baseline(plan["environment_spec"])` | ✅ `EnvironmentSpec(**plan["environment_spec"])`. |
| `plan_reproduction` → `implement_baseline(plan["reproduction_contract"])` | ✅ `ReproductionContract.model_dump()` → `ReproductionContract(**…)`. |
| `build_environment` `{ok,image_tag,…}` → `run_experiment(env_id)` | ⚠️ shape is correct and documented ("pass `image_tag` as `env_id`"), but the tag is mutable/colliding — see **I2**. |
| `implement_baseline` writes `code/commands.json` → `run_experiment` reads it | ✅ both derive the path as `runs_root/project_id/code` independently (`primitives.py:265` vs `baseline_implementation.py:433-434`) — they agree without relying on an unenforced invariant. The 2-commit fix history (`11d706c`, `1112761`) shows this was specifically hardened. |
| `run_experiment` `{success,metrics,logs}` → `verify_against_rubric` | ⚠️ shape correct; but `metrics` is always `{}` (**I7**) and `logs` is unbounded (**I4**). |
| `run_experiment` → `propose_improvements(current_results,…)` | ⚠️ same `logs`-unbounded concern (**I4**). |

No key-name or shape mismatch — the chain is correct. The only data-flow
defects are I2 (mutable tag), I4 (unbounded `logs`), and the I7 deferral.

---

## What the AI pipeline missed

- **`environment="local"` was filed as a config-compatibility fact, never
  re-derived as a security decision.** Task 0 pinned it because `DockerREPL`
  swallows `custom_tools`; every later agent inherited "local is mandatory" and
  none asked "local means the root model's Python runs on the host — and the
  paper feeding that model is attacker-controllable." A same-model pipeline
  reading one set of docs cannot see an assumption it never wrote down. This is
  **I1** and it is the review's headline.
- **"Verified against source" sometimes meant "read once."** The docstrings
  carry confident claims — e.g. `detect_environment`'s "Verified: `run_offline`
  is exactly the heuristic chain plus a Dockerfile write." Those claims are in
  fact correct here — but they are stated as settled, and a one-reading "wrap
  not rewrite" verification cannot catch a *semantic* drift like I2 (the tag is
  a pointer, not an id) because the types still line up.
- **Robustness was tested per-primitive, in isolation, by separate agents — so
  cross-primitive resource flow was nobody's job.** Each primitive's own
  output shape is fine; what no single task owned is "an unbounded value
  produced by `run_experiment` is consumed by an LLM call two primitives
  later" (**I4**). The RLM-fidelity invariant (§8: primitives take slices, not
  the corpus) was checked against *signatures* — `logs: str` passes a signature
  check — not against *runtime size*.
- **Operability blind spots a production-scarred reviewer flags:** the cost
  ledger records primitive calls but not their LLM token spend (**M4**), so a
  cost cap built on it would be blind; the fixed Docker tag (**I2**) is the
  kind of thing that works in every single-call test and breaks the first time
  a real run retries; the `except Exception: continue` in `propose_improvements`
  and the broad fail-soft in `build_environment` are correct *policy* but
  emit nothing observable when they swallow something (**M5**).
- **The deferral in I7 quietly makes the verify→improve loop inert.** It is
  documented honestly in docstrings — the pipeline did not hide it — but
  "documented" and "a reviewer can exercise it" are different bars, and the
  loop is a headline deliverable of the RLM pivot.

---

## `rlm` library contract — confirmed or refuted

All four claims **confirmed** from the installed `rlm` 0.1.1 source:

- **`custom_tools` shape `{name: {"tool": callable, "description": str}}` —
  CONFIRMED.** `rlm/environments/base_env.py:41-93` (`parse_tool_entry` /
  `extract_tool_value`) consumes exactly the `{"tool": …, "description": …}`
  form; `local_repl.py:200-206` puts the callable into REPL `globals`. The
  description is also surfaced into the system prompt
  (`format_tools_for_prompt`, `base_env.py:96-127`). None of the nine primitive
  names collide with `RESERVED_TOOL_NAMES` (`base_env.py:13-24`), so
  `validate_custom_tools` passes.
- **`environment="local"` is mandatory for `custom_tools` — CONFIRMED.**
  `DockerREPL.__init__` (`docker_repl.py:195-204`) takes no `custom_tools`
  param — it is swallowed by `**kwargs` and never injected; `_build_exec_script`
  (`docker_repl.py:158`) builds the in-container globals without it. Only
  `LocalREPL` injects custom tools. See **I6**.
- **`FINAL_VAR(name)` / `FINAL(text)` termination — CONFIRMED.**
  `rlm/utils/parsing.py:890-931` (`find_final_answer`) matches both, anchored to
  start-of-line; `core/rlm.py:351-361` prefers the REPL `final_answer` from a
  `FINAL_VAR` call (`LocalREPL._final_var`, `local_repl.py:208-232`) and falls
  back to text parsing.
- **The `rlm.core.rlm.get_client` monkeypatch seam — CONFIRMED correct and
  stable.** `core/rlm.py:7` does `from rlm.clients import … get_client`, binding
  the name in the `rlm.core.rlm` module namespace; `_spawn_completion_context`
  (`core/rlm.py:198`) calls the module-global `get_client`. Patching
  `rlm.core.rlm.get_client` (as `test_integration_custom_tools.py:53-54` does)
  intercepts every client construction. The integration test mocks **only** the
  LLM — the real `RLM`, real `LocalREPL`, real `custom_tools` injection and real
  `extract_hyperparameters` primitive all run. It is a genuine integration test,
  not a stubbed one.
  *Coverage gap:* it exercises only a heuristic (non-LLM, non-async) primitive
  inside the real REPL. No test runs an async-bridge primitive
  (`build_environment`/`implement_baseline`/`run_experiment`) or an LLM
  primitive *from inside the real `rlm` REPL* — those are unit-tested in
  isolation only. The `ThreadPoolExecutor + asyncio.run`-in-a-worker-thread
  bridge is correct (a fresh thread never has a running loop, so `asyncio.run`
  always succeeds, in both the sync test harness and Phase 3's REPL), but its
  behavior inside the real REPL is currently unverified by test.

## Empirical verification

- `.venv/bin/python -m pytest tests/rlm/ -q` → **31 passed**.
- Full suite `tests/` → **909 passed, 2 failed, 2 skipped** (912 collected).
  The 2 failures —
  `test_issue17_runtime.py::test_local_process_backend_executes_commands_with_artifact_env`
  and
  `test_issue26_experiment_runner.py::TestRunWithRuntime::test_local_process_mode_runs_without_dockerfile`
  — are **not Phase 2 regressions**: Phase 2 touches neither those test files
  nor `backend/services/runtime/` (`git diff 63adb10..HEAD` confirms), and both
  fail with a local-process-backend `FileNotFoundError`
  (`artifacts/metrics.json` not produced) — a pre-existing local-environment /
  runtime-backend issue, unrelated to the primitive layer. The 2 skips are the
  usual optional-`chromadb` skips.
- `rlm` 0.1.1 source read directly at
  `.venv/lib/python3.14/site-packages/rlm/` for every API claim above.
