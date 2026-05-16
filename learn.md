# learn.md — bugs we shipped and what we changed so we don't ship them again

This is a runbook of post-mortem entries for production-shaped bugs in the
OpenResearch agent stack. Each entry is short and follows the same shape:

> **Symptom → Root cause → Fix → Lesson → Guardrail (test or pattern)**

Add a new entry to the top of the list. Keep entries surgical: one bug per
section, no broad essays. If a class of bug recurs, escalate it to a section
in **Cross-cutting principles** below.

---

## 2026-05-14 — A "successful" docker build can still ship a broken environment

**Symptom.** A docker+sonnet e2e run on the demo PPO paper sailed past Track 4
(env built clean attempt 1), got past the sandbox-mount contract fix (script
wrote correctly to `$OUTPUT_DIR`), and died at the very first `gym.make`:
`ModuleNotFoundError: No module named 'imageio'` from
`gymnasium/envs/mujoco/mujoco_rendering.py`. The Dockerfile pinned
`gymnasium[mujoco]` but `imageio` isn't in gymnasium's `setup.py` for the
mujoco extra — it's imported at first env load. The build had no way to know.

**Root cause.** Track 4 validates that `docker build` *exits 0*. That only
proves "pip install didn't crash" — not "every Python module in this image
actually imports." Lots of pip packages have *transitive runtime imports*
(here: `imageio` for gymnasium-mujoco; in other papers, `cv2` wanting
`libGL.so.1`, torch wanting a specific CUDA runtime, etc.) that pass at
install time and explode at module-load time. Track 4 had no trigger for
runtime-import failures — so they died downstream at `baseline_run`, with
no repair feedback.

**Fix.** Force the runtime-import failure into the build phase by making
the FINAL Dockerfile layer a no-network smoke: `RUN python -c '<imports + a
minimal instantiation of the paper's primary entity>'`. A failure there is
a build failure — and Track 4's build-and-repair loop already knows how to
fix build failures (add the missing dep via env-detective repair mode,
rebuild). Zero new code; just a prompt rule.

**Lesson.** A repair loop's *trigger event* is as load-bearing as its
*repair mechanism*. Track 4 had the repair mechanism (env-detective in
repair mode) but only fired it on `docker build`'s exit code. The
import-time class of failures was invisible to that trigger. The
generalization: whenever you have a recovery mechanism, audit *every*
failure mode it should cover and make sure each has a trigger pointing at
it — not just the one that motivated building the mechanism in the first
place.

**Guardrail.** No new tests — the smoke layer is verified by the e2e run
(if it's missing, the prompt change is in but the agent ignored it; the
existing prompt-format tests catch import + brace regressions). The next
demo-paper run is the live regression check — the imageio failure should
now appear at *build* time, get repaired automatically, and the experiment
should reach `baseline_run` with a working environment.

## 2026-05-14 — The sandbox mount contract lived in env-var names, not in any prompt

**Symptom.** A docker+sonnet e2e run on the demo PPO paper sailed past Track 4's
environment build (clean attempt 1), reached `baseline_run`, and died at the very
first command: `mkdir: cannot create directory '/work/results': Read-only file
system`. Gate 2 halted on `failed_reproduction`. The reproduction *code* was
fine — the failure was that the script tried to write outputs under the project
mount.

**Root cause.** The sandbox runtime enforces a clear mount contract — project
read-only at `/work`, writable artifact volume at `$OUTPUT_DIR` — but this
contract existed only implicitly, in env-var names exposed to the container.
The `baseline-implementation` and `improvement-path` prompts never stated it,
so the agent wrote scripts assuming the CWD was writable. A load-bearing
contract that lived only in the runtime's env-var dictionary was advisory to
the agent, not enforced.

**Fix.** Made the contract a first-class artifact. `backend/agents/prompts/_sandbox_contract.py`
defines a single brace-free `SANDBOX_EXECUTION_CONTRACT` block — the mount
model, the env vars, the required write patterns (every output under
`$OUTPUT_DIR`; cache-hungry tools redirected; metrics.json path pinned). It is
imported and spliced into every agent prompt that emits sandbox-executable code
(`baseline-implementation`, `improvement-path`, `composition`), positioned
right before the `# Output` section at peak attention. Identical across docker,
local, and runpod — same env vars, same model.

**Lesson.** An interface contract between code that *generates* artifacts (an
LLM agent) and code that *executes* them (the runtime) must be stated in the
generator's prompt, not just enforced by the executor. Same lesson as
"a 'hard cap' in a prompt is advisory unless enforced in code" — but in the
other direction: a runtime invariant the agent must respect is advisory
unless stated in the prompt. Put it in one shared module, splice it where it
matters, and the prompts cannot drift from the runtime.

**Guardrail.** `tests/test_track4_environment_build_repair.py` is unaffected;
the contract is verified by a focused import-and-format assertion in
`backend/agents/prompts/__init__.py`'s consumers and by every existing prompt
test that imports the three updated prompts. The next e2e run on demo_paper.pdf
is the live regression check.

## 2026-05-14 — The reproduction Dockerfile was never built until it was too late to fix

**Symptom.** `environment-detective` generated the Dockerfile one-shot at the
`ENVIRONMENT_BUILT` stage, but nothing ran `docker build` until `run_experiment`
at `BASELINE_RUN` — five stages and tens of minutes later. A broken Dockerfile
(missing system lib, a non-existent pin like `ale-py 0.8.1`, base-image
mismatch) burned all that work, then dead-ended the run at Gate 2 with
`blocked_requires_human`. No run had ever reached the Track 3 flow live.

**Root cause.** The pipeline had a *judge* for the environment
(`environment-verifier` at Gate 1) but no *builder*. The first real validation
of the generated artifact happened far downstream from where it was produced,
so the feedback loop that could fix it never existed — and the terminal state
for that failure was a human-required halt, not an autonomous recovery.

**Fix.** Build the Dockerfile at the stage that produces it. A build-only
`build_image()` primitive runs `docker build` at `ENVIRONMENT_BUILT`; on failure
the build error is fed back to `environment-detective` in a repair mode and the
build is retried, hard-capped at `environment_build_max_attempts`. After the cap
the run is **fail-soft** — it proceeds and completes with an honest
partial-reproduction verdict instead of halting for a human.

**Lesson.** Validate a generated artifact at the stage that generates it, not at
the stage that first consumes it — the distance between the two is wasted time
and a feedback loop you don't have. And an autonomous pipeline's terminal state
for a *recoverable* failure should be an honest verdict, not a halt: a bounded
repair loop plus fail-soft beats `blocked_requires_human`.

**Guardrail.** `tests/test_track4_environment_build_repair.py` — `build_image`
returns `(False, …)` for a broken Dockerfile but raises for an infrastructure
failure; `_run_environment_build_loop` is bounded (capped attempts, repair
invoked between them) and fail-soft (cap spent → `environment_build_ok` false,
no raise).

## 2026-05-14 — A "hard cap" that lived only in a prompt was advisory, not enforced

**Symptom.** The rubric-verifier prompt told the model "no executable code →
score ≤ 0.20", "code never ran → ≤ 0.35", etc., and the plan/changelog called
these "honesty hard caps" — but nothing checked them. A model that returned 0.9
for a run that never executed would be accepted verbatim.

**Root cause.** Load-bearing invariants were expressed *only* as natural-language
instructions to an LLM. A capable model usually follows them, but "usually" is
not a guarantee, and the reported score is a metric users act on.

**Fix.** Added a mechanical backstop in `_run_rubric_verifier`: the orchestrator
already knows `experiment_artifacts.success`, so when the reproduction did not
execute it clamps every area score before aggregation — independent of what the
model returned. The prompt still states the caps (so the model cooperates).

**Lesson.** A guarantee a prompt makes is only as strong as the model's
compliance. If an invariant is load-bearing — a safety gate, a reported metric,
a stopping criterion — enforce it in code at the boundary; let the prompt *also*
state it, not *only* state it.

**Guardrail.** `tests/test_rubric_verifier.py::test_run_rubric_verifier_caps_score_when_run_did_not_succeed`
feeds a high model score for a failed run and asserts it is capped.

## 2026-05-14 — A self-improvement loop compared scores from regenerated rubrics

**Symptom.** The rubric verifier ran at Gate 2 and Gate 3, and the re-iteration
loop stopped when `improved_verification.overall_score` met the target — but the
baseline and improved verifications were not actually comparable.

**Root cause.** Each checkpoint created a fresh `GeneratedRubricSource()` and
passed `rubric: null`, so the verifier LLM generated *new* areas and weights
every time. `baseline_verification` and `improved_verification` were scored
against different rubrics; their delta — and the loop's stop criterion —
measured rubric churn, not reproduction progress.

**Fix.** Resolve the canonical rubric once per run (a vendored bundle's rubric,
or LLM-generated on the first call), persist it in `PipelineState.rubric_spec`,
and pass it back at every later checkpoint. Weights come from the persisted
spec; the LLM supplies per-area scores only.

**Lesson.** A metric you compare across time must be *defined* once. If the
judge is free to redefine the rubric at each measurement, the series of scores
is not a series — it is noise wearing a trend's clothes.

**Guardrail.** `tests/test_rubric_verifier.py` asserts the first verifier call
persists `rubric_spec` and a later call reuses its weights verbatim — a model
that returns different weights is overridden, not trusted.

---

## 2026-05-14 — A `backend.agents` module eager-importing `backend.evals` was a circular import

**Symptom.** Adding `from backend.agents.rubric_source import GeneratedRubricSource`
to `backend/agents/orchestrator.py` broke *every* import of the orchestrator:
`ImportError: cannot import name 'PipelineState' from partially initialized
module 'backend.agents.orchestrator'`.

**Root cause.** `rubric_source.py` had a module-level
`from backend.evals.paperbench.bundle import ...`. Importing any
`backend.evals.*` submodule runs `backend/evals/__init__.py`, which eagerly
imports `backend.evals.runner` → which imports `backend.agents.orchestrator`.
While `orchestrator` was *mid-import* (at the new `rubric_source` line, before
`PipelineState` was defined), `runner` tried to import `PipelineState` from it.
Phase A didn't hit this because nothing in the main import graph pulled in
`rubric_source` — only the tests did, and by then `orchestrator` was complete.

**Fix.** Made `rubric_source.py` import the `bundle` loader **lazily**, inside
the two functions that actually load a bundle. The cycle is broken because by
call time `orchestrator` is fully initialized.

**Lesson.** A package `__init__.py` that eagerly imports heavy submodules turns
*every* `from that_package.x import y` into a transitive import of the whole
package graph. A leaf-looking module (`bundle.py` only imports stdlib) is not
leaf if its package `__init__` is not.

**Guardrail.** A `backend.agents.*` module that needs `backend.evals.*` (or any
package whose `__init__` reaches back into `backend.agents`) imports it lazily
inside the function that needs it — never at module scope.

---

## 2026-05-14 — A timed-out enrichment frame silently blanked the live graph

**Symptom.** Mid-run, the workflow graph's per-path improvement nodes
(`opt/bb/aug/hor/div`) intermittently dropped back to "upcoming" for a tick,
then recovered on the next frame.

**Root cause.** Both `/api/demo` GET (750 ms) and `/api/demo/events` SSE
(250 ms) cap payload enrichment and, on timeout, forward the *un-enriched*
backend run state — which carries no `payload`. `stateMapForRun` reads
`run.payload.pathStates`; with `payload` undefined every path node fell
through to "upcoming". The UI overwrote good state with a strictly poorer
frame.

**Fix.** `coalesceRunState` merges an incoming `run_state` frame onto the
current one, carrying the last `payload`/`telemetry`/`log` forward when the
new frame lacks them. Both the SSE handler and the poll fallback route
through it; it warns in dev when it has to coalesce.

**Lesson.** A frame that arrives with *less* information than the one it
replaces must not be applied verbatim — partial frames are an expected
steady-state condition here (enrichment timeouts), not an error.

**Guardrail.** State updates fed from a stream/poll should be **monotonic in
information**: merge-don't-replace when the transport can legitimately
deliver a degraded frame. (`stateMapForRun` already encoded this for stage
progress; `coalesceRunState` extends the same rule to the payload.)

## 2026-05-14 — A stage-ordering test froze the pipeline at 15 stages after it became 14

**Symptom.** `tests/test_issue22_orchestrator.py::test_pipeline_stages_are_ordered`
failed on `claw_demo` (and on its parent commit): the test's `expected_order` placed
`composition_tested` between `improvements_run` and `gate_3_passed`; the real
`PipelineStage` enum had no such stage.

**Root cause.** `composition_tested` was removed from `backend/agents/orchestrator.py`
when the pipeline became 14 stages, but the ordering test still hard-coded the old
15-stage list — it re-typed the enum as a literal and then drifted from it.

**Fix.** Dropped `"composition_tested"` from the test's `expected_order`.

**Lesson.** A test that re-types an enum as a literal sequence is a second source of
truth; it goes stale silently the moment the enum legitimately changes.

**Guardrail.** Derive the expectation from the enum (`[s.value for s in PipelineStage]`)
and assert the *properties* that matter (no gaps, each gate after its prerequisites,
`complete` last) instead of re-typing the sequence.

## 2026-05-10 — Pipeline SIGINT dumped a 50-line stack trace and left status="running"

**Symptom.** Killing the `python -m backend.cli reproduce` subprocess (Ctrl-C
or backend restart) produced a noisy traceback in `runner.stderr.log`:

```
asyncio.exceptions.CancelledError
…
File "/home/abheekp/openresearch/backend/cli.py", line 485, in cmd_reproduce
    state = asyncio.run(run_pipeline_sdk(
KeyboardInterrupt
```

The dashboard meanwhile showed `status="running"` until the user hit `/lab`
again, at which point `live_runs._load_run` detected the dead PID via
`_pid_exists` and rewrote status to `failed` with whatever string the log
heuristic happened to extract — usually misleading.

**Root cause.** The application code had **zero** explicit handlers for
`KeyboardInterrupt` or `asyncio.CancelledError`:

1. `cli.py:485` wrapped `asyncio.run(run_pipeline_sdk(...))` with only
   `except Exception` (catches `BudgetExhausted`). `BaseException`
   subclasses fell through, which is correct Python convention but meant
   we never got a chance to write a clean status before exiting.
2. `orchestrator.py:1441` step loop's `except Exception` likewise didn't
   catch `CancelledError`. The "X FAILED:" line never printed for
   cancellation either, so the log just stopped mid-stage with no
   actionable signal.
3. `live_runs._write_status` wrote `demo_status.json` non-atomically, so
   a crash during a status write could leave a half-written JSON that
   `_read_status` then failed to parse. Compounding the original
   interrupt with a corruption bug.

**Fix.**

- `cli.py` catches `(KeyboardInterrupt, asyncio.CancelledError)` around
  `asyncio.run(run_pipeline_sdk(...))`, prints a single readable line,
  calls `_mark_demo_status_stopped()` to flip the status to `stopped`
  with a descriptive `error` field, and exits 130 (SIGINT convention).
  No more stack-trace dumps.
- `orchestrator.py:1431` step loop now catches cancellation **before**
  the generic `except Exception`, prints `|| STOPPED at <stage>`, calls
  `state.save_checkpoint(self.runs_root)` so a future
  `reproduce --resume` picks up from the last completed stage, and
  re-raises so the CLI's outer handler runs.
- `cli._atomic_write_json` (and the equivalent in
  `live_runs._write_status`) writes via tempfile + `os.replace` so
  `demo_status.json` is never half-written. Readers always see either
  the previous valid JSON or the new one.

**Lesson.** **`asyncio.CancelledError` is a `BaseException`, not an
`Exception` — your `except Exception` does NOT catch it.** Long-running
async pipelines need an explicit `(asyncio.CancelledError, KeyboardInterrupt)`
handler at every layer that owns persistent state, before the generic
`except Exception` clause. The handler should: (1) log a clean message,
(2) flush partial state to disk so resume works, (3) re-raise so callers
above can do their own cleanup. Status files that record run lifecycle
should be written atomically (`tempfile.write_text` + `os.replace`) so a
crash during the write doesn't corrupt the file the dashboard is about
to read.

**Open edge cases (documented, not yet fixed):**
- Concurrent runs on the same `project_id` will race on
  `demo_status.json`, `pipeline_state.json`, and `runs/{project_id}/*`.
  Atomic writes prevent corruption but don't prevent overwrite.
- SIGKILL bypasses the CLI's interrupt handler entirely — the pipeline
  dies, any orphaned ephemeral runpod sandbox stays running until
  someone (or `_owned_pod_ids` reconciliation on the next backend
  restart) kills it. Persistent pods (`REPROLAB_RUNPOD_POD_ID`) are
  unaffected.
- Single-worker uvicorn (`--reload`) blocks all other endpoints behind
  one slow SSE stream. The frontend already mitigates this with SSR +
  proxy + client-poll timeouts (`lab/page.tsx`, `api/demo/route.ts`,
  `live-demo-client.tsx`); the durable fix is multi-worker uvicorn or
  an ASGI server with proper concurrency.

**Guardrail.**
- The `(asyncio.CancelledError, KeyboardInterrupt)` handler in
  `cli.py:cmd_reproduce` is the single chokepoint where pipeline runs
  exit. Future async entrypoints (CLI subcommands, scheduled jobs)
  should follow the same shape: catch cancellation FIRST, write status,
  return 130, then `except Exception` for anything else.
- `_atomic_write_json` / `_write_status` use the canonical
  tempfile+replace pattern. New status writers should reuse one of
  these helpers, not write directly.
- `orchestrator.py:1431` has the per-step cancellation guard. Stages
  added to the pipeline list inherit it for free.

---

## 2026-05-10 — Runpod smoke trap destroyed a pod we wanted to keep

**Symptom.** Running `START_FULL_SMOKE=1 ./start.sh` to verify Runpod
end-to-end booted pod `nfh9zaeetfubv0` (RTX 4090 SECURE, $0.69/hr) — exactly
what we wanted. When we SIGTERM'd the script mid-boot, we were about to lose
the pod even though we hadn't gotten our verification yet. Separately, when
the user later asked "can we just use my coworker's pod that's already on
the account?", the answer was "the smoke flow has no concept of that — it
always creates and destroys its own."

**Root cause.** Two design assumptions in the Runpod tooling collided with
the actual workflow:

1. `scripts/runpod_check.sh` installs `trap cleanup_pod EXIT` immediately
   after pod creation (line 361). The trap issues a raw `curl -X DELETE`
   against `/pods/${POD_ID}`, **bypassing** the `RunpodBackend._owned_pod_ids`
   allowlist + `reprolab-` name-prefix guard that protects coworker pods on
   the same account. The trap is correct for its designed purpose
   (boot → nvidia-smi → tear down, never leak money on failure), but
   incompatible with "boot a pod and keep it."
2. `RunpodBackend.delete_on_destroy` defaults to `True` (config.py:89), so
   even pods created via the dashboard get deleted after each run unless
   `.env` overrides it. There is no first-class "attach to existing pod"
   mode — every `create_sandbox` call hits `POST /pods`.
3. The May 2026 REST v1 API has no GPU-listing endpoint, so the only way to
   know whether a 4090 is bookable is to actually book one. That pushes
   teams toward `--start-pod`-style smokes, which then collide with point 1.

**Fix.**

- For *auth + key* verification only: `./scripts/runpod_check.sh` with **no
  flag**. Free, no pod boot, no trap risk. This is what `start.sh` runs by
  default before booting uvicorn.
- For *first-time GPU bookability* verification: `--start-pod` is fine
  **provided you let the trap finish naturally**. SIGKILL bypasses the trap
  and leaks a pod; SIGTERM lets the trap fire and destroys the pod. Neither
  is what you want if you intend to keep using the pod afterwards.
- For *persistent pod usage* (the real workflow): set
  `REPROLAB_RUNPOD_DELETE_ON_DESTROY=false` in `.env`. The dashboard /
  `--sandbox runpod` flow will then leave pods running after each pipeline
  finishes. Reuse a coworker's pod by adding their public key to your local
  `REPROLAB_RUNPOD_SSH_PUBLIC_KEY` — RunPod injects it via `PUBLIC_KEY` env
  var on `runpod/*` images, no custom start command needed.
- For *single-pod reuse across runs* (skip per-run boot, attach to a fixed
  worker): set `REPROLAB_RUNPOD_POD_ID=<pod-id>` in `.env`. The backend
  fetches the pod, attaches via SSH, and reuses it for every pipeline run.
  The pod is structurally undeletable — never added to `_owned_pod_ids`,
  so `_delete_pod` refuses. If the configured pod is missing or stopped,
  the backend creates a new persistent pod and logs the new id at WARNING
  (`RUNPOD_PERSISTENT_POD_CREATED pod_id=…`); update `.env` with that id
  to reuse it on subsequent runs. Constraint: this assumes one pipeline
  run at a time on the shared pod (the `/workspace/work` symlink is
  per-pod, not per-run).
- The `_owned_pod_ids` allowlist + `reprolab-` name-prefix check in
  `runpod_backend.py:_delete_pod` already prevents the backend from deleting
  any pod it didn't create itself (defense against logic bugs and shared-
  account accidents). That guard is the *only* thing protecting your
  coworker's pods if they share a Runpod account with you.

**Lesson.** **A "smoke test" that boots real paid infrastructure is two
features in a trench coat, and they fight.** The cleanup-on-failure trap is
correct for "did this work end-to-end, free if not," and wrong for "boot
something I want to keep." Don't try to repurpose one for the other by
killing the script with the right signal — that's spell-casting, not
engineering. When the workflow shifts from "verify + tear down" to "verify +
keep," take a different code path: skip the smoke, set
`DELETE_ON_DESTROY=false`, and let the backend's normal create-sandbox flow
do the booking with the real safeguards (`_owned_pod_ids`, name prefix)
intact.

May 2026 Runpod REST v1 facts worth remembering so we don't drift:
- Endpoint: `POST https://rest.runpod.io/v1/pods`
- Auth: `Authorization: Bearer <key>` (key prefix is `rpa_…`)
- Payload uses `gpuTypeIds: ["NVIDIA GeForce RTX 4090"]` (plural array form,
  per the docs' curl examples). The OpenAPI schema lists `gpuTypeId`
  singular, but the live API accepts the plural array — match the curl
  examples, not the schema.
- `ports: ["22/tcp"]` — string form with protocol suffix.
- Official `runpod/*` images read `PUBLIC_KEY` (and `SSH_PUBLIC_KEY`) env
  vars automatically; do **not** override `dockerStartCmd` for them or you
  will lose RunPod's own SSH bootstrap. Custom `dockerStartCmd` is only
  needed for third-party images (handled in
  `runpod_backend.py:_runpod_start_command`).
- REST v1 has no GPU-listing endpoint. Fail-on-creation is the only signal
  that a configured GPU type isn't bookable on your account/region.

**Guardrail.**
- `RunpodBackend._owned_pod_ids: set[str]` (`runpod_backend.py:98`) is
  populated only on backend-created pods, and `_delete_pod` refuses to issue
  DELETE for any pod ID outside that set. Coworker's pods on the same
  account are structurally unreachable from the backend's delete path.
- `_delete_pod` belt-and-suspenders: even if a pod ID ended up in the
  allowlist via some future code path, the pod's name must start with
  `reprolab-` or DELETE is refused (`runpod_backend.py:444-449`).
- `.env` documents `REPROLAB_RUNPOD_DELETE_ON_DESTROY` and recommends
  `false` for shared-pod workflows. The default (`true`) stays as-is so
  one-off runs still clean up.
- `start.sh` runs the **free** preflight by default; `START_FULL_SMOKE=1`
  is opt-in only. Never make the paid smoke the default — money + traps =
  silent footguns.

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
