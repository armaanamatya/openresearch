# RLM Phase 3 — Primitive Integration: Design Spec

| | |
|---|---|
| **Issue** | [#60](https://github.com/armaanamatya/openresearch/issues/60) — primitive integration (extract stage agents) |
| **Date** | 2026-05-20 · **v3: 2026-05-21** |
| **Status** | Draft **v3** — rebuilt on the `rlms` library after the 2026-05-21 brief rewrite; pending review |
| **Branch** | `rlm-pivot` @ `63adb10` — = the head of Armaan's PR [#65](https://github.com/armaanamatya/openresearch/pull/65) (open, `rlm-pivot → main`); verified current 2026-05-21 |
| **Canonical spec** | `docs/design/rlm-pivot-brief.md` (rewritten 2026-05-20 — "RLM Pivot Plan"), `docs/rlm-pivot-mapping.md` |
| **Engine** | `rlms` v0.1.1 (PyPI; **import name `rlm`**) — studied 2026-05-21, all API claims grounded in that recon |

> **Revision history.**
> **v1** — first design. **v2** — Codex adversarial review (`task-mpf0tfzv-49zj8b`); 15 findings
> fixed. **v3** — the 2026-05-21 `git pull` brought a rewritten brief that **changed the
> architecture**: the RLM engine is no longer hand-built — the project depends on the
> **`rlms` PyPI library**. v3 rebuilds the framework (§3–§5, §9–§11) on the library's real
> API; the per-primitive specs (§6) and cross-primitive contracts (§7–§8) carry forward
> from the Codex-reviewed v2 ~unchanged. An **Opus analytical pass (2026-05-21)** then
> tightened v3 — resolved the `environment` local-vs-docker contradiction (§4), defined
> `CorpusFingerprints`, and closed cost-accounting, threading, and test-harness gaps.
>
> **Phase-label note.** The rewritten brief renumbers the build order — primitive
> integration is now its **"Phase 2."** GitHub issue #60 still says "Phase 3." Same work;
> a label drift the team may want to reconcile on the issues. This spec covers the
> primitive-integration work for issue #60 regardless of label.

---

## 1. Summary

Issue #60 delivers the **nine domain primitives** of the RLM reproduction system: stateless
functions the RLM root model calls from inside its REPL. They are assembled into the
`custom_tools` dict passed to the `rlms` library's `RLM(...)`.

The design is one abstraction: nine **primitive cores**, one `_adapt` higher-order wrapper
carrying every cross-cutting concern, and one `build_primitive_namespace(ctx)` closure
factory whose output **is** the `custom_tools` dict. `rlms` provides the REPL host, the
Algorithm-1 loop, and `llm_query`/`rlm_query` — we do not build them. We provide the
domain layer.

The legacy 14-stage pipeline keeps working throughout (brief §11 — keep `main` runnable).
Domain logic is extracted into shared helpers that **both** the legacy stage agent and the
new primitive call.

---

## 2. Context & locked decisions

The 2026-05-20 brief rewrite (`rlm-pivot-brief.md`, "RLM Pivot Plan") sets the architecture:
**depend on the `rlms` library** (`pip install rlms`) — the RLM authors' published
Algorithm-1 implementation — rather than hand-build a REPL host / root loop / sub-calls.
The engine choice was **reconfirmed by the user on 2026-05-21** — weighing DSPy 3 /
`dspy.RLM` (rejected: its WASM interpreter cannot run Docker; issue #66, closed) and
hand-build (retired) against the `rlms` library — see issue #64's resolution.
The brief's build order: Phase 1 spike (`pip install rlms` + a minimal `RLM` run) →
**Phase 2 primitives (= this spec, issue #60)** → Phase 3 orchestrator + `system_prompt.py`
+ `run.py` → Phase 4 frontend → Phase 5 end-to-end → Phase 6 cleanup.

**Three decisions locked by the user (2026-05-20), retargeted onto `rlms`:**

1. **Contract-first parallel build.** This spec defines and freezes the seam (§5) between
   `primitives.py` (#60) and the `RLM`-wiring work (`run.py`, brief Phase 3); the nine
   cores are built and unit-tested now.
2. **Extract to shared helpers.** Logic inside legacy stage agents and `orchestrator.py`
   is extracted into shared helpers both the legacy caller and the new primitive use.
3. **Acceptance gate = tests + an `rlms` integration harness** (§9) — now a *real*
   `RLM(custom_tools=…)` run on a mock context, stronger than v2's planned fake harness.

### In scope
- The nine primitive cores + the framework (§4); the seamless extension of the Phase 1
  `primitives.py` (§3).
- Extraction of shared helpers from legacy stage agents and `orchestrator.py` (§6, §10).
- One behavior addition: `build_environment` retains + registers its built image so
  `env_id` is real (§6.7).
- The `env_id` contract, input dict schemas, the error matrix (§7, §8).
- Adding `rlms` to the project dependencies; the test suite + the `rlms` integration
  harness (§9).

### Out of scope
- The `rlms` library itself — a pinned dependency, not our code.
- `run.py`, `system_prompt.py`, the SSE bridge, `final_report` writing — brief Phase 3.
- `set_final` — **superseded.** `rlms` provides a built-in `FINAL_VAR` reserved function
  for termination (§4.6); a `set_final` custom tool is unnecessary. The Phase-1 stub for
  it in `primitives.py` is left for the brief-Phase-3 work to remove.
- Frontend; the live PaperBench run (brief Phase 5); Phase 6 cleanup.
- `run_path_offline`, artifact discovery — confirmed no primitive (mapping doc §1).

### Non-destructive guarantee (brief §11 — keep `main` runnable)
Every helper extraction is behavior-preserving: the legacy function becomes a thin wrapper
over the new shared helper. Each extraction is gated by the legacy module's existing test
suite staying green; thin coverage gets a characterization test added first.

---

## 3. Module layout — extension of the Phase 1 skeleton

`rlms` is a **new pinned dependency** (`pip install rlms`; pin `rlms==0.1.1`; import name
is `rlm`). It is added to the project's dependency manifest as part of Step 0.

Armaan's PR **#65** (`rlm-pivot → main`, open, head `63adb10`) is the integration point:
its five commits carry the Phase 1 `backend/agents/rlm/` skeleton (six files, unchanged
since commit `e92e11e`), `docs/rlm-pivot-mapping.md`, and the 2026-05-20 doc/brief rewrite.
#60's work continues on `rlm-pivot` on top of #65 — becoming part of it, or a follow-up PR
if #65 merges first. Phase 3 **edits one skeleton file in place** and **adds new sibling
files**:

```
backend/agents/rlm/
  primitives.py          [Phase 1 — EDITED in place] façade: keeps PRIMITIVE_REGISTRY;
                         fills the 9 stub bodies (delegate to primitive_impl/); adds
                         build_primitive_namespace. set_final stub left for brief-Phase-3.
  primitive_context.py   [NEW] PrimitiveContext, PrimitiveEvent, EnvironmentRegistry, AgentInvoker
  primitive_adapter.py   [NEW] _adapt, build_primitive_namespace, _guard_no_corpus, error types
  primitive_bridge.py    [NEW] _run_sync, _invoke_llm
  primitive_impl/        [NEW] the nine primitive-core functions
    understanding.py     — understand_section, extract_hyperparameters
    detection.py         — detect_environment
    verification.py      — verify_against_rubric
    planning.py          — plan_reproduction, implement_baseline
    environment.py       — build_environment
    execution.py         — run_experiment
    improvement.py       — propose_improvements
  system_prompt.py       [Phase 1 — UNTOUCHED by #60; brief-Phase-3 fills it]
  __init__.py            [Phase 1 — UNTOUCHED by #60 — see note below]
  repl_host.py           [Phase 1 — SUPERSEDED by rlms — see note below]
  root_loop.py           [Phase 1 — SUPERSEDED by rlms]
  sub_call.py            [Phase 1 — SUPERSEDED by rlms]
```

**Superseded Phase-1 files.** The rewritten brief replaces the hand-built RLM engine with
`rlms`, so `repl_host.py` / `root_loop.py` / `sub_call.py` are dead code. **#60 neither
uses nor deletes them** — deletion (and reconciling `__init__.py`, which currently
re-exports symbols from them) belongs to the `rlms`-adoption work (brief Phase 1/3) or
Phase 6. #60's only Phase-1-file edit is `primitives.py`. This is the PR-#65 contradiction
surfaced cleanly: #65 hand-built an engine the rewritten brief no longer wants — flag it
on #65; do not let #60 depend on those three files.

**New shared-helper modules** (extracted logic, called by both legacy + primitive):
`backend/agents/reproduction_planner.py`, `backend/agents/environment_build.py`,
`backend/agents/rubric_verifier.py`.

**Legacy modules edited (behavior-preserving extraction only):** `paper_understanding.py`,
`environment_detective.py`, `baseline_implementation.py`, `experiment_runner.py`,
`orchestrator.py`.

---

## 4. The framework — built on the `rlms` API

`rlms` v0.1.1 facts this design depends on (all from the 2026-05-21 library recon):
- `from rlm import RLM`; `RLM(...).completion(prompt) -> RLMChatCompletion` — **synchronous**,
  blocks the calling thread; runs `exec` of root-written code **on that calling thread**.
- `custom_tools: dict` — `{name: callable}` or `{name: {"tool": callable, "description": str}}`;
  callables become REPL functions called **exactly as the root writes them, synchronously,
  with no library-injected context**; the `description` feeds the auto-generated
  system-prompt tool section.
- Reserved REPL names (collision → `ValueError` at construction): `llm_query`,
  `llm_query_batched`, `rlm_query`, `rlm_query_batched`, `FINAL_VAR`, `SHOW_VARS`,
  `context`, `history`.
- Only `environment="local"` reliably supports callable `custom_tools` (isolated
  environments cannot serialize callables).

**Two environments — do not conflate.** ReproLab has two distinct "environments," and the
`rlms` REPL is *not* the one that runs Docker:
- The **`rlms` REPL environment** = where the root model's `exec` runs. It is
  `environment="local"` (host-side) — required, because our domain primitives are Python
  *callables* passed via `custom_tools`, and `rlms` cannot serialize callables into an
  isolated REPL.
- The **experiment sandbox** = where reproduced code is built and run. That is Docker,
  driven *inside* the `build_environment` / `run_experiment` primitives via our own
  `RuntimeAppService` (`ctx.sandbox_runtime`).

This reconciles issue #64's "`environment='docker'` was decisive": the decisive point was
that `rlm` can drive Docker reproduction work *at all* (vs. `dspy.RLM`'s WASM sandbox,
which cannot) — not that the `rlms` REPL itself is containerized. **Verification item
(Step 0):** confirm against `rlms` that callable `custom_tools` work under
`environment="local"` and that nothing forces the REPL to `"docker"`.

### 4.1 `primitive_context.py` — the run-scoped context (the seam)

```python
class AgentInvoker(Protocol):
    """The orchestrator's resilient agent path. Carries provider failover, RunBudget,
    provider-health, telemetry, cost-ledger accounting, runtime spec — everything
    orchestrator._invoke_agent already does (orchestrator.py:573, 647-655)."""
    async def __call__(self, agent: str, prompt: str, *,
                       model_override: str | None = None) -> str: ...

@dataclass(frozen=True)
class PrimitiveContext:
    project_id: str
    project_dir: Path
    invoke_agent: AgentInvoker                # bound to orchestrator._invoke_agent by run.py
    sandbox_runtime: RuntimeAppService        # pre-bound backend (docker/runpod/local)
    event_loop: asyncio.AbstractEventLoop     # the orchestrator loop, kept live by run.py
    emit: Callable[[PrimitiveEvent], None]    # run.py supplies; MUST be worker-thread-safe
    environments: EnvironmentRegistry         # run-scoped env_id ledger (mutable)
    corpus_fingerprints: CorpusFingerprints   # for the Algorithm-2 guard (§7.4)
    settings: PrimitiveSettings               # sandbox mode + timeout/attempt caps

@dataclass(frozen=True)
class PrimitiveEvent:
    primitive: str
    status: Literal["running", "completed", "failed"]
    args_summary: str                         # bounded metadata — never the corpus
    result_summary: str | None = None
    duration_ms: int | None = None
    error: str | None = None

@dataclass(frozen=True)
class CorpusFingerprints:
    """SHA-1 of each offloaded corpus value + their total byte size — captured by run.py
    at run init; the reference set the Algorithm-2 guard (§7.4) checks primitive args
    against."""
    hashes: frozenset[str]
    total_corpus_bytes: int
```

`AgentInvoker` is the one cohesive LLM capability — primitives that call agents inherit
every production safeguard with zero re-implementation. `PrimitiveContext` is frozen; only
`EnvironmentRegistry` contents mutate. `rlms` injects nothing into `custom_tools` callables,
so the context reaches each core by **closure binding** in the factory (§4.2) — exactly
the mechanism `rlms`'s API leaves to the caller.

### 4.2 `primitive_adapter.py` — the wrapper + the `custom_tools` factory

```python
def _adapt(name: str, core: Callable, ctx: PrimitiveContext) -> Callable:
    @functools.wraps(core)
    def primitive(*args, **kwargs):                        # sync — rlms calls it sync
        _guard_no_corpus(name, args, kwargs, ctx)          # Algorithm-2 invariant (§7.4)
        ctx.emit(PrimitiveEvent(name, "running", _summarize_args(args, kwargs)))
        t0 = time.monotonic()
        try:
            result = core(*args, **kwargs, ctx=ctx)
        except Exception as exc:
            ctx.emit(PrimitiveEvent(name, "failed", _summarize_args(args, kwargs),
                                    duration_ms=_ms(t0), error=repr(exc)))
            raise
        ctx.emit(PrimitiveEvent(name, "completed", _summarize_args(args, kwargs),
                                duration_ms=_ms(t0), result_summary=_summarize_result(result)))
        return result
    return primitive

def build_primitive_namespace(ctx: PrimitiveContext) -> dict[str, dict]:
    """Produces the rlms `custom_tools` dict. Called once per run by run.py."""
    ns = {name: {"tool": _adapt(name, core, ctx), "description": PRIMITIVE_DESCRIPTIONS[name]}
          for name, core in PRIMITIVE_REGISTRY.items()}
    _assert_no_reserved_collision(ns)        # rlms raises ValueError anyway; fail early + clearly
    return ns
```

- The factory's output **is** the `custom_tools` dict: `RLM(custom_tools=build_primitive_namespace(ctx), …)`.
  The `{"tool": …, "description": …}` shape lets `rlms` auto-generate the system-prompt
  tool section, so `system_prompt.py` (brief Phase 3) need not hand-duplicate signatures.
  `PRIMITIVE_DESCRIPTIONS` (in `primitives.py`, beside the registry) is a #60 deliverable —
  one precise line per primitive: signature, purpose, and the slices-not-corpus reminder.
- Each entry is a **sync** callable (`rlms` requires sync tools) closed over `ctx`. The
  root sees the clean signature (`understand_section(text_slice)`); `ctx` is injected by
  `_adapt`, invisible.
- The factory is re-entrant — each run/`rlm_query` child builds its own dict from its own
  context; no module-global mutable state.
- `_adapt` handles **only** guard + emit + timing + the error contract. LLM cost rides
  `ctx.invoke_agent`.

### 4.3 `primitive_bridge.py` — the async→sync bridge

`rlms` runs `custom_tools` **synchronously on the thread that calls `.completion()`**. So
`run.py` (brief Phase 3) must call `.completion()` on a **worker thread**
(`await asyncio.to_thread(rlm.completion, context)`), keeping the orchestrator event loop
free. Primitives execute on that worker thread; their async work bridges back to the loop:

```python
class PrimitiveTimeout(PrimitiveError): ...

def _run_sync(coro: Awaitable[T], ctx: PrimitiveContext, *,
              timeout: float, cleanup_grace: float = 15.0) -> T:
    if _is_loop_thread(ctx.event_loop):
        raise PrimitiveError("_run_sync invoked on the event-loop thread")   # deadlock guard
    fut = asyncio.run_coroutine_threadsafe(coro, ctx.event_loop)
    try:
        return fut.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        fut.cancel()                                          # actually cancel the coroutine
        try:
            fut.result(timeout=cleanup_grace)                 # let `finally:` cleanup settle
        except Exception:
            pass
        raise PrimitiveTimeout(f"timed out after {timeout}s")

def _invoke_llm(ctx: PrimitiveContext, agent: str, prompt: str, *,
                timeout: float, model_override: str | None = None) -> str:
    return _run_sync(ctx.invoke_agent(agent, prompt, model_override=model_override),
                     ctx, timeout=timeout)
```

`_run_sync` is the only async bridge; `_invoke_llm` is the only LLM path for primitives.
The cancellation + same-loop guard are carried from the Codex-reviewed v2.

### 4.4 Error types

```python
class PrimitiveError(RuntimeError): ...
class Algorithm2ViolationError(PrimitiveError): ...
class UnknownEnvironment(PrimitiveError): ...
class PrimitiveTimeout(PrimitiveError): ...
```

### 4.5 LLM path

Primitive LLM calls go through `_invoke_llm` → `ctx.invoke_agent` →
`orchestrator._invoke_agent` → `run_agent_with_resilience(..., ledger=, budget=, health=)`
(`orchestrator.py:647-655`) — reusing failover, the `RunBudget`, health tracking,
telemetry, cost-ledger writes. **`rlms`'s `llm_query`/`rlm_query` are not used by
primitives** — those are the *root's* decomposition sub-calls; a primitive doing its
domain job is not a sub-call (brief §8: the reproduction-run budget is separate from the
LM-call budget). Note also: `rlms` `llm_query` calls have *no* tool access by design — a
further reason primitive LLM work uses our own resilient path, not the library's.

**Cost accounting splits two ways:** the root's own `llm_query`/`rlm_query` spend is
tracked by `rlms` in `RLMChatCompletion.usage_summary`; primitive-internal spend lands in
our `cost_ledger`. `run.py` must sum both for the run's total LLM cost in `final_report`.

### 4.6 Concurrency model & termination

- `rlms` `exec`s root code blocks **sequentially**; one primitive runs at a time on the
  worker thread. All orchestrator-owned mutable state (cost ledger, `RunBudget`, health)
  is mutated only on the event loop, because every agent call is marshalled there via
  `_run_sync` — no data race. `EnvironmentRegistry` and `sandbox_runtime` are touched only
  by the single worker thread. This holds **under recursion** too: `rlms`'s `rlm_query`
  spawns a child `RLM` whose `.completion()` runs synchronously on the *same* worker
  thread — nesting adds no second thread. `run.py`'s `emit` must be thread-safe (§5).
- **Termination is not our concern and not via a `set_final` tool.** `rlms` terminates
  when the root emits `FINAL(text)` / `FINAL_VAR(varname)` (reserved built-ins). The root
  builds the final report as a REPL variable and calls `FINAL_VAR` on it;
  `RLMChatCompletion.response` carries the result. `set_final` from the old mapping doc is
  obsolete — dropped from #60's scope.

---

## 5. The seam — `primitives.py` ↔ `run.py`

#60 **defines and owns** `PrimitiveContext`, `PrimitiveEvent`, `EnvironmentRegistry`,
`AgentInvoker`, `CorpusFingerprints`, `build_primitive_namespace`, and the nine cores.

`run.py` (brief Phase 3) **consumes** them: it constructs a `PrimitiveContext` (binding
`invoke_agent` to `orchestrator._invoke_agent`, supplying `sandbox_runtime`, the live
`event_loop`, the corpus fingerprints, a `PrimitiveSettings` snapshot), implements `emit`
(a `PrimitiveEvent` → `primitive_call` SSE event + `dashboard_events.jsonl` line —
**thread-safe**: called from the REPL worker thread for `primitive_call`, and from
`rlms`'s callback thread if `run.py` also wires `on_subcall_*`), calls
`build_primitive_namespace(ctx)`, and passes the result to `RLM(custom_tools=…)`. `run.py` also calls `.completion()` on a worker thread and keeps
the loop alive (§4.3).

All seam types live in `primitive_context.py`; `run.py` imports them; CI type-checking on
both sides catches drift. This is a strictly smaller seam than v2's hand-built-engine
contract — `rlms` absorbs the REPL/loop/sub-call surface entirely.

---

## 6. Per-primitive specifications

Each primitive is a `core(<domain args>, *, ctx) -> <result>` wrapped by `_adapt`. Cores
6.1–6.3 are **pure**; 6.4–6.9 are **effectful** (LLM and/or filesystem/sandbox). "Helper"
names the shared function extracted from the legacy module. Sourced from the 2026-05-20/21
contract-inventory recon. (Unchanged from the Codex-reviewed v2 — these are engine-agnostic.)

### 6.1 `understand_section(text_slice: str) -> dict` — pure
Source: `paper_understanding.py` `_extract_*` (`:131–346`), bundled today by `run_offline`
(`:31`). Core runs the heuristic extractors over the slice. Helper: each
`_extract_X(sections)` refactored to a pure `extract_X(text: str)`. Failure: empty fields
on weak input, never an exception.

### 6.2 `extract_hyperparameters(text_slice: str) -> dict` — pure
Source: `paper_understanding._extract_training_recipe`. Helper: `extract_recipe(text)`.

### 6.3 `detect_environment(method_spec: dict) -> dict` — pure
Source: `environment_detective.py:54` `run_offline` + `_infer_*` (`:158–319`). Helper:
`infer_env(method_spec)`. Input schema frozen (§7.5) — the inference needs
`training_recipe`, `datasets`, `hardware_clues`, `model_architecture`; a malformed dict
raises `PrimitiveError`. Drop the unused `artifact_index` param and the FS writes.

### 6.4 `verify_against_rubric(results: dict, rubric: dict) -> dict` — effectful (LLM)
Source: `orchestrator._run_rubric_verifier` (`:1672`, async) — the LLM `"rubric-verifier"`
agent (`registry.py:183`). Core invokes it via `_invoke_llm`, post-processes through
`RubricVerification.from_areas(...)`, returns a dict. Helper: extract the verifier core
into `rubric_verifier.py` (orchestrator-coupled — §10 Step 1a). Invariant:
`overall_score`/`meets_target` come from `from_areas` (deterministic), not the LLM's raw
values. Failure: fail-closed — `None` → sentinel `{"error":…, "overall_score":0.0,
"meets_target":false}`.

### 6.5 `plan_reproduction(method_spec: dict, env_spec: dict) -> dict` — effectful (LLM)
Source: `orchestrator.run_reproduction_planner` (`:1504`, async) + `_normalize_reproduction_contract`
(`:911`). Core builds the planner prompt from the two dicts, invokes the
`reproduction-planner` agent via `_invoke_llm`, normalizes, returns a `ReproductionContract`
dict. Helper: new `reproduction_planner.py` (`build_planner_prompt`, `normalize_contract`).
Drop `project_id`/`assumption_ledger`/the file-write instruction from the prompt.

### 6.6 `implement_baseline(plan: dict) -> str` — effectful (LLM + FS)
Source: `baseline_implementation.py:418` `run_with_sdk`. Core generates baseline code from
`plan`, writes the code dir + the existing `baseline_result.json` artifact, returns
`code_path`. Helper: `generate_baseline(plan, *, ctx, out_dir)`. **Algorithm-2 fix:** takes
`plan: dict` (slices the root assembles), not `paper_claim_map`; `_guard_no_corpus`
enforces it. No `commands.json` invented — `run_experiment` reconstructs `BaselineResult`
from `baseline_result.json` (§7.2).

### 6.7 `build_environment(env_spec: dict) -> dict` — effectful (LLM + Docker)
Source: assembled from `orchestrator.py:1354–1500` (the build-and-repair loop). Core runs
the loop and — **behavior addition** — *tags and retains* the built image (the current
loop discards it, `:1410`), registers a `SandboxConfig` in `ctx.environments`, returns
`{success, env_id, image_id, error, attempts}`. Helper: new `environment_build.py`
(`build_and_repair`). Docker build via the existing `build_image` service through
`_run_sync`; repair sub-call via `_invoke_llm("environment-detective", …)`. Checkpoint/
audit side effects stay with the orchestrator wrapper (legacy path). Failure: build fails
after `settings.environment_build_max_attempts` → `{success:false, env_id:null, …}`.

### 6.8 `run_experiment(code_path: str, env_id: str) -> dict` — effectful (Docker)
Source: `experiment_runner.py` `run_with_runtime` (`:164`) + the sandbox dispatch
(`orchestrator.py:1627`). Core resolves `env_id` via `ctx.environments`, reconstructs
`BaselineResult` from `<code_path>/baseline_result.json`, runs the experiment on
`ctx.sandbox_runtime` through `_run_sync` (`timeout=settings.experiment_timeout_s`),
returns `{success, metrics, logs_path, error}`. Helper: extract the dispatch to
`run_experiment_in_sandbox`. Failure: crash/non-zero exit → `{success:false,…}`; unknown
`env_id` → `UnknownEnvironment` (raises).

### 6.9 `propose_improvements(current_results: dict, rubric_scores: dict, k: int | None = None) -> list[dict]` — effectful (LLM)
Source: the **existing `improvement-orchestrator` LLM agent** (`registry.py:201`, invoked
`orchestrator.py:1991`) — *not* a from-scratch rewrite (brief: the live selector is
already an LLM agent; hardcoding lives only in offline/`topology.py`/UI). Core builds the
proposer input from `current_results` + weak nodes in `rubric_scores`, invokes the agent
via `_invoke_llm`, returns a variable-length `[{id,title,tag,description,reasoning,
expected_delta}]` with free-form `tag`. The offline `improvement.py` is untouched (Phase 6
deletes it). Guard test FM-variation: candidate lists differ across three papers (§9).

---

## 7. Cross-primitive contracts

### 7.1 `env_id` — `build_environment` → `run_experiment`
`build_environment` builds + retains a Docker image, constructs a `SandboxConfig`, calls
`ctx.environments.register(config)` → an opaque `env_id`. `run_experiment` calls
`ctx.environments.resolve(env_id)`; unknown → `UnknownEnvironment`. Run-scoped registry.

### 7.2 `BaselineResult` handoff — `implement_baseline` → `run_experiment`
`implement_baseline` writes the existing `baseline_result.json` (schema `schemas.py:146`).
`run_experiment` reads it back and reconstructs `BaselineResult` for the executor —
lossless (`commands_to_run`, `dockerfile_path`, `assumptions_applied`, provenance). No new
artifact invented.

### 7.3 Failure is data, not an exception
`run_experiment`, `build_environment`, `verify_against_rubric` return a structured
result with a `success`/sentinel field on domain failure. Contract violations (unknown
`env_id`, malformed input dict, an Algorithm-2 arg) raise — `_adapt` emits `failed` and
re-raises so the traceback reaches the root via REPL stdout.

### 7.4 The Algorithm-2 arg-guard (brief §8.2)
`_guard_no_corpus` runs in `_adapt`, **defense-in-depth**: (1) a total-argument-byte cap
(recursive over lists/dicts/tuples) — the offloaded `context` is megabytes, a legitimate
slice is small; (2) SHA-1 of large string args vs `ctx.corpus_fingerprints`. The
**primary** guarantee is the static design — no primitive signature names a corpus
parameter (review rule + the FM#1 test, §9).

### 7.5 Primitive input dict schemas
Frozen, documented in `primitive_context.py`, validated at the core boundary: `method_spec`
(subset of `PaperClaimMap`), `env_spec` (`EnvironmentSpec` shape), `plan`, `results` /
`current_results`, `rubric` / `rubric_scores`. Malformed → `PrimitiveError`.

---

## 8. Error-handling matrix

| Primitive | Domain failure → data | Contract violation → raise |
|---|---|---|
| `understand_section` / `extract_hyperparameters` | empty fields on weak input | malformed (non-str) input |
| `detect_environment` | best-effort partial spec | `method_spec` fails §7.5 schema |
| `verify_against_rubric` | `None` → sentinel dict | malformed `results`/`rubric` |
| `plan_reproduction` | — | unparseable LLM output after retries |
| `implement_baseline` | — | codegen failure after retries; corpus arg |
| `build_environment` | `{success:false, env_id:null, …}` | malformed `env_spec` |
| `run_experiment` | `{success:false, metrics:{}, error}` | unknown `env_id`; missing artifact |
| `propose_improvements` | `[]` on no viable candidates | unparseable LLM output after retries |

No primitive uses bare `except` or silently swallows.

---

## 9. Test plan — the acceptance gate

- **Per primitive (`tests/rlm/primitives/`):** unit tests on each core with fakes (fake
  `AgentInvoker`, fake `sandbox_runtime`, fake `emit` → list); contract tests (signature
  matches the brief §7 list, return shape/keys, §7.5 schema validation, §8 failure behavior).
- **Framework (`tests/rlm/`):** `_adapt` emits running→completed / failed+re-raise;
  `_guard_no_corpus` byte-cap + recursive traversal + fingerprint match; `_run_sync`
  cancels on timeout (asserted via a fake coroutine recording cancellation+cleanup order)
  and the same-loop guard raises; `build_primitive_namespace` yields the rlms
  `custom_tools` shape and rejects reserved-name collisions.
- **Brief fidelity invariants (§8 of the brief):** invariant #1/#2 → the Algorithm-2 guard
  tests; invariant #9 → `propose_improvements` candidate lists differ across three papers.
- **`rlms` integration harness (`tests/rlm/test_primitive_integration.py`):** build a real
  `RLM(custom_tools=build_primitive_namespace(fake_ctx), environment="local",
  backend=<stub>)`, run `.completion()` on a tiny mock `context`, and assert the primitives
  are callable from REPL code, compose (the `build_environment`→`run_experiment` `env_id`
  handoff; the `implement_baseline`→`run_experiment` `baseline_result.json` handoff), and
  emit the expected `PrimitiveEvent` sequence. This exercises the *real* library — a
  genuine integration test, not a fake. **Step-0 prerequisite:** confirm how to give
  `rlms` a deterministic stub model backend (a mock `ClientBackend`, or `backend="litellm"`
  pointed at a local fake) — without it the harness cannot run hermetically.
- **Boundary (honest):** the harness proves the primitives compose under the real `rlms`
  loop with a stubbed model backend. The live PaperBench run with a real model is brief
  Phase 5, sequenced after `run.py` (brief Phase 3) — not gating #60.
- **Non-destructive proof:** every edited legacy module's existing test suite stays green.

---

## 10. Implementation sequence & delegation

**Execution model — Opus + Sonnet, no Codex.** Opus owns the design (this spec) and reviews
every diff; Sonnet sub-agents do the execution, with this spec as the gate. Each delegated
workstream gets exact paths, the interface contract, what not to touch, and acceptance
commands; Opus reviews the resulting diff before the next step starts.

- **Step 0 — Dependency + framework.** `pip install rlms`, pin it in the dependency
  manifest; write `primitive_context.py`, `primitive_adapter.py`, `primitive_bridge.py` +
  their tests. Blocks everything. One workstream — a Sonnet sub-agent against §4's detailed
  contract, **reviewed especially closely by Opus** as the keystone (it carries the `rlms`
  integration and the concurrency model).
- **Step 1a — Orchestrator-coupled extractions.** Extract `build_and_repair`
  (`environment_build.py`), the planner helpers (`reproduction_planner.py`), the
  rubric-verifier core (`rubric_verifier.py`), the improvement-proposer shaping, and the
  sandbox dispatch; rewire `orchestrator.py`. **All `orchestrator.py` edits here**,
  behavior-preserving, gated by orchestrator tests. One workstream.
- **Step 1b — Easy-tier helpers + cores (parallel with 1a).** Helper extraction in
  `paper_understanding.py` / `environment_detective.py` + `primitive_impl/understanding.py`,
  `detection.py`. No shared file — a Sonnet sub-agent, diff-reviewed by Opus.
- **Step 2 — Remaining cores (parallel; disjoint `primitive_impl/` files).**
  `verification.py`, `planning.py`, `environment.py`, `execution.py`, `improvement.py` +
  the `primitives.py` façade wiring. Each `primitive_impl/` file is one Sonnet sub-agent
  against its precise §6 spec; the hard cores (`build_environment`, `run_experiment`) get
  the closest diff-by-diff Opus review.
- **Step 3 — Integration.** The `rlms` integration harness + cross-primitive tests.

Commits serialized (or worktrees) so no two concurrent workstreams write the same file.

---

## 11. Risks & open items

1. **`rlms` v0.1.1 discrepancies vs the brief** (recon 2026-05-21). These affect **brief
   Phase 3 (`run.py`), not #60's primitives**, but are flagged here so the team sees them:
   (a) the brief's `answer` termination variable does not exist — termination is
   `FINAL`/`FINAL_VAR`; (b) `on_iteration_start/complete` callbacks are declared but
   **never fire** in v0.1.1 — `repl_iteration` SSE events need another mechanism;
   `on_subcall_*` do fire. #60 is unaffected — its `primitive_call` events come from
   `_adapt`, our code.
2. **`rlms` is v0.1.1** — an early library. Pin the version; isolate the dependency behind
   `run.py` + `build_primitive_namespace` so a future API change has a small blast radius.
3. **The two environments (§4).** The `rlms` REPL must be `environment="local"` (callable
   `custom_tools` cannot serialize into an isolated REPL); Docker is driven *inside* the
   primitives via `RuntimeAppService`. Verify at Step 0 and reconcile with issue #64's
   "`environment='docker'` decisive" phrasing — see the §4 "Two environments" note. If
   `rlms` ever forced a containerized REPL, the callable-`custom_tools` transport would
   need a redesign.
4. **`orchestrator.py` extraction (Step 1a)** edits the working demo path. Mitigation:
   behavior-preserving, gated by existing tests; characterization tests where thin.
5. **`build_environment` behavior addition (§6.7)** — image retention is new behavior.
   Scoped into #60, covered by unit tests with a fake Docker backend.
6. **PR #65's superseded files** (`repl_host.py`/`root_loop.py`/`sub_call.py`) — #60 does
   not depend on them; their removal + `__init__.py` reconciliation is the rlms-adoption
   work's job. Flag on #65.

---

## 12. Acceptance criteria for #60

- [ ] `rlms` pinned in the dependency manifest; `from rlm import RLM` works in the venv.
- [ ] All nine cores implemented; `build_primitive_namespace(ctx)` returns a valid `rlms`
      `custom_tools` dict (nine entries, `{"tool","description"}` shape, no reserved-name
      collision).
- [ ] Framework (`_adapt`, `_run_sync` with cancellation, `_invoke_llm`, `_guard_no_corpus`,
      `EnvironmentRegistry`) implemented and unit-tested.
- [ ] Every primitive has unit + contract tests, green, with mocked invoker/sandbox.
- [ ] `_run_sync` cancellation + same-loop-guard tests pass.
- [ ] Brief fidelity tests pass: Algorithm-2 corpus-arg rejection; `propose_improvements`
      varies across three papers.
- [ ] The `rlms` integration harness runs a real `RLM(custom_tools=…).completion()` on a
      mock context and asserts composition (the `env_id` and `baseline_result.json`
      handoffs).
- [ ] Every edited legacy module's existing test suite stays green.
- [ ] `PrimitiveContext` / `PrimitiveEvent` / `AgentInvoker` documented as the `run.py`
      seam; no static path by which the corpus reaches a primitive argument.
- [ ] `primitives.py` is the only edited Phase-1 file; the superseded skeleton files are
      untouched and undepended-on.

---

*v3 rebuilt on the `rlms` library (v0.1.1, recon 2026-05-21) after the brief rewrite.
Implements issue #60 (primitive integration) against `docs/rlm-pivot-mapping.md` and the
rewritten `docs/design/rlm-pivot-brief.md`. Per-primitive specs (§6) and contracts (§7–§8)
carry forward from the Codex-reviewed v2.*
