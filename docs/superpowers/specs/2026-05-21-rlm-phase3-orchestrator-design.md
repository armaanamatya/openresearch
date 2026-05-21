# RLM Phase 3 — Orchestrator: Design Spec

| | |
|---|---|
| **Issue** | [#60](https://github.com/armaanamatya/openresearch/issues/60) — RLM orchestrator: system prompt, `run.py`, SSE bridge |
| **Date** | 2026-05-21 · **v2** (Codex adversarial review folded in) |
| **Scope** | **Orchestrator only** — decoupled from #59 (the primitive layer). See §2. |
| **Branch** | impl on `feat/rlm-phase3-orchestrator` off `rlm-pivot` @ `63adb10` |
| **Canonical specs** | `docs/design/rlm-pivot-brief.md` (§5, §9, §10, §11 Phase 3) · `docs/rlm-pivot-mapping.md` |
| **#59 seam source** | Aayush's `docs/design/phase2-implementation-plan.md` + `phase2-execution-brief.md` (branch `feat/rlm-phase2-foundation`) |
| **Engine** | `rlms` 0.1.1 (PyPI; import name `rlm`) — every API claim grounded in a 2026-05-21 source recon |

> **Revision history.**
> **v1** — first orchestrator design, decoupled from #59 (Option 1, user-confirmed 2026-05-21).
> **v2** — Codex adversarial review (2026-05-21, 8 findings: 2 Critical, 3 High, 3 Medium), all
> verified against source and resolved. The resolutions are §18; the load-bearing ones —
> the corpus sanitizer (§9), the JSON `FINAL_VAR` contract (§7, §11), the lazy seam
> resolver (§2, §8), the precise event-store contract (§10), real emit locking (§9), and
> the full run-mode wiring surface (§12) — are folded into the body.

> **Context.** ReproLab is being re-architected from a 14-stage `PipelineStage` state
> machine to an RLM orchestrator built on the `rlms` library. Issue #60 is the
> **orchestrator** — the glue that builds an `rlm.RLM`, runs it on a paper, streams it
> live, checkpoints it, and writes `final_report.{json,md}`. Issue #59 (Aayush, in flight)
> is the **primitive layer**. The two build in parallel; #60 consumes #59's seam and never
> implements it. Decisions locked by the user 2026-05-21: orchestrator-only scope, a new
> `rlm` run mode, event-log + REPL-state-snapshot checkpointing, a configurable
> multi-provider root model (GPT-5 / Qwen3-Coder / Kimi K2.5 / Claude).

---

## 1. Summary

#60 replaces the `PipelineStage` machine — for a new `rlm` run mode — with a single
`rlm.completion()` call whose root model drives paper reproduction by writing REPL code
that calls #59's primitives. #60 is six new/filled modules in `backend/agents/rlm/` plus
run-mode wiring. It owns: the reproduction-domain **system prompt**, the **`run.py`** entry
that constructs and runs the `rlm.RLM`, the **SSE bridge** that streams the run live, the
**event-store checkpoint** layer, the **`final_report` writer**, and a **root-model
registry**. It does not implement the 9 primitives (#59), the frontend (#61), or a
real-paper run (#62).

`rlms` provides the Algorithm-1 loop, the REPL host, `llm_query`/`rlm_query`, and the
callbacks. #60 provides the reproduction domain layer and the operational layer (`rlms`
checkpoints nothing, persists nothing, writes no report, and — critically — sanitizes
nothing: see §9).

## 2. Scope & the #59 seam

### In scope
`backend/agents/rlm/`: `run.py`, `system_prompt.py` (fill the stub), `models.py`,
`sse_bridge.py`, `checkpoint.py`, `report.py`, `stub_primitives.py`, `__init__.py`
(re-export edit), and a vendored copy of `context.py` (§2 last paragraph). Run-mode wiring
in `backend/cli.py`, `backend/agents/pipeline.py`, `backend/services/events/live_runs.py`.
The test suite (§14).

### Out of scope
- **The primitive layer (#59):** `binding.py`, the filled `primitives.py`, the 9 primitive
  cores, the *authoritative* `RunContext`, `PRIMITIVE_DESCRIPTIONS`.
- Frontend rendering of the new events (#61).
- The real `ftrl` PaperBench bundle + a real-paper reproduction with a real rubric score
  (#62, brief Phase 5).
- The legacy `PipelineStage` / gate / five-path code — left runnable, deleted in Phase 6.

### The seam — and why it is lazily resolved (Codex H3)
#60 needs two #59 symbols: `RunContext` and `build_custom_tools`. As of 2026-05-21 #59 has
shipped `RunContext` (`context.py`, on `feat/rlm-phase2-foundation` only) and the test
scaffold; `binding.build_custom_tools` and the 9 primitives are fully specified in
`phase2-implementation-plan.md` but **not built and not on any branch #60 builds from**. A
top-level `from backend.agents.rlm.binding import build_custom_tools` would therefore
`ImportError`. Resolution:

1. **`context.py` is vendored** onto `feat/rlm-phase3-orchestrator` — a byte-identical copy
   of #59's shipped file (it is a tiny dataclass). When #59 merges, the identical content
   is a no-op merge. #60 does **not** edit `requirements.txt` (the `rlms` pin is #59's, on
   its branch; #60's `.venv` already has `rlms==0.1.1`).
2. **`build_custom_tools` is lazily resolved** by `run.py` — never imported at module load.
   A `_resolve_custom_tools(ctx)` function (§8) tries `backend.agents.rlm.binding`; on
   `ImportError`, or when `REPROLAB_RLM_STUB_PRIMITIVES=1`, it falls back to
   `build_stub_custom_tools` (§13). `run.py` therefore imports cleanly today and is fully
   testable; switching to #59's real primitives is automatic once `binding.py` lands.

### Non-collision guarantee
The only file both issues touch is `backend/agents/rlm/__init__.py`, append-only on both
sides (imports / `__all__`). `context.py` is vendored identical → clean merge. Every other
#60 file is untouched by #59. Verified by an alignment digest of #59's full plan.

## 3. Module layout — `backend/agents/rlm/`

```
# orchestrator — #60 owns
run.py            NEW   run_pipeline_rlm(): build the RLM, run .completion() on a worker
                        thread, drive checkpoint + report. The entry replacing the machine.
system_prompt.py  FILL  build_system_prompt(): reproduction-domain RLM prompt (stub exists).
models.py         NEW   root-model registry: key -> {rlm backend, kwargs, sub-model,
                        prompt addendum, paper-validated}.
sse_bridge.py     NEW   ReproLabRLMLogger(RLMLogger) + the corpus SANITIZER + on_subcall_*
                        adapters + the event schema.
checkpoint.py     NEW   per-iteration -> SQLite event store + sanitized REPL-state snapshot.
report.py         NEW   RLMChatCompletion -> RLMFinalReport -> final_report.{json,md}.
stub_primitives.py NEW  build_stub_custom_tools(): deterministic fake primitives so the
                        orchestrator is runnable/testable before #59 lands (shipped
                        internal code — NOT in tests/; see Codex M1).
__init__.py       EDIT  re-export the #60 public API; drop dead-skeleton re-exports.
context.py        VEND  byte-identical copy of #59's RunContext dataclass (see §2).

# primitive layer — #59 owns; #60 only consumes
binding.py        #59   build_custom_tools (specified, unbuilt — lazily resolved by run.py).
primitives.py     #59   the 9 cores + PRIMITIVE_REGISTRY + PRIMITIVE_DESCRIPTIONS.

# dead Phase-1 skeleton — superseded by the rlms library; untouched by #60, deleted Phase 6
repl_host.py · root_loop.py · sub_call.py
```

Wiring outside the package (§12): `backend/cli.py`, `backend/agents/pipeline.py`,
`backend/services/events/live_runs.py`.

## 4. The `rlm` 0.1.1 contract — verified facts `run.py` depends on

All confirmed by a 2026-05-21 source read of the installed package.

- **`RLM(...).completion(prompt, root_prompt=None) -> RLMChatCompletion`** — synchronous,
  blocks the calling thread. `prompt` (a `dict` is accepted) becomes the offloaded
  **`context`** REPL variable; `root_prompt` is a *small* instruction the root sees
  directly. → #60 passes the **paper corpus dict as `prompt`** and a **short task
  instruction as `root_prompt`**.
- **`RLMChatCompletion`** fields: `root_model`, `prompt`, `response`, `usage_summary`
  (root + sub-call LLM cost), `execution_time`, `metadata`.
- **Termination — `FINAL_VAR(name)` is `str()`-ified (Codex C1).** `LocalREPL._final_var`
  (`local_repl.py:208-218`) does `answer = str(self.locals[variable_name])`. A dict bound
  in the REPL therefore returns as **Python-repr text**, not structured data.
  `RLMChatCompletion.response` is always a string. → the report contract (§7, §11) is
  **JSON**: the root `json.dumps(...)` its report into a variable and `FINAL_VAR`s *that*.
- **`RLMIteration.to_dict()` carries the entire corpus (Codex C2).** `types.py:183-218`:
  `RLMIteration.to_dict()` → `code_blocks` → `CodeBlock.to_dict()` → `REPLResult.to_dict()`
  → `"locals": {k: _serialize_value(v) ...}`. `local_repl.py:382` binds `context = context_0`
  — the whole paper — into `locals`. Anything that logs/streams/snapshots `to_dict()`
  unsanitized **leaks the paper corpus** (Algorithm-2 violation, brief §8). → §9's
  sanitizer is mandatory and is the single chokepoint.
- **Per-iteration hook** — `rlm` calls `logger.log(iteration: RLMIteration)` once per loop
  (`rlm/core/rlm.py:367`). `on_iteration_start/complete` are declared but **never fire**.
- **`on_subcall_start(depth, model, prompt_preview)` / `on_subcall_complete(depth, model,
  duration, error)`** — these **do** fire (`rlm/core/rlm.py:739, 805`).
- **`max_timeout` is checked between iterations only (Codex H2).** `rlm.py:308`
  `_check_timeout(i, time_start)` runs at the top of the loop; the iteration body then
  blocks inside `environment.execute_code(...)`. A primitive that hangs (a stuck Docker
  build) overruns `max_timeout` indefinitely. → §8's three-layer time bound.
- **`environment="local"` is mandatory** — `DockerREPL` does not inject `custom_tools`.
- **`custom_tools`** — `{name: {"tool": callable, "description": str}}`. `rlm` builds the
  final system prompt from `custom_system_prompt` **plus** an auto-generated tool section
  from the `description` fields (`rlm.py:259`). → `system_prompt.py` need not hand-list
  signatures.
- **`custom_sub_tools={}`** — recursive `rlm_query` children get no domain primitives.
- **`compaction=True`** — compacts the root's own history into the `history` REPL variable
  near the context limit; safe to enable.
- Reserved REPL names: `llm_query(_batched)`, `rlm_query(_batched)`, `FINAL_VAR`,
  `SHOW_VARS`, `context`, `history`.

## 5. The #59 seam — what `run.py` consumes

### `RunContext` (shipped on #59's branch; vendored — §2)
```python
@dataclass
class RunContext:
    project_id: str
    project_dir: Path
    runs_root: Path
    dashboard: Any           # DashboardEmitter
    cost_ledger: Any         # RunCostLedger
    llm_client: Any          # LlmClient protocol: .complete(*, system, user) -> str
    provider: str            # "anthropic" | "openai"
    model: str
    runtime: Any = None      # AgentRuntime — only implement_baseline uses it
    workspace_service: Any = None
    workspace_id: str | None = None
```
`run.py` constructs and populates this. No `event_loop` field — #59's primitives
self-isolate async work (below), so `run.py` exposes no shared loop.

### `build_custom_tools` (specified, `binding.py`) — lazily resolved (§2, §8)
`build_custom_tools(ctx: RunContext, *, registry=None, descriptions=None) -> dict[str,
dict]` — returns the `rlm` `custom_tools` dict, each primitive closed over `ctx`, wrapped
with `primitive_call` event emission + a cost-ledger row.

### The 9 primitives (the `custom_tools` the root sees)
| Primitive | Root-visible signature | Returns |
|---|---|---|
| `understand_section` | `(text_slice: str)` | `dict` (partial `PaperClaimMap`) |
| `extract_hyperparameters` | `(text_slice: str)` | `dict` (`TrainingRecipe`) |
| `detect_environment` | `(method_spec: dict)` | `dict` (`EnvironmentSpec`) |
| `build_environment` | `(env_spec: dict)` | `dict` `{ok, image_tag, error, attempts}` |
| `plan_reproduction` | `(method_spec: dict, env_spec: dict)` | `dict` (`ReproductionContract`) |
| `implement_baseline` | `(plan: dict)` | `str` (code-dir path) |
| `run_experiment` | `(code_path: str, env_id: str)` | `dict` `{success, metrics, logs}` |
| `verify_against_rubric` | `(results: dict, rubric: dict)` | `dict` (`RubricVerification`) |
| `propose_improvements` | `(current_results, rubric_scores, k=None)` | `list[dict]` |

### Async model — no shared loop
#59's `build_environment` / `run_experiment` / `implement_baseline` bridge async work with
a per-call throwaway `ThreadPoolExecutor` + `asyncio.run` — fully self-contained. → `run.py`
runs `rlm.completion()` on one worker thread (`asyncio.to_thread`) and exposes **no** event
loop to primitives. The mapping-doc §6.5 shared-loop decision is superseded.

### #59→#60 handoffs (#60 must close these)
1. **Cost/usage.** #59's wrapper logs a *zero-usage* `CostLedgerEntry` per call. #60
   supplies a usage-capturing `llm_client` and captures root/sub cost from `usage_summary`
   (§11).
2. **Iteration index.** #59's `primitive_call` events carry `iteration=None`. #60's
   `ReproLabRLMLogger` knows the index; the UI correlates `primitive_call` to
   `repl_iteration` by stream order (§9). Non-blocking.
3. **Final report.** Phase 2 defines no report schema; `run_experiment` returns
   `metrics={}`. #60 defines the contract and tolerates empty metrics (§11).
4. **Per-primitive deadlines (Codex H2).** Phase 2's primitives have no internal time
   bound. A hung `build_environment`/`run_experiment` defeats `rlm`'s `max_timeout`. #60
   cannot reach inside #59's primitives — this is flagged to #59 as a **required**
   addition (each Docker/sandbox primitive must bound its own execution and cancel the
   sandbox on deadline). #60's mitigation is the §8 process-level wall-clock backstop.

## 6. `models.py` — root-model registry

The root model is configurable; dispatch is through a registry, never `isinstance`.

```python
@dataclass(frozen=True)
class RootModel:
    key: str                       # "gpt-5" | "qwen3-coder" | "kimi-k2.5" | "claude"
    rlm_backend: str               # an rlm ClientBackend literal
    backend_kwargs: dict           # {"model_name": ...}
    sub_backend: str               # cheaper sub-call model backend
    sub_backend_kwargs: dict
    prompt_addendum: str           # per-model system-prompt addendum ("" for most)
    paper_validated: bool

ROOT_MODELS: dict[str, RootModel] = { ... }   # the four entries below

def resolve_root_model(name: str | None) -> RootModel:
    """name -> RootModel. None: REPROLAB_RLM_ROOT_MODEL env, else the layered default
    (gpt-5 if an OpenAI key is present, else qwen3-coder). Unknown name -> ValueError."""
```

| key | `rlm_backend` | sub-call | validated | addendum |
|---|---|---|---|---|
| `gpt-5` | `openai` | `openai` (gpt-5-mini) | ✅ | — |
| `qwen3-coder` | `openrouter` | `openrouter` (a cheap Qwen) | ✅ | the paper's Qwen anti-over-`llm_query` line |
| `kimi-k2.5` | `openrouter` | `openrouter` (cheap) | ❌ | — |
| `claude` | `anthropic` | `anthropic` (haiku) | ❌ | — |

OpenRouter model slugs are config-overridable env vars. An unvalidated root emits a
`root_model_unvalidated` warning at run start. Wired into `RLM(backend=…, backend_kwargs=…,
other_backends=[sub_backend], other_backend_kwargs=[…])`.

## 7. `system_prompt.py` — the reproduction-domain RLM prompt

`build_system_prompt(*, context_metadata: dict, root_model: RootModel) -> str` returns the
`custom_system_prompt`. `rlm` appends the auto-generated primitive tool section, so this
prompt carries **principles, not signatures**:

1. **RLM operating model** — paper §2 properties 1/2/3: the paper is offloaded in
   `context` (a dict — slice it, never read it whole into a message); the output is built
   as a REPL variable; `llm_query`/`rlm_query` are Python functions for navigating
   `context`.
2. **`context` metadata** — each key (`paper_text`, `paper_metadata`, `supplementary_text`,
   `repo_files`, `prior_work_refs`, `rubric_spec`) by name, type, length — never contents.
3. **The primitives** — what they are for (domain operations on slices/specs the root
   assembles) and the Algorithm-2 rule: never pass a whole `context` value to a primitive.
4. **Termination & the JSON report contract (Codex C1).** Build the final report as a
   Python dict, then `import json; report_json = json.dumps(<dict>, default=str)`, then
   call `FINAL_VAR("report_json")`. The prompt states the required report dict shape
   (the §11 `RLMFinalReport` fields, in plain language). `FINAL_VAR` of a raw dict is
   forbidden — it would stringify to un-parseable Python repr.
5. **≥1 in-context decomposition example** — paper Fig 4a.
6. **Triage instruction** — decline improvement candidates unlikely to lift weak rubric
   nodes (brief §8 invariant #7).
7. **`root_model.prompt_addendum`** appended verbatim.
8. **No fixed workflow.**

## 8. `run.py` — the run entry

```python
async def run_pipeline_rlm(
    project_id: str, runs_root: Path, workspace_claim_map: dict, *,
    model: str | None = None,            # root-model key; None -> resolve default
    provider: str | None = None,         # primitive LLM provider for RunContext
    runtime: Any = None,                 # AgentRuntime
    run_budget: Any = None,              # RunBudget -> max_timeout + wall-clock backstop
    sandbox_mode: str = DEFAULT_SANDBOX_MODE,
    seed: int | None = None, execution_profile: Any = None,
    attempt_id: str | None = None, run_group_id: str | None = None,
    workspace_service: Any = None, workspace_id: str | None = None,
) -> RLMRunResult
```
`RLMRunResult`: `project_id, status, iterations, rubric_score, cost_usd,
final_report_path`. Signature parallels `run_pipeline_sdk` for uniform dispatch.

**Flow:**
1. Build `RunCostLedger` (`project_dir/cost_ledger.jsonl`), `DashboardEmitter`, resolve
   `RunBudget` → `max_timeout` + a wall-clock budget.
2. Build a **usage-capturing `llm_client`** — wrap `ClaudeLlmClient`/`OpenAILlmClient`
   (mapping §3) so each `.complete()` records a real `CostLedgerEntry`.
3. Build `sandbox_runtime` / `AgentRuntime`.
4. Construct `RunContext` (§5).
5. `custom_tools = _resolve_custom_tools(ctx)` — tries `binding.build_custom_tools`; on
   `ImportError` or `REPROLAB_RLM_STUB_PRIMITIVES=1`, uses `build_stub_custom_tools` (§13).
   Logs which provider is active.
6. Assemble the offloaded `context` dict from `workspace_claim_map` + `PaperExtractor`
   output: `{paper_text, paper_metadata, supplementary_text, repo_files, prior_work_refs,
   rubric_spec}`. (`workspace_claim_map`'s exact shape — Step-0 recon.)
7. `root_model = resolve_root_model(model)`; emit `root_model_unvalidated` if applicable.
8. `system_prompt = build_system_prompt(...)`.
9. `checkpointer = IterationCheckpointer(...)` (§10); `logger = ReproLabRLMLogger(emit=…,
   checkpointer=…)` (§9) — **no `log_dir`** (the base would write unsanitized JSONL).
10. Construct the RLM:
    ```python
    rlm = RLM(
        backend=root_model.rlm_backend, backend_kwargs=root_model.backend_kwargs,
        environment="local", max_depth=2, max_iterations=20,
        max_timeout=<from run_budget>, compaction=True,
        other_backends=[root_model.sub_backend],
        other_backend_kwargs=[root_model.sub_backend_kwargs],
        custom_tools=custom_tools, custom_sub_tools={},
        custom_system_prompt=system_prompt, logger=logger,
        on_subcall_start=<cb>, on_subcall_complete=<cb>,
    )
    ```
11. `result = await asyncio.to_thread(rlm.completion, context_dict, root_prompt)`.
12. `final_report = report.build_final_report(result, ctx=ctx, root_model=root_model)`;
    write `final_report.{json,md}` + `demo_status.json`; emit `run_complete`; return
    `RLMRunResult`.

**Three-layer time bound (Codex H2).** (a) `rlm`'s `max_timeout` — soft, between
iterations. (b) #59's per-primitive deadlines — the real bound on a hung Docker build
(§5 handoff 4, a #59 requirement). (c) **Process-level wall-clock backstop** — the run
executes as a subprocess spawned by `live_runs.py`; `run.py` arms a `threading.Timer`
watchdog at the `RunBudget` wall-clock that, on breach, writes an honest partial report +
`demo_status.json` and calls `os._exit(EXIT_WALLCLOCK)`. The OS reclaims the worker thread
— the only reliable way to stop a `.completion()` blocked in a primitive.

**Failure handling.** `BudgetExhausted`, `CancellationError`, `rlm` timeout, `max_errors`
stop → write an honest **partial** `final_report` (verdict `partial`/`failed`), set
`demo_status.json`, return a non-crashing `RLMRunResult`. No bare `except`.

## 9. `sse_bridge.py` — the sanitizer, the logger, the event schema

### 9.1 The corpus sanitizer (Codex C2) — the single chokepoint
```python
def sanitize_iteration(iteration: RLMIteration, index: int) -> dict:
    """Corpus-free projection of one RLMIteration. The ONLY form that may be
    streamed (SSE), persisted (event store), or snapshotted. Never returns a
    primitive's input/output value, never `locals` values, never `context`."""
```
Rules — applied to every field before it leaves the process:
- `iteration.response` → kept, **bounded** to ≤4000 chars (the root's own reasoning text).
- each `code_block.code` → kept (the Python the root wrote — the UI centerpiece).
- each block's `result.stdout`/`stderr` → reduced to **metadata only**: `{length, prefix
  (≤200 chars), has_traceback}` (Algorithm 1: only stdout *metadata* propagates).
- each block's `result.locals` → reduced to `{name: {"type": str, "size": int}}` for keys
  not starting with `_` and **excluding any `context*` key entirely**. Never a value.
- `result.rlm_calls` → count only.
- `iteration.prompt` and `iteration.final_answer` raw text → **dropped** (`prompt` is the
  message history; the final answer is surfaced via `report.py`, not the event stream).
- output: `{iteration, response, code_blocks: [{code, stdout_meta, stderr_meta, vars}],
  sub_calls, timing}`.

This sanitizer is the Algorithm-2 guard for the logging path (brief §8 invariant). It is
unit-tested with an `RLMIteration` whose `locals` contains a `context` value — the test
asserts that value never appears in the output.

### 9.2 `ReproLabRLMLogger`
```python
class ReproLabRLMLogger(RLMLogger):
    def __init__(self, *, emit, checkpointer): super().__init__(log_dir=None); ...
    def log(self, iteration: RLMIteration) -> None:
        clean = sanitize_iteration(iteration, self.next_index())
        self.emit(_repl_iteration_event(clean))
        self.checkpointer.record(clean)
```
**Does NOT call `super().log(iteration)`** — the base would capture the raw
`iteration.to_dict()` (corpus) in memory and, with `log_dir`, write it to JSONL. #60 owns a
sanitized trajectory instead; `log_dir=None`. The base's `log_metadata` (config only,
corpus-free) and `clear_iterations` remain inherited and harmless. `RLMChatCompletion.
metadata` will be empty — `report.py` does not depend on it.

### 9.3 Event emission — real locking (Codex M3)
`DashboardEmitter._emit` (`dashboard_emitter.py:66`) writes **without a lock**. The `emit`
callable `run.py` builds therefore owns a `threading.Lock` and serializes every
`DashboardEmitter` write itself — it is called from the worker thread
(`ReproLabRLMLogger`) and `rlm`'s callback thread (`on_subcall_*`).

### 9.4 The #60 event schema
Written to `dashboard_events.jsonl`, ride the existing `dashboard_event` SSE frame;
snake_case keys; every payload already sanitized (§9.1).

| `event` | Source | Key fields |
|---|---|---|
| `repl_iteration` | `ReproLabRLMLogger.log()` | `iteration`, `code`, `response`, per-block `stdout_meta`/`vars`, `sub_calls` |
| `sub_rlm_spawned` | `on_subcall_start` | `depth`, `model`, `prompt_preview` (≤200 chars) |
| `sub_rlm_complete` | `on_subcall_complete` | `depth`, `model`, `duration_ms`, `error` |
| `run_complete` | `run.py` end | `status`, `iterations`, `rubric_score`, `cost_usd`, `final_report_path` |

`primitive_call` events are #59's. The richer brief-§9 events (`candidate_proposed`,
`rubric_score`, …) are projections #61 derives from the `primitive_call` stream — #60 does
not over-emit them.

## 10. `checkpoint.py` — event store + sanitized snapshot

Per the locked decision (event-log **+ REPL-state snapshots**; resume deferred).

```python
class IterationCheckpointer:
    def __init__(self, *, project_id, event_store, snapshot_dir): self._version = 0; ...
    def record(self, clean: dict) -> None: ...
```
`record(...)` takes an **already-sanitized** iteration dict (§9.1) and:

1. **SQLite event store (Codex M2).** Appends one `RLMRunIteration` domain event:
   ```python
   event_store.append(
       aggregate_id=f"rlm-run:{project_id}",   # DISTINCT from the ingestion/workspace
       aggregate_type="rlm_run",               #   aggregate — its own version sequence
       events=[RLMRunIteration(**clean)],      # a @register_event() Pydantic model
       expected_version=self._version,         # 0 for the first; tracked, single-writer
       envelopes=[EventEnvelope(event_id=uuid4(), correlation_id=project_id, ...)],
   )
   self._version += 1
   ```
   `RLMRunIteration` is a new `DomainEvent` subclass registered via the existing
   `@register_event()` mechanism; its payload is the sanitized dict — bounded, corpus-free.
   The dedicated aggregate id means the RLM run never collides with ingestion's
   `expected_version` sequence. Single writer (the worker thread) → `expected_version`
   tracking is race-free; a `ConcurrencyError` is a hard bug, surfaced not swallowed.
2. **Local snapshot.** Appends the sanitized dict to `runs/<id>/rlm_state/iterations.jsonl`
   — the forensic trajectory + the REPL-state *shape* per iteration (variable names, types,
   sizes from §9.1's `vars`). Corpus-free by construction.

**Honest boundary.** "REPL-state snapshot" here is the per-iteration *trajectory + the
variable-shape manifest* — never variable *values* (`locals` holds the corpus). A true
value-snapshot, and resume-from-snapshot, need `rlm` support (`.completion()` cannot
re-enter at iteration N) — **explicitly deferred**; no speculative redaction-for-resume
machinery is built now.

## 11. `report.py` — `final_report.{json,md}`

```python
class RLMFinalReport(BaseModel):
    paper: dict                 # id, title
    verdict: Literal["reproduced", "partial", "failed"]
    reproduction_summary: str
    baseline_metrics: dict      # may be {} — Phase 2 run_experiment returns metrics={}
    paper_claims: dict
    rubric: dict                # overall_score, meets_target, areas
    improvements: list[dict]
    primitive_trace: dict
    cost: dict                  # llm_usd total + {root, sub, primitives} breakdown
    iterations: int
```

`build_final_report(result: RLMChatCompletion, *, ctx, root_model) -> RLMFinalReport`:
- **Parse `result.response` (Codex C1).** It is a string. Try `json.loads(raw)`; on failure
  try `ast.literal_eval(raw)` (recovers a dict that was `str()`-ified by `FINAL_VAR`); on
  both failing, produce a `failed` verdict carrying the raw text in `reproduction_summary`
  — never crash.
- Validate the parsed dict into `RLMFinalReport`, honest defaults for missing fields (an
  under-reporting root → `partial`/`failed`, not an exception).
- **Cost reconciliation:** `result.usage_summary` (root + sub LLM) + the `cost_ledger`
  entries from the usage-capturing `llm_client` (primitive-internal LLM). #59's zero-usage
  wrapper rows are a call log — counted for `primitive_trace`, ignored for `cost`.
- Tolerate empty `baseline_metrics`.

A fresh `.md` renderer (no legacy `FinalReport`/gate fields): rubric score prominent,
baseline vs. paper claims, improvement candidates + outcomes, an honest verdict.
`write_final_report_rlm(report, project_dir) -> (json_path, md_path)` — atomic (temp +
`os.replace`).

## 12. Run-mode wiring (Codex H1 — fuller than v1 stated)

`rlm` mode needs first-class handling at every point the current code branches on mode:

- **`backend/cli.py`** — (a) `--mode` choices `("offline","sdk","rlm")` (`cli.py:683`);
  (b) `cmd_reproduce` dispatch — add the `rlm` branch → `asyncio.run(run_pipeline_rlm(...))`;
  (c) `cmd_reproduce`'s final summary (`cli.py:611-626`) reads `state.stage.value`,
  `state.gate_*`, `state.path_results` — all `PipelineState` fields absent from
  `RLMRunResult`. Add an `rlm`-mode summary branch built from `RLMRunResult`
  (`status`, `iterations`, `rubric_score`, `cost_usd`).
- **`backend/agents/pipeline.py`** — add `run_pipeline_rlm(...)` (thin: delegates to
  `backend.agents.rlm.run.run_pipeline_rlm`; keeps `pipeline.py` the dispatch home).
- **`backend/services/events/live_runs.py`** — (a) `RunMode = Literal["offline","sdk","rlm"]`
  (`:24`); (b) the live-run request model / any `RunMode`-typed field; (c) `_python_script`
  imports `run_pipeline_rlm` alongside the offline/sdk runners (`:1209-1210`); (d) the
  `_python_script` dispatch (`:1331` — currently `sdk` vs. else→offline) gains an explicit
  `rlm` branch; (e) status/report enrichment that assumes a `PipelineState` shape tolerates
  the `rlm` run's `demo_status.json` / `final_report.json`.
- **`__init__.py`** — re-export `run_pipeline_rlm`, `build_system_prompt`,
  `resolve_root_model`; drop the dead `ReplHost`/`RootLoop`/`llm_query`/`rlm_query`
  re-exports. Append-only vs. #59's edits.

`offline` and `sdk` stay fully runnable (brief §11 — runnable until the Phase 6 cutover).

## 13. `stub_primitives.py` — orchestrator testability before #59

`build_stub_custom_tools(ctx: RunContext) -> dict[str, dict]` — **shipped internal code**
(`backend/agents/rlm/stub_primitives.py`, not `tests/` — Codex M1), returning the 9
primitives as deterministic fakes matching the §5 signatures (`understand_section` → a
fixed partial-claim dict; `build_environment` → `{ok: True, image_tag: "stub:latest", …}`;
etc.). `run.py`'s `_resolve_custom_tools` uses it when `binding.py` is absent or
`REPROLAB_RLM_STUB_PRIMITIVES=1`. It makes the whole orchestrator — loop, SSE, checkpoint,
report — runnable and testable before #59 lands. Module docstring marks it a
development/integration aid.

## 14. Test plan — the acceptance gate

- **Unit (`tests/rlm/`):** `models.py` (resolution, default layering, unknown →
  `ValueError`); `system_prompt.py` (RLM principles, the JSON report contract, the
  per-model addendum, no workflow); **`sanitize_iteration`** (an `RLMIteration` carrying a
  `context` value in `locals` → that value is absent from the output — the C2 regression
  test); `sse_bridge.py` (`ReproLabRLMLogger.log()` does not call `super().log()`, emits
  sanitized `repl_iteration`, locks emission; `on_subcall_*` → events); `checkpoint.py`
  (`RLMRunIteration` append with a tracked `expected_version`, distinct aggregate, snapshot
  JSONL); `report.py` (a JSON `response` → valid report; a `str()`-ified dict → recovered
  via `ast.literal_eval`; un-parseable → `failed` not a crash; cost reconciliation; empty
  metrics tolerated).
- **Integration — the real `rlm` harness (`tests/rlm/test_run_integration.py`):** a real
  `RLM(custom_tools=build_stub_custom_tools(ctx), environment="local", backend=<stub>)`,
  run via `run_pipeline_rlm` on a tiny mock corpus; assert the loop runs, stub primitives
  are callable, `repl_iteration`/`run_complete` stream, an `RLMRunIteration` lands in the
  event store, `final_report.{json,md}` is written and valid, **and the streamed events +
  the event store contain no `context` corpus substring** (the end-to-end C2 assertion).
  Step-0: determine a deterministic `rlm` stub model backend.
- **Boundary:** the harness proves the orchestrator end-to-end with stub primitives + a
  stub model. A live run with a real model + #59's primitives + the real `ftrl` bundle is
  #62 — not gating #60.
- **Non-regression:** `offline` and `sdk` mode tests stay green.

## 15. Delegation & sequencing

Opus owns this spec and reviews every diff; Sonnet sub-agents execute against the
per-module §-contracts; **no Codex on implementation**. Disjoint file sets; serialized
commits.

| Wave | Agent | Files | Depends on |
|---|---|---|---|
| **A** | A1 | `models.py` + `system_prompt.py` + tests | — |
| **A** | A2 | `sse_bridge.py` (incl. `sanitize_iteration`) + `checkpoint.py` + tests | — |
| **A** | A3 | `report.py` + `RLMFinalReport` + `stub_primitives.py` + tests | — |
| **B** | Opus | `run.py` — the integration keystone (incl. `_resolve_custom_tools`, the wall-clock watchdog) | A |
| **B** | B1 | §12 wiring: `cli.py`, `pipeline.py`, `live_runs.py`, `__init__.py`; vendor `context.py` | A |
| **C** | C1 | the real-`rlm` integration harness + Step-0 stub-backend recon | B |

Wave A is three parallel agents on disjoint files. Wave B runs `run.py` (Opus) and the
wiring (B1) in parallel — disjoint files. Each delegated task gets exact paths, the
§-contract, what NOT to touch (everything #59-owned), and acceptance commands
(`.venv/bin/python -m pytest tests/rlm/<file> -v`).

## 16. Risks & open items

1. **#59 timing.** `build_custom_tools` + the 9 primitives are specified, unbuilt. #60 is
   unblocked (lazy resolver + stub, §2/§13); real-primitive integration is automatic when
   #59 lands.
2. **Hung primitives (Codex H2).** `rlm`'s `max_timeout` does not interrupt a primitive
   blocked in `execute_code`. Layered mitigation (§8); the real fix — per-primitive
   deadlines — is a **required** #59 addition flagged in §5. Until then a wedged Docker
   build is bounded only by the process-level wall-clock backstop.
3. **`rlm` 0.1.1 is early.** Pin `rlms==0.1.1`; the library surface is isolated behind
   `run.py` + `sse_bridge.py`.
4. **`RLMIteration` field names (Step-0).** `sanitize_iteration` and the event builders
   depend on the exact `RLMIteration`/`REPLResult` fields (`rlm/core/types.py:160-218`,
   read for this spec) — re-verify on any `rlm` upgrade.
5. **Stub model backend (Step-0).** The integration harness needs a deterministic `rlm`
   model backend; if no clean stub seam exists, use a recorded/replayed backend.
6. **`__init__.py` shared file.** Append-only both sides; serialize the commit.
7. **Real-paper run is #62.** #60's done-condition is the machinery proven end-to-end on
   stub primitives — not a real PaperBench score.

## 17. Acceptance criteria for #60

- [ ] `--mode rlm` runs `run_pipeline_rlm` end-to-end; `offline` and `sdk` still work.
- [ ] `run.py` builds a real `RLM(environment="local", max_depth=2, …)`, runs
      `.completion()` on a worker thread, writes `final_report.{json,md}` + `demo_status.json`.
- [ ] No corpus leak: `sanitize_iteration` strips `locals`/`context`; the integration test
      asserts no corpus substring in any streamed event or stored event.
- [ ] `FINAL_VAR` JSON round-trip: the root emits `json.dumps`'d report; `report.py`
      recovers it (incl. the `ast.literal_eval` fallback).
- [ ] `system_prompt.py` carries RLM principles, `context` metadata, the JSON report
      contract, ≥1 decomposition example, the per-model addendum — and no workflow.
- [ ] `models.py` resolves GPT-5 / Qwen3-Coder / Kimi K2.5 / Claude; layered default;
      unvalidated-root warning.
- [ ] SSE: `repl_iteration`, `sub_rlm_spawned`/`complete`, `run_complete` stream;
      emission is lock-serialized; `on_subcall_*` wired.
- [ ] Checkpoint: one `RLMRunIteration` event per iteration on a dedicated aggregate with
      a tracked `expected_version`; sanitized snapshot JSONL.
- [ ] `report.py` → valid `RLMFinalReport`, cost reconciled, empty metrics tolerated.
- [ ] Every module unit-tested; the real-`rlm` integration harness passes with stub
      primitives + stub backend.
- [ ] Run-mode wiring complete in `cli.py`, `pipeline.py`, `live_runs.py`; zero edits to
      #59-owned files; `__init__.py` edit append-only; `context.py` vendored byte-identical.

## 18. Codex review resolutions (v2)

| # | Severity | Finding | Resolution |
|---|---|---|---|
| C1 | Critical | `FINAL_VAR` `str()`-ifies its value — a dict becomes repr text | JSON contract: root `json.dumps` → `report.py` `json.loads` w/ `ast.literal_eval` fallback (§7, §11) |
| C2 | Critical | `RLMIteration.to_dict()` carries `locals` incl. the `context` corpus | `sanitize_iteration` — the single chokepoint; `ReproLabRLMLogger` drops `super().log()`/`log_dir`; regression test (§9, §14) |
| H1 | High | Run-mode wiring understated | Full surface enumerated — `cli.py` args+dispatch+summary, `live_runs.py` ×4, `pipeline.py`, `__init__.py` (§12) |
| H2 | High | `max_timeout` doesn't bound a hung primitive | Three-layer bound; per-primitive deadlines flagged as a required #59 addition (§5, §8, §16) |
| H3 | High | "testable now" false — `context.py`/`binding.py` absent | `context.py` vendored; `build_custom_tools` lazily resolved; stub provider (§2, §8, §13) |
| M1 | Medium | Stub provider in `tests/` imported by product code | Shipped as `backend/agents/rlm/stub_primitives.py`, internal (§3, §13) |
| M2 | Medium | Event-store `append` contract underspecified | Dedicated aggregate, tracked `expected_version`, `EventEnvelope`, registered `RLMRunIteration` (§10) |
| M3 | Medium | `DashboardEmitter._emit` has no lock | The `emit` callable owns a `threading.Lock` and serializes writes (§9.3) |

---

*v2, 2026-05-21. Issue #60 — the RLM orchestrator, decoupled (Option 1) from #59's
primitive layer, Codex-review-hardened. Consumes `RunContext` + `build_custom_tools`
(lazily); ships a stub provider for standalone testability. Per-module contracts §6–§13;
delegation §15.*
