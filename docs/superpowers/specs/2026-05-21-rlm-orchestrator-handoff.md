# RLM Phase 3 — Orchestrator (#60): Session Handoff & Continuation Plan

> **Purpose.** Self-contained context to resume this work in a fresh Claude Code session
> with nothing lost. Written 2026-05-21 at the end of a long planning session; the
> Phase-2 digest (§8) is folded in — the new session does NOT need to re-run it.
>
> **To resume:** open a new session in `/home/abheekp/openresearch` and paste:
> *"Read `docs/superpowers/specs/2026-05-21-rlm-orchestrator-handoff.md` in full, then
> continue from §9. Use /iterate."*
>
> Read this top-to-bottom first. §2 (instructions), §8 (the seam), §9 (plan), and §10
> (open decisions) are load-bearing.

---

## 1. Who & where

- **User:** lolout1 / Abheek (`sww35@txstate.edu`).
- **Repo:** `/home/abheekp/openresearch`, branch **`rlm-pivot`**.
- **Remotes:** `origin` = `armaanamatya/openresearch` (canonical — push here). `replix` =
  `lolout1/Replix` (fork — **never push here**). Both repos are **public** — `gh` CLI is
  NOT logged in; use `curl https://api.github.com/repos/armaanamatya/openresearch/...`.
- **Stakes:** the RLM pivot is shown to a **Microsoft VP** and a **Deepinvent funder**.
  Bar = VP-grade, production rigor, genuinely elegant.

## 2. Standing instructions & working preferences — ALL carry forward

- **Execution model: Opus designs + reviews every diff; Sonnet sub-agents execute; NO
  Codex on the implementation** (not executing it, not reviewing it). Codex *adversarial
  review* is allowed for **specs / design docs** when the user explicitly asks.
- **Quality bar:** production-grade, modular, scalable, big-tech standards. **Root-level
  elegant solutions, not patches** — the smallest correct change + a guard test.
- **Process:** the `/iterate` discipline — recon → plan → **confirm with the user** →
  implement → verify. Brainstorm before creative work; present a design, get approval
  *before* writing code.
- **Delegation:** fan out **Sonnet** sub-agents for execution & recon; parallel; efficient
  — *without sacrificing quality* (held by a tight spec + Opus diff-review). Every
  delegated task gets exact paths, the contract, what NOT to touch, acceptance commands.
- **Grill the user** on design choices & tradeoffs before locking a design. Surface
  contradictions; never guess silently.
- **Integrate seamlessly** with teammates' work — Armaan's PR #65, Aayush's Phase 2 branch.
- **Git:** push to `origin`, never `replix`. Commit/push **only when the user asks**.
- **Memory** (auto-loaded each session via `MEMORY.md`): `project_rlm_pivot`,
  `feedback_delegation`, `feedback_implement_skill`, `feedback_solution_quality`,
  `feedback_git_remote`, `system_overview_doc`, `project_deepinvent_productionization`.

## 3. The RLM pivot — what it is

ReproLab reproduces research papers end-to-end and scores them against a PaperBench-style
rubric. It is being re-architected from a **14-stage `PipelineStage` state machine** to the
**RLM (Recursive Language Models)** paradigm (arXiv 2512.24601).

- **Engine: the `rlms` PyPI library** — `pip install rlms`, **import name `rlm`**. RESOLVED
  by spike (#64): NOT hand-built; NOT `dspy.RLM` (rejected — WASM can't run Docker; #66
  closed). Reconfirmed by the user 2026-05-21.
- **Canonical spec:** `docs/design/rlm-pivot-brief.md` ("RLM Pivot Plan", rewritten
  2026-05-20). Phase-1 artifact: `docs/rlm-pivot-mapping.md`.
- Driver #1 is **honesty** — the funding pitch was "RLM-based"; the implementation must be.

## 4. Issue map (renumbered 2026-05-21)

| Issue | Phase | Scope | Owner |
|---|---|---|---|
| #58 | 1 | `backend/agents/rlm/` skeleton + mapping doc — **done** (in PR #65) | Armaan |
| #59 | 2 | **Primitives** — extract the stage agents into `custom_tools` functions | **Aayush** |
| **#60** | **3** | **RLM orchestrator — THIS SESSION'S WORK** | **user (lolout1)** |
| #61 | 4 | Frontend redesign | — |
| #62 | 5 | End-to-end PaperBench runs | — |
| #63 | 6 | Cleanup | — |
| #64 | — | Umbrella (architecture fork RESOLVED → `rlm` library) | Armaan |
| #66 | — | dspy.RLM evaluation — **closed**, not adopted | — |

- **PR #65** (`rlm-pivot → main`, open) carries the Phase 1 skeleton + mapping doc + the
  2026-05-20 doc/brief rewrite. The skeleton's hand-built `repl_host.py` / `root_loop.py` /
  `sub_call.py` are **superseded by the `rlms` library** — #60 neither uses nor deletes them.
- Aayush's Phase 2 branch: **`origin/feat/rlm-phase2-foundation`** — has
  `docs/design/phase2-implementation-plan.md` (~1621 lines), `phase2-analysis.md` (~699),
  `rlms-spike-report.md` (~106), `tools/rlms_spike.py`. **Digested in §8/§10 below.**

## 5. #60 — the deliverable (the user's task)

From issue #60 ("RLM Pivot Phase 3: RLM orchestrator"). All new code in `backend/agents/rlm/`:

1. **`system_prompt.py`** — the reproduction-domain RLM system prompt →
   `rlm.RLM(custom_system_prompt=…)`. RLM operating principles + primitive signatures +
   `context`-variable metadata + in-context decomposition examples. No fixed workflow.
2. **`run.py`** — the run entry replacing the `PipelineStage` machine: builds `rlm.RLM(…)`,
   calls `.completion()`, writes `final_report.{json,md}`.
3. **SSE bridge** — per-subcall events via `on_subcall_*`; per-iteration events via a
   custom `RLMLogger` subclass `.log()` override (`on_iteration_*` never fire — §6).
4. **Event-store bridge + checkpoint/resume** — `rlm`'s `persistent=True` + the SQLite
   event store (brief §10 — `rlm` does not provide run-level checkpoint/resume).
5. **`pipeline.py` run modes** — `sdk` mode runs the RLM root via `run.py`; keep `offline`
   working or explicitly retire it.

Done condition: a real `rlm` run on a PaperBench paper executes primitive-by-primitive,
streams SSE, writes `final_report.{json,md}`. NOT in scope: primitive *implementations*
(#59), frontend rendering (#61).

## 6. `rlms` v0.1.1 — verified API facts (spike-confirmed 2026-05-21)

- `from rlm import RLM` — package `rlms`, **module `rlm`**. Installed in `.venv` (Py 3.12).
- `.completion(prompt) -> RLMChatCompletion` — **synchronous, blocks**, `exec`s root code
  on the calling thread. Use `.response` for the final result.
- `custom_tools = {name: {"tool": callable, "description": str}}`. Reserved names
  (collision → `ValueError`): `llm_query(_batched)`, `rlm_query(_batched)`, `FINAL_VAR`,
  `SHOW_VARS`, `context`, `history`.
- **`environment="local"` is MANDATORY.** `DockerREPL._build_exec_script` does **not**
  inject `custom_tools` into the in-container globals — under `environment="docker"` the
  primitives would not exist in the REPL (`NameError`). Docker is used **only inside** the
  `build_environment` / `run_experiment` primitives, via `RuntimeAppService` — never by the
  root REPL. (Issue #60's body and the spike report still say `"docker"` in places — a
  documented Task-0 correction.)
- **Termination = `FINAL(text)` / `FINAL_VAR(varname)`** (parsed by
  `rlm/utils/parsing.py::find_final_answer`). There is **no reserved `answer` variable**.
- `on_subcall_start/complete` **fire** (`rlm/core/rlm.py:739, 805`).
  `on_iteration_start/complete` are declared but **NEVER invoked in 0.1.1** —
  per-iteration SSE must come from an `RLMLogger` subclass: `rlm.RLM` calls
  `logger.log(iteration)` once per iteration (`rlm/core/rlm.py:367-368`).
- `max_depth=2` → an `rlm_query()` spawns a real nested child RLM; at the cap it degrades
  to a plain LM call. `rlm` 0.1.1 is early — pin `rlms==0.1.1`.

## 7. Current state & artifacts

- **v3 primitives spec** — `docs/superpowers/specs/2026-05-20-rlm-phase3-primitive-integration-design.md`.
  A thorough, Codex-reviewed (v2) + Opus-tailored **primitives** design. It is a **#59
  artifact**, NOT the #60 deliverable (it exists because #60 was *originally* scoped as
  "primitive integration" before the 2026-05-21 renumber). It diverges from Aayush's
  Phase 2 plan on real points — see §10.
- **Uncommitted / on disk (survive a session clear; lost only to `git clean`):**
  `docs/explainer/` (untracked — DeepInvent explainer docs); the v3 spec; this handoff
  file. A **git stash** "system_overview explainer pointer (pre-pull 2026-05-21)" holds a
  3-line `system_overview.md` edit.
- The Phase-2 digest and a Codex review of the v3 spec were in-flight; the digest is
  **folded into §8/§10**. The Codex-on-v3 review is moot for #60 (v3 is a #59 artifact).

## 8. The #59 → #60 seam — what `run.py` consumes (from the completed digest)

Aayush's `phase2-implementation-plan.md` defines exactly what #60 imports. **#60 must
align to this** (Aayush owns #59 — he is building this surface):

**`run.py` imports:**
```python
from backend.agents.rlm.context import RunContext        # Aayush's context.py
from backend.agents.rlm.binding import build_custom_tools # Aayush's binding.py
```

**`RunContext`** (Aayush's `backend/agents/rlm/context.py`):
```python
@dataclass
class RunContext:
    project_id: str
    project_dir: Path
    runs_root: Path
    dashboard: Any          # DashboardEmitter — SSE emission
    cost_ledger: Any        # RunCostLedger
    llm_client: Any         # protocol: .complete(*, system, user) -> str
    provider: str
    model: str
    runtime: Any = None     # AgentRuntime — only implement_baseline uses it
    workspace_service: Any = None
    workspace_id: str | None = None
```

**`build_custom_tools(ctx: RunContext) -> dict[str, {"tool": callable, "description": str}]`** —
closes `ctx` over each primitive (the wrapper injects `ctx` as a keyword arg, invisible to
the root; emits `primitive_call` SSE; appends a `CostLedgerEntry`). `primitives.py` exports
the 10 primitive functions + `PRIMITIVE_REGISTRY` + `PRIMITIVE_DESCRIPTIONS`.

**How `run.py` builds the RLM:**
```python
custom_tools = build_custom_tools(ctx)
rlm = RLM(
    backend="openai", backend_kwargs={"model_name": "<root model>"},
    environment="local",                 # MANDATORY — §6
    max_iterations=20, max_depth=2,
    custom_tools=custom_tools, custom_sub_tools={},
    custom_system_prompt=build_system_prompt(...),   # from #60's system_prompt.py
    logger=<RLMLogger subclass: .log() emits repl_iteration SSE>,
    on_subcall_start=<cb>, on_subcall_complete=<cb>,  # these fire
)
result = rlm.completion(context_dict)    # SYNC — call on a worker thread
```
`run.py` calls `.completion()` on a **worker thread** (`await asyncio.to_thread(...)`) so
the orchestrator event loop stays free; primitives bridge async work back to that loop.
The `emit`/`dashboard` path must be thread-safe (called from the worker thread).

## 9. Continuation plan

1. **Grill the user** — on §10's open decisions AND the orchestrator design tradeoffs:
   the SSE-bridge mechanism (`RLMLogger`-subclass vs alternatives), checkpoint/resume
   (`persistent=True` + SQLite — what's persisted, resume semantics), `offline`-mode
   retire-or-keep, how much of the `PipelineStage` machine `run.py` deletes (keep `main`
   runnable until cutover — brief §11), cost accounting (`usage_summary` + `cost_ledger`).
2. **Write the #60 orchestrator design spec** → `docs/superpowers/specs/2026-05-21-rlm-phase3-orchestrator-design.md`,
   aligned to the §8 seam. Brainstorming discipline: present in sections, get approval.
3. **(Optional) Codex adversarial review of the orchestrator *spec*** — only if the user asks.
4. **Implement** — Opus designs + reviews every diff; Sonnet sub-agents execute in
   parallel; **no Codex**. Verify with a real `rlm` run.

## 10. Open decisions — Aayush's #59 plan vs. the v3 spec (must resolve before `run.py`)

The digest cross-checked Aayush's `phase2-implementation-plan.md` against the v3 spec.
They **agree** on: `environment="local"`, the `{"tool","description"}` shape, a closure
factory, wrap-not-rewrite, `on_subcall_*`-for-SSE, the `RLMLogger`-subclass for iterations,
`propose_improvements` wrapping the existing `improvement-orchestrator` agent, and a
mock-`RLM` integration test. They **diverge** here — these need user + Aayush resolution:

1. **Context object — the deepest divergence.** Aayush's `RunContext` uses a plain
   `llm_client.complete()` + a direct `cost_ledger`, **no `event_loop`**. The v3 spec's
   `PrimitiveContext` routes LLM calls through a resilient `AgentInvoker`
   (`orchestrator._invoke_agent` — failover, `RunBudget`, health, telemetry) and carries
   `event_loop`, `EnvironmentRegistry`, `CorpusFingerprints`. **#60 must consume whatever
   #59 ships — default to Aayush's `RunContext`** — but push the v3 improvements
   (resilient invoker, runtime Algorithm-2 guard) to Aayush for #59.
2. **Async bridge — a concrete seam conflict.** Aayush's plan spins a **fresh event loop
   per async primitive call** (`asyncio.run` in a worker). The v3 spec uses a **shared,
   kept-alive loop** (`asyncio.run_coroutine_threadsafe`, with cancel-on-timeout + a
   same-loop guard). **Aayush's approach breaks once `run.py` keeps a loop alive** — must
   be reconciled. The v3 spec's mechanism is the correct one for #60.
3. **`env_id` handoff.** Aayush: `build_environment` returns a raw Docker image-tag string,
   passed directly to `run_experiment`. v3: an opaque token + `EnvironmentRegistry.resolve()`.
   The root model's REPL code differs — pick one.
4. **`implement_baseline → run_experiment` artifact.** Aayush: writes `code/commands.json`.
   v3: writes `baseline_result.json`. Pick one format.
5. **Algorithm-2 guard.** Aayush: naming-convention only (he flags "nothing enforces it at
   runtime"). v3: a real `_guard_no_corpus`. Recommend adopting the v3 runtime guard.
6. **Two primitives plans** — Aayush's 1621-line `phase2-implementation-plan.md` and the
   v3 spec must be reconciled to ONE before #59 is implemented.

Items 1–4 directly shape `run.py`. The natural resolution: **#60 aligns to #59's actual
shipped surface (Aayush's `RunContext`/`build_custom_tools`); the user feeds the v3 spec's
stronger choices (resilient invoker, shared-loop bridge, runtime guard) to Aayush.** But
confirm with the user — they may want to drive a reconciliation.

---

*Handoff written 2026-05-21. The user owns #60 (the RLM orchestrator). This session:
recon → planned the primitives (v3 spec, while #60 was still scoped as primitives) →
caught the architecture change to the `rlms` library → caught the 2026-05-21 issue
renumber (#60 → orchestrator) → user pivoted to #60 → digested Aayush's Phase 2 plan
(§8, §10). Continue from §9.*
