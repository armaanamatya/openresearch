# RLM Pivot — Phase 2 Analysis (issue #59)

> **Status: BLOCKED on an architecture decision. The executable checkbox
> implementation plan is intentionally deferred.**
>
> This document delivers the analysis for Steps 2–4 of the Phase 2 task plus a
> risk list. It does **not** contain a task-by-task checkbox plan, because
> issue #59 and the canonical brief `docs/design/rlm-pivot-brief.md` currently
> specify **mutually exclusive architectures** (see §4.1). Writing a plan now
> would silently commit Phase 2 to one of two contradictory specs. The plan
> should be written once the contradiction is reconciled — see §6 for what that
> requires and §4.6 for the ~60% of Phase 2 that is the same either way.
>
> Analysis pass, 2026-05-21. Read-only — no code was changed. Branch `rlm-pivot`.

---

## 0. Executive summary

The single most important finding: **PR #65 contradicts itself, and the
contradiction lands squarely on Phase 2.**

PR #65's five commits, in branch order:

| # | Commit | What it does |
|---|--------|--------------|
| 1 | `e92e11e` | Adds the **hand-built** `backend/agents/rlm/` skeleton — `repl_host.py`, `root_loop.py`, `sub_call.py`, `system_prompt.py`, `primitives.py` — and `docs/rlm-pivot-mapping.md` |
| 2 | `1a4a6de` | Freezes 6 **hand-build** design decisions in the mapping doc §6 |
| 3 | `529a3db` | Deletes pre-pivot docs |
| 4 | `d662475` | Rewrites README / `CLAUDE.md` / `system_overview.md` |
| 5 | `63adb10` | **Rewrites the canonical brief** — and the new brief §5 states: *"We do **not** write a REPL host, a root loop, or `sub_LLM`/`sub_RLM` — `rlms` provides all three."* |

So commit 5 — the **last** commit of the PR — rewrites the canonical spec to
forbid exactly the three modules commit 1 created. Issue #59 (Phase 2, assigned
to AayushBaniya2006) and issue #58/#64 were all written against the *old* brief:
they cite brief "Section 14" and "the preamble", both deleted by commit 5.

The 5 paper-accuracy corrections in issue #59's 2026-05-20 update (depth=2,
`FINAL_VAR` termination, no token cap, root-iteration cap, root-model knob) are
real and correct — but they are a **separate axis** from this fork. They tell
you *how to build Algorithm 1 accurately*; they do not address *whether to
hand-build it or depend on the `rlms` library*. The brief rewrite introduced
that second decision after #59 was written.

Other key findings:

- **Phase 1 (PR #65) landed** the mapping doc and a 6-file, ~451-line skeleton.
  Of the skeleton's 23 stub bodies, **14 raise `NotImplementedError("Phase 2
  (#59) …")`** and 9 raise `… ("Phase 3 (#60) …")`. The stub partition is
  correct (§3.3).
- The skeleton's leaf modules are import-clean (stdlib-only, verified). `import
  backend.agents.rlm` could **not** be verified here — it fails in the *parent*
  `backend/agents/__init__.py` because `pydantic_settings` is not installed in
  this checkout (an environment gap, not an rlm defect — §3.2).
- The `FINAL_VAR` / `FINAL` termination regexes in `root_loop.py` **false-positive
  on comments and string literals** (verified — §3.4 C3, §5 R2).
- `rlms` is **not installed** and **not in `backend/requirements*.txt`**; its API
  is unverified (§5 R6).
- The user (assignee) chose **"reconcile docs first"** when shown this fork.
  This document is that reconciliation input.

---

## 1. Sources read

Umbrella #64; issues #58 and #59 (+ both frozen-decision comments); the canonical
brief `docs/design/rlm-pivot-brief.md` (rewritten) and the old brief on `main`;
`docs/rlm-pivot-mapping.md`; all six `backend/agents/rlm/` skeleton files;
`backend/services/context/workspace/tools/rlm_query.py` (513 lines);
`backend/agents/orchestrator.py` (state machine + RLM helpers), `topology.py`,
`pipeline.py`; the PR #65 commit graph; `CLAUDE.md`. The rlm leaf modules were
imported and the termination regexes were probed empirically.

---

## 2. Current project analysis

### 2.1 The 14-stage `PipelineStage` machine

The pre-pivot architecture being decomposed:

- **`PipelineStage`** (`orchestrator.py:162-178`) — a 14-member string enum:
  `INGESTED → PAPER_UNDERSTOOD → ARTIFACTS_DISCOVERED → ENVIRONMENT_BUILT →
  PLAN_CREATED → GATE_1_PASSED → BASELINE_IMPLEMENTED → BASELINE_RUN →
  GATE_2_PASSED → IMPROVEMENTS_SELECTED → IMPROVEMENTS_RUN → GATE_3_PASSED →
  RESEARCH_MAP_GENERATED → COMPLETE`. Order is load-bearing.
- **`PipelineState`** (`orchestrator.py:181-216`) — a dataclass that accumulates
  results stage by stage; `advance_stage()` is the *only* sanctioned transition
  and atomically checkpoints `pipeline_state.json` (`orchestrator.py:218-305`).
- **`topology.py`** mirrors this for the UI as a 12-node / 14-stage / 3-gate
  graph (`topology.py:184-199` for stages, `:178-182` for gates).
- **Two run modes** (`pipeline.py`): `run_pipeline_sdk()` (async, LLM-driven,
  the primary mode) and `run_pipeline_offline()` (sync, deterministic, no LLM).
- **Gates 1/2/3** are inline control-flow checkpoints — `run_gate_1()` etc.
  (`orchestrator.py:1537+`), `run_gate_offline()` in offline mode. They can
  halt a run (`blocked_requires_human`) unless fail-soft modes are on.
- **Five hardcoded improvement paths** (optimizer / backbone / augmentation /
  horizon / diffusion) live in `topology.py:119-144`.
- **RLM is already present but dormant.** `ReproLabOrchestrator` builds an
  `RlmQueryTool` (`orchestrator.py:475-482`), with helpers `_build_rlm_llm_client`
  (`:532-540`), `_rlm_query` (`:542-560`), `_rlm_evidence_for_stage` (`:562-571`),
  and a `rlm_calls_remaining` budget (`:216`, default 120). The comment at
  `:475` says it is *"dormant until stages call `_rlm_query()`"* — there is no
  production caller.

### 2.2 What the pivot keeps / replaces / deletes

Cross-checking brief §4/§5/§6 against the code:

| Disposition | Item | Notes |
|---|---|---|
| **Keep** | `PaperExtractor` | Produces the paper text loaded into the REPL. |
| **Keep** | Docker / RunPod sandboxes | Wrapped by `build_environment` / `run_experiment` primitives. |
| **Keep** | Env build-and-repair loop | Becomes `build_environment()`; retry loop stays *inside* the primitive. |
| **Keep** | Rubric verifier agent | Becomes `verify_against_rubric()`. |
| **Keep** | Stage-agent **core logic** | Extracted as plain functions (Phase 3 / #60 — out of scope here). |
| **Keep** | PaperBench bundle, SQLite event store, `assumption_ledger.json` / `cost_ledger.jsonl` / `agent_telemetry.jsonl`, lab UI shell | Ledgers now emitted from inside primitives. |
| **Keep (disputed)** | `rlm_query.py` | **Brief §4** says *"superseded by the `rlms` library … keep as reference or retire; do not invest further."* **Issue #59 #2** says *"reuse `rlm_query.py`'s `_recursive_query` as the `sub_RLM` engine."* — direct contradiction (drift D3). |
| **Replace** | `PipelineStage` machine | → RLM root loop (Algorithm 1). |
| **Replace** | Gates 1/2/3 control-flow | → `verify_against_rubric()` primitive called when the root judges useful — no fixed checkpoints. |
| **Replace** | 5 hardcoded improvement paths | → dynamic `propose_improvements()`, variable-length, proposer-tagged. |
| **Replace** | Fixed 14-stage UI strip | → dynamic exploration tree + REPL panels (Phase 4 / #61). |
| **Delete** | `PipelineStage` enum + stage-advancement logic | brief §6. |
| **Delete** | Gate control-flow, `PPO_HYPOTHESES` 5-path fanout (`topology.py`) | brief §6. |

The keep/replace/delete picture is internally consistent **except** for
`rlm_query.py` (D3) — and the brief↔issue architecture fork (§4.1).

### 2.3 Where Phase 1 stands — landed vs stubbed

**Landed (PR #65):** `docs/rlm-pivot-mapping.md` (185 lines); the
`backend/agents/rlm/` skeleton (6 files, ~451 lines); doc rewrites; deletion of
26+ pre-pivot docs.

**Implemented in the skeleton (real code, not stubs):** `__init__.py` exports;
the `ReplOutput`, `RootHistoryEntry` dataclasses and `RootIterationCapExceeded`
exception; `SubCallBudget` (with working `can_spend` / `record`);
`PRIMITIVE_REGISTRY` (a populated 10-key dict); all module constants; the two
termination regexes.

**Stubbed — `NotImplementedError` inventory (23 stubs):**

| File | Symbol | Stub target |
|---|---|---|
| `repl_host.py` | `ReplHost.__init__`, `.exec`, `.has_variable`, `.read_variable`, `.serialize`, `.resume` | #59 (×6) |
| `root_loop.py` | `RootLoop.__init__`, `RootLoop.run`, `parse_final_tag` | #59 (×3) |
| `sub_call.py` | `bind_sub_calls`, `llm_query` (stub), `rlm_query` (stub) | #59 (×3) |
| `system_prompt.py` | `build_system_prompt` | #59 (×1) |
| `primitives.py` | `set_final` | #59 (×1) |
| `primitives.py` | `understand_section`, `extract_hyperparameters`, `detect_environment`, `build_environment`, `plan_reproduction`, `implement_baseline`, `run_experiment`, `verify_against_rubric`, `propose_improvements` | #60 (×9) |

**14 stubs → Phase 2 (#59), 9 → Phase 3 (#60).** The partition is correct:
Phase 2 owns the REPL/loop/sub-call machinery + the registry shell + `set_final`;
Phase 3 owns the 9 domain-primitive bodies (stage-agent extraction). This matches
the task constraint *"design the registry that will hold them, not the
extraction."*

### 2.4 Documentation drift

| ID | Drift | Severity |
|---|---|---|
| **D1** | **Brief ⇄ issues #58/#59/#64 — architecture.** Rewritten brief §3/§5 mandates depending on the `rlms` library and explicitly forbids hand-writing the REPL host / root loop / sub-calls. Issues #58/#59/#64 + the skeleton + mapping doc mandate hand-building exactly those. Issues cite brief "Section 14" and "the preamble" — both deleted by commit `63adb10`. | **Blocking** |
| **D2** | **Brief ⇄ issues — phase numbering.** Brief §11: Phase 1 *Spike* / Phase 2 *Primitives* / Phase 3 *Orchestrator+prompt*. Issues #58–63: Phase 1 *Prep+mapping* / Phase 2 *Backend foundation (REPL/loop/sub-calls)* / Phase 3 *Primitive integration*. Brief-Phase-2 ≈ issue-Phase-3; they are different decompositions. | High |
| **D3** | **Brief §4 ⇄ issue #59 #2 — `rlm_query.py`.** Brief: "do not invest further in it." Issue #59: "reuse `_recursive_query` as the `sub_RLM` engine." | High |
| **D4** | **Mapping doc ⇄ rewritten brief — section refs.** Mapping doc cites brief "§7.7", "§13 FM#1–#10", "correction #1/#8". The rewritten brief has none of these — it has §8 "RLM fidelity invariants" (9 items). The mapping doc references the *old* brief's structure throughout. | Medium |
| **D5** | **Mapping/skeleton ⇄ brief — termination model.** Skeleton + mapping use `FINAL_VAR(name)` tag + a `set_final` primitive (10 primitives in mapping §2). Rewritten brief §7 lists 9 primitives and terminates via the library's reserved `answer` variable — no `set_final`, no `FINAL_VAR`. | Medium |
| **D6** | **`CLAUDE.md` ⇄ skeleton.** `CLAUDE.md` (rewritten in PR #65, commit `d662475`) says the target is *"an RLM-based orchestrator built on the `rlms` library."* It agrees with the brief, **not** the skeleton. Within PR #65, README + `CLAUDE.md` + brief all say `rlms`; only the skeleton (`e92e11e`) + mapping §6 (`1a4a6de`) say hand-build. | Medium |
| **D7** | **`CLAUDE.md` Python version.** `CLAUDE.md` "Common commands" still says "Backend (Python 3.11…)". Verified actual: **Python 3.14.2**. | Low |
| **D8** | **`CLAUDE.md` stale file ref.** `CLAUDE.md` cites `frontend/src/components/repro-lab-client.tsx`; the old brief preamble itself already corrected this to `frontend/src/components/lab/lab-shell.tsx`. | Low |
| **D9** | **Root-iteration cap value.** Skeleton `DEFAULT_MAX_ROOT_ITERATIONS = 20` (paper Appx A). Brief §3 table says `rlms`' `max_iterations` default is **30**. Under the `rlms` path the wrapper must explicitly pass `max_iterations=20`. | Low |

D1 is the reason this analysis stops short of a plan. D6 is the sharpest tell:
**the brief author changed direction mid-PR** — the doc rewrites (commits 3–5)
adopt `rlms`, but the skeleton and mapping §6 (commits 1–2) were never revised to
match, and neither were issues #58/#59/#64.

---

## 3. PR #65 audit against issue #58's done-condition

Issue #58's done-condition: `docs/rlm-pivot-mapping.md` exists with (a) each stage
agent → its target primitive, (b) function signatures for all brief §7 primitives,
(c) the `rlm_query.py` survives-vs-new audit; plus the `backend/agents/rlm/`
skeleton with `import backend.agents.rlm` succeeding and all stubs raising
`NotImplementedError`.

### 3.1 Mapping doc completeness

- **(a) Stage agent → primitive — ✓.** Mapping §1 maps all 9 stage modules.
  `artifact-discovery` correctly maps to "no primitive — folded into REPL init"
  (the `repo_files` / `prior_work_refs` variables).
- **(b) Signatures for all brief §7 primitives — ✓ (against the old brief).**
  Mapping §2 lists **10** signatures: the 9 primitives + `set_final`. Against the
  *rewritten* brief §7 (which lists 9 and uses the `answer` variable, not
  `set_final`), `set_final` is an extra (drift D5).
- **(c) `rlm_query.py` survives-vs-new audit — ✓ (against the old brief).**
  Mapping §3 is thorough — but it classifies `repl_host.py` / `root_loop.py` /
  `sub_call.py` as **"NEW"** code to write. Against the rewritten brief those
  three are "do not write." §3 answers the *old* brief's question.

**Verdict:** the mapping doc fully satisfies #58's done-condition *as #58 defined
it*. It does not reflect the rewritten brief — because the brief was rewritten in
a later commit of the same PR.

### 3.2 Skeleton completeness & import-cleanliness

- All 6 files present; every name in `__init__.py.__all__` resolves. Verified:
  the five leaf modules (`repl_host`, `root_loop`, `sub_call`, `system_prompt`,
  `primitives`) **import clean** — stdlib-only, no backend coupling.
- **`import backend.agents.rlm` could not be verified to succeed here.** It fails
  at `backend/agents/__init__.py:3` → `registry` → `runtime.factory` →
  `backend.config` → `import pydantic_settings` → `ModuleNotFoundError`. This is
  an **environment gap** (dev dependencies not installed in this checkout), not an
  rlm-skeleton defect. In a provisioned venv it would succeed. Note for Phase 2:
  importing the `rlm` *subpackage* runs the heavy parent `backend/agents/__init__.py`
  — the rlm package is import-time-coupled to the whole backend, so even a
  "standalone" rlm test drags in the full backend import graph.
- The skeleton is faithful to **issue #59** (5 hand-built modules). It is **not**
  faithful to the **rewritten brief** (which wants only `primitives.py` +
  `system_prompt.py` + a `run.py`, with `rlms` as a dependency).

### 3.3 Stub `NotImplementedError` audit — ✓

Every stub raises `NotImplementedError`; every message names the owning issue
(#59 or #60). Partition verified correct (§2.3 table). One subtlety worth noting
in the plan: `primitives.py`'s 9 domain functions point at #60 ("Phase 3 — wrap
…"), but `set_final` points at #59 — so Phase 2 owns the registry shell + the
event/ledger wrapper + `set_final`, and Phase 3 owns the domain bodies.

### 3.4 Phase-1 stub decisions that constrain Phase 2

These are the decisions baked into the skeleton that Phase 2 inherits. **C3, C4,
C7, C11 are traps** — they look done but are under- or mis-specified.

- **C1 — `DEFAULT_SUB_RLM_DEPTH = 2`** (`sub_call.py:26`). Correct per correction
  #1. Constraint: `rlm_query` must genuinely use depth 2, and "depth" must mean
  RLM-recursion depth — not `_recursive_query`'s content-chunking depth (see C7).
- **C2 — `DEFAULT_MAX_ROOT_ITERATIONS = 20`, `DEFAULT_MAX_OUTPUT_TOKENS_PER_TURN
  = 4096`** (`root_loop.py:31-32`). Correct per correction #4 / paper Appx A.
  `RootIterationCapExceeded` is already defined; `RootLoop.run()` must raise it.
- **C3 — `FINAL_VAR_TAG_RE` / `FINAL_TEXT_TAG_RE`** (`root_loop.py:39-40`).
  **Verified trap.** Empirically:
  - `FINAL_VAR_TAG_RE` matches inside a comment: `x=1 # FINAL_VAR(report)` →
    captures `report`.
  - `FINAL_VAR_TAG_RE` matches inside a string literal: `"call FINAL_VAR(foo)
    when done"` → captures `foo`.
  - `FINAL_TEXT_TAG_RE` (`FINAL\((.+?)\)`, non-greedy) matches in prose and stops
    at the first `)` — `FINAL(f(x))` mis-parses; `FINAL(a) and FINAL(b)` yields
    `['a','b']`.
  - `FINAL_TEXT_TAG_RE` does *not* match `FINAL_VAR(...)` (verified) — the two are
    disjoint, so check order is for precedence, not correctness.

  `root_loop.py`'s docstring says `parse_final_tag` scans `code or stdout`. As
  committed, the regexes scan raw text — so a root model that writes the literal
  string `FINAL_VAR(...)` in a comment, docstring, example, or while constructing
  the system prompt **terminates the run prematurely with the wrong variable**.
  The docstring promises "safeguards"; the regexes do not implement them. Phase 2
  must parse a *designated channel* (e.g. the tag alone on the last output line),
  or strip strings/comments via `ast` before scanning. **This is the single
  most consequential Phase-1 decision Phase 2 inherits.**
- **C4 — `PRIMITIVE_REGISTRY: dict[str, Callable]` holds bare module functions**
  (`primitives.py:113-124`). Issue #59 #4 requires each primitive to "emit a
  `primitive_call` SSE event" and "update `cost_ledger.jsonl`" — but a bare
  function `understand_section(text_slice: str) -> dict` has **no run context**
  (project id, event sink, ledger path). The registry is fine as a *catalog*, but
  it is not what the REPL should receive. Phase 2 needs a binding factory —
  `bind_primitives(*, run_context) -> dict[str, Callable]` — mirroring the
  `bind_sub_calls` pattern. The skeleton has `bind_sub_calls` but **no
  `bind_primitives`**; that asymmetry is a Phase-2 gap.
- **C5 — `ReplHost.__init__(project_dir: Path)` is underspecified.** Its docstring
  says it must populate globals with `paper_text` etc., the primitive registry,
  and bootstrap `llm_query` / `rlm_query` — but none are parameters. The sub-call
  closures come from `bind_sub_calls()`, the bound primitives from a factory (C4),
  the paper data from disk. Phase 2 must redesign this signature (e.g.
  `ReplHost(project_dir, *, repl_variables, primitives, sub_calls)`).
- **C6 — no event-sink parameter.** `RootLoop.__init__` and `bind_sub_calls` have
  no event-emitter argument, yet their docstrings require emitting `repl_iteration`,
  `variable_update`, `primitive_call`, `sub_rlm_spawned`. Phase 2 must add an
  event-sink dependency to both.
- **C7 — `_recursive_query` as the `sub_RLM` engine: depth-semantics mismatch.**
  `rlm_query.py::_recursive_query` (`:240-286`) is a fixed *chunk → select →
  recurse-on-chunks → aggregate* summarizer. Its `state.max_depth` bounds
  content-chunking recursion. Issue #59 #2 says invoke it "with `depth=0,
  max_depth=sub_rlm_depth`" (=2). But the paper's `sub_RLM` "depth=2" means a
  sub-call may itself be an RLM that writes code and spawns its own sub-calls.
  Setting `_recursive_query`'s `max_depth=2` controls how many times a big string
  is split — **not** whether nested RLMs spawn. `_recursive_query` has no REPL and
  no code execution. Reusing it as `sub_RLM` yields a recursive *summarizer*, not
  a recursive *language model*. (See risk R4.)
- **C8 — two budgets.** `_recursive_query` has its own `_RecursionState`
  (`max_llm_calls = 24`, `rlm_query.py:68`); `sub_call.py` adds a run-wide
  `SubCallBudget` (50 calls / $10). Phase 2 must make every `_recursive_query`
  LLM call also decrement the run-wide `SubCallBudget`, or the run-wide cap is
  silently bypassed. The skeleton does not wire this.
- **C9 — `__init__.py` exports the non-functional `llm_query` / `rlm_query`
  stubs.** `from backend.agents.rlm import llm_query` returns a stub that raises
  `NotImplementedError`; the real callables come from `bind_sub_calls()`.
  Documented, but a footgun.
- **C10 — `ReplOutput` carries full `stdout`** (`repl_host.py:20`). Algorithm 1
  requires only *metadata* in root history. `RootLoop` must extract
  `{length, prefix, has_traceback, var_assignments}` and never append
  `ReplOutput.stdout`. A careless append re-creates FM#3 (risk R9).
- **C11 — `set_final` vs `FINAL_VAR` — underspecified termination.**
  `set_final(report: dict) -> None`'s docstring says it "binds `report` to a REPL
  variable and emits `FINAL_VAR(report)`." But a function cannot make the *model*
  emit a tag, and `parse_final_tag` scans model *output*. Phase 2 must decide
  whether termination is (i) tag-parse only, or (ii) tag-parse **or**
  `set_final`-was-called-this-iteration. Mixing the two without a single
  authority is exactly the ambiguity FM#3 warns against.
- **C12 — the Algorithm-2 guard is a convention, not an enforced constraint.**
  The invariant "no primitive accepts `paper_text` / `supplementary_text` /
  `repo_files` as a whole-corpus argument" (brief §8.2 — *primitives take slices,
  not the corpus*; mapping §1) is encoded in Phase 1 only as (i) the `primitives.py`
  module docstring and (ii) the signature shapes themselves (`text_slice: str`,
  `method_spec: dict`, `env_spec: dict` — slices and structured specs). **Nothing
  enforces it:** a Phase 2 or Phase 3 change that adds a `paper_text: str`
  parameter to a primitive compiles and runs. Phase 2 inherits the obligation to
  keep the signature shapes intact and should harden the guard — a runtime
  assertion in the `bind_primitives` wrapper (e.g. reject any argument whose
  length matches a known corpus variable), or at minimum an explicit code-review
  checklist item. Without that, FM#1's flaw silently re-enters one level below the
  root (this is risk R4's structural sibling).

### 3.5 Paper-accuracy gaps in PR #65

- The skeleton's numeric constants are all paper-correct (depth 2, 20 iters,
  4096 tokens, 50 calls / $10).
- But the skeleton **hand-builds Algorithm 1**, and the rewritten brief (same PR)
  argues the *faithful* move is to depend on `rlms` — *"a reviewer sees
  `from rlm import RLM`, not our re-derivation of the paradigm"* (brief §3).
- `parse_final_tag`'s regexes false-positive (C3) — premature termination.
- `_recursive_query`-as-`sub_RLM` (C7) is not a recursive language model.
- `set_final` + `FINAL_VAR` are two termination mechanisms with an unspecified
  interaction (C11); and the brief uses a *third* model (`answer`) (D5).

These feed the risk list (§5).

---

## 4. Phase 2 (issue #59) analysis

### 4.1 The architecture fork — why the plan is deferred

| | **Path A — hand-build (issue #59)** | **Path B — `rlms` library (rewritten brief)** |
|---|---|---|
| Source of truth | Issues #58/#59/#64, mapping doc, skeleton | `docs/design/rlm-pivot-brief.md` §3/§5/§11 |
| `repl_host.py` | Implement (REPL host, exec, pickle checkpoint) | **Delete** — `rlms` provides the REPL |
| `root_loop.py` | Implement (Algorithm 1, `FINAL_VAR` parser) | **Delete** — `rlms` provides the root loop |
| `sub_call.py` | Implement (`llm_query` / `rlm_query`) | **Delete** — `rlms` provides `llm_query` / `rlm_query` |
| `system_prompt.py` | Implement (passed to `RootLoop`) | Keep — passed as `custom_system_prompt` |
| `primitives.py` | Implement (registry + wrapper + `set_final`) | Keep — assembled into `custom_tools` |
| New | standalone mock-paper harness | `run.py` (builds `RLM(...)`, wires `on_*` callbacks) |
| Termination | `FINAL_VAR(name)` tag + `set_final` | reserved `answer` variable |
| External dep | none new | `pip install rlms` (**unverified** — §5 R6) |

Three of issue #59's five named deliverables (`repl_host.py`, `root_loop.py`,
`sub_call.py`) **do not exist as work items** under Path B. Only `primitives.py`
and `system_prompt.py` survive in both. The 5 paper-accuracy corrections apply to
**both** paths (they become `rlms` constructor arguments under Path B).

The assignee selected **"reconcile docs first."** Accordingly, §4.2–§4.5 below
analyze issue #59's contract (the Path A reading, since that is what #59 names),
and §4.6 isolates the part that is identical under both paths. The checkbox plan
is written after reconciliation (§6).

### 4.2 The five deliverables — contracts

Contracts as issue #59 specifies them. "(P2-add)" marks a parameter the skeleton
omits that Phase 2 must add.

**Deliverable 1 — `repl_host.py` :: `ReplHost`**
- *Inputs:* `project_dir: Path`; (P2-add) the seed REPL variables (`paper_text`,
  `paper_metadata`, `supplementary_text`, `repo_files`, `prior_work_refs`,
  `rubric_spec`), the bound sub-calls dict, the bound primitives dict.
- *`exec(code: str) -> ReplOutput`:* runs `exec(code, self._globals)` in the
  persistent namespace; captures stdout via `contextlib.redirect_stdout`; derives
  `var_assignments` via `ast.parse` over `ast.Assign`/`ast.AugAssign` targets;
  returns `ReplOutput{stdout, length, prefix≤200, has_traceback, var_assignments}`.
  *Does not raise* on root-code errors — a traceback is captured into
  `has_traceback=True` so the root sees its own error as stdout metadata.
- *`has_variable(name) -> bool`, `read_variable(name) -> Any`:* namespace
  introspection for `FINAL_VAR` resolution. `read_variable` on a missing name →
  raise a typed error (e.g. `KeyError`).
- *`serialize(path: Path) -> None`:* pickle the *data* globals; large strings →
  file refs (`_paper_text_path` …); strip non-picklable handles **and the
  injected callables** (re-bound on resume, not restored).
- *`resume(project_dir, path) -> ReplHost`:* classmethod; restore data, re-bind
  callables.
- *Events:* none directly (`RootLoop` emits).

**Deliverable 2 — `sub_call.py` :: `bind_sub_calls`**
- *Inputs:* `llm_client: LlmClient` (sync), `budget: SubCallBudget`,
  `sub_rlm_depth: int = 2`; (P2-add) event sink.
- *Output:* `dict` exposing `llm_query(prompt, model="default") -> str` and
  `rlm_query(context, query) -> str` as REPL callables (closures over client +
  budget). **Not** `tool_use` blocks (FM#2).
- *`llm_query`:* one `llm_client.complete(system=…, user=prompt)`; records cost on
  `budget`; emits a sub-call event.
- *`rlm_query`:* constructs `RlmQueryTool` + `_RecursionState(max_depth=2,
  max_llm_calls=24)`, calls `_recursive_query(context, query, state, depth=0)`; at
  the depth cap `_recursive_query` already degrades to `_leaf_answer` on truncated
  content (its base case 2) — that *is* the documented `llm_query` fallback; emits
  `sub_rlm_spawned`.
- *Errors:* budget exhausted — recommend a typed `SubCallBudgetExceeded` the
  caller catches, consistent with `_recursive_query`'s own graceful degradation.
- *Events:* sub-call start/complete; `sub_rlm_spawned` (rlm_query, depth ≥ 1).

**Deliverable 3 — `system_prompt.py` :: `build_system_prompt`**
- *Inputs:* `repl_variables: dict[name -> {type, length, …}]`,
  `primitive_signatures: list[str]`, `root_model: str = "default"`.
- *Output:* `str` — Appendix-C-derived: RLM operating principles; **≥1 in-context
  decomposition example** (Fig 4a); REPL variables described by name + type +
  length (**never values**); primitive signatures + one-line notes; the
  termination tag; per-model addenda (Qwen anti-over-subcall line). No workflow
  prescription. No token cap.
- *Errors / Events:* none.

**Deliverable 4 — `primitives.py`**
- *Phase 2 scope:* the `PRIMITIVE_REGISTRY` catalog (done); `set_final`; the
  cross-cutting wrapper giving each primitive a `primitive_call` SSE event +
  `cost_ledger.jsonl` append. Recommended: a `bind_primitives(*, run_context) ->
  dict[str, Callable]` factory (C4). The 9 domain bodies stay `#60` stubs (or
  mocks for the Phase 2 harness).
- *`set_final(report: dict) -> None`:* contract underspecified (C11) — Phase 2
  must define how it binds into the REPL namespace and how it interacts with
  `parse_final_tag`.
- *Events:* `primitive_call` per call; one `cost_ledger.jsonl` row per call.

**Deliverable 5 — `root_loop.py` :: `RootLoop`**
- *Inputs:* `repl_host`, `llm_client`, `system_prompt`, `max_iterations = 20`,
  `max_output_tokens_per_turn = 4096`; (P2-add) event sink, checkpoint hook.
- *`run() -> Any`:* Algorithm 1 — initial history = `[Metadata(REPL vars)]`; loop
  ≤ `max_iterations`: call LLM → emit `repl_iteration` → `repl_host.exec(code)` →
  emit `variable_update` → append `code` + `Metadata(stdout)` (never raw stdout)
  → `parse_final_tag` → on `FINAL_VAR(name)` with the variable present, return
  `repl_host.read_variable(name)` → checkpoint. Loop exhausted → raise
  `RootIterationCapExceeded`.
- *`parse_final_tag(text) -> (kind, value)`:* with the C3 safeguards.
- *Errors:* `RootIterationCapExceeded`.
- *Events:* `repl_iteration`, `variable_update`, `run_complete` (+ sub-call /
  primitive events bubbling up). The root payload must **never** contain a
  `paper_text` substring (FM#1).

### 4.3 Hard design problems & recommended resolutions

The six "frozen decisions" (mapping doc §6 / issue #59 comment) resolve much of
this. Where they do, that is stated; where they leave a gap, a resolution is
recommended.

**(a) Sync/async bridge.** *Frozen:* a worker thread owned by the orchestrator's
async loop runs the synchronous `exec(code, namespace)`; `LlmClient.complete()`
stays sync and blocks the *worker thread*, not the event loop. *Analysis:* sound
and matches the paper's reference implementation. Caveat: `ClaudeLlmClient.complete()`
itself calls `asyncio.run(self._async_complete(...))` (`rlm_query.py:485-487`) —
calling `asyncio.run` from a worker thread is acceptable (the thread has no
running loop) but spins a fresh loop per call. For the `run_experiment` primitive,
the frozen decision is `asyncio.run_coroutine_threadsafe` onto the orchestrator
loop, then block on the future. **Architecture-independent:** the bridge is
needed under Path B too — `rlms`' REPL `exec`s synchronous code, and our async
primitives still need to cross back.

**(b) Pickling non-picklable REPL globals.** *Frozen:* large strings → on-disk
file refs (`_paper_text_path` …); non-picklable handles stripped on serialize,
re-issued "pending" on resume. *Analysis (gap):* the **injected callables**
(`llm_query`, `rlm_query`, bound primitives) close over LLM clients / sockets —
they are not picklable and must **not** be pickled; they are re-bound on resume,
not restored. Recommend an explicit allowlist of picklable *data* variable names
(or a denylist of the injected names) so `serialize` pickles data only and
`resume` re-runs `bind_sub_calls` / `bind_primitives`. Also: keep primitive
return values plain JSON-able dicts (the signatures already say `-> dict` / `->
str`) so primitive *results* stored in REPL variables stay picklable.

**(c) `FINAL_VAR` parsing + an answer that exceeds the context window.** Two
sub-problems. *Parsing* — C3: do not scan raw code. Recommend requiring the tag
alone on the model output's final line, or an `ast`-based scan that ignores
string/comment tokens. *Oversized answer* — this is the point of the design: the
answer is **read from the REPL variable** (`repl_host.read_variable(name)`) and
the orchestrator writes `final_report.{json,md}` from it directly. It never
passes through a model prompt, so a multi-megabyte report is fine.
**Architecture-independent:** Path B's `answer` variable works the same way.

**(d) `sub_RLM` reuse at depth 2 + degrade-at-cap.** `_recursive_query` already
implements the depth-cap → `_leaf_answer` fallback. The real problem is C7:
`_recursive_query` is a summarizer, not a nested RLM. Options:
(i) accept it as a "good enough" recursive sub-query and document that it is not
a nested Algorithm-1 loop (cheap; ships now; weak on fidelity invariant §8.5);
(ii) make `sub_RLM` a genuine nested `RootLoop` (faithful; more work);
(iii) Path B — `rlms`' `rlm_query` is a real nested RLM.
*Recommendation:* flag for reconciliation — if `sub_RLM` fidelity matters
(invariant §8.5 — "the root actually recurses"), (i) is insufficient.

**(e) Primitive events + cost ledger.** C4 — bare registry functions cannot emit
per-run events. Recommend `bind_primitives(*, run_context)`: each wrapped
primitive emits `primitive_call` on entry, runs the body, emits the outcome, and
appends one `cost_ledger.jsonl` row (the ledger format already exists —
`RunCostLedger.load_jsonl`, `orchestrator.py:453`). The wrapper closes over the
ledger path + event sink. **Architecture-independent:** Path B's `custom_tools`
callables need the identical wrapper.

### 4.4 Verification surface (the tests issue #59 names)

Per-deliverable tests (#59):

- **repl_host:** load a string variable, `exec` code that slices it, retrieve the
  result.
- **sub_call:** `sub_LLM` → exactly one API call; `sub_RLM` → the nested loop runs
  at depth 2.
- **system_prompt:** the prompt contains ≥ 1 in-context decomposition example (no
  token-cap test).
- **primitives:** each primitive callable from the REPL with correct outputs and
  events (Phase 2: registry + `set_final` + wrapper; the 9 domain bodies are
  Phase 3 / #60).
- **root_loop:** a standalone RLM root on a *mock paper with mocked primitives*
  terminates via a parsed `FINAL_VAR` tag that reads from REPL state, respects the
  root-iteration cap, and emits the full SSE schema.

Anti-pattern tests (#59 + brief §8) — these are the fidelity guard:

1. The root model's message payload never contains a `paper_text` substring (FM#1).
2. `sub_LLM` / `sub_RLM` are not in the root's `tools` list as `tool_use` blocks
   (FM#2).
3. The run terminates only when a `FINAL_VAR(name)` tag is parsed **and** the
   named REPL variable is read out (FM#3).
4. The root loop respects `MAX_ROOT_ITERATIONS` and raises
   `RootIterationCapExceeded` rather than hanging.
5. At depth 2 a `sub_RLM` call emits `sub_rlm_spawned`; at depth 1 it falls back
   to `llm_query` with no spurious event.
6. The system prompt contains ≥ 1 in-context decomposition example.

A worthwhile addition not in #59: an explicit C3 regression test —
`parse_final_tag` must **not** terminate on `FINAL_VAR(...)` appearing inside a
comment or string literal.

### 4.5 Build order & dependencies

Even without a checkbox plan, the dependency DAG is fixed (Path A):

```
repl_host.py ───┐
                ├─→ root_loop.py ─→ mock-paper harness + anti-pattern tests
sub_call.py ────┤        ↑
                │        │
primitives.py ──┴────────┤   (registry + bind_primitives + set_final)
                         │
system_prompt.py ────────┘   (needs the final variable list + primitive sigs)
```

- `repl_host.py` and `sub_call.py` are foundational and mutually independent —
  parallelizable.
- `parse_final_tag` (inside `root_loop.py`) is pure text logic — build and test it
  early and independently of the rest of `root_loop.py`.
- `primitives.py` (Phase 2 scope) depends on `repl_host`'s namespace-binding
  contract being fixed (for `set_final`).
- `system_prompt.py` depends on the final REPL-variable list and primitive
  signatures being settled.
- `root_loop.py` wires all four — build it last.
- The standalone mock-paper harness and the anti-pattern tests come last.

### 4.6 What is constant across both architectures

This is the reconciliation-useful conclusion. **Roughly 60% of Phase 2 is
identical under Path A and Path B:**

- **`primitives.py`** — the domain primitives plus the event/ledger wrapper. Path
  A exposes them as REPL functions; Path B assembles them into `custom_tools`. The
  *domain logic and the wrapper are the same.*
- **`system_prompt.py`** — the Appendix-C-derived prompt. Path A passes it to
  `RootLoop`; Path B passes it as `custom_system_prompt`. *Same content.*
- The sync/async bridge for async primitives (`run_experiment`).
- The SSE event schema and emission wiring.
- Checkpoint/resume — the brief §10 explicitly says `rlms` does **not** provide
  this; we own it under either path.
- All 5 paper-accuracy corrections (depth 2, `FINAL_VAR`/`answer` termination, no
  token cap, root-iteration cap, root-model knob).

**What actually differs is only `repl_host.py` / `root_loop.py` / `sub_call.py`**
— hand-written under Path A, deleted under Path B. So the reconciliation decision
is low-stakes for most of Phase 2 and high-stakes for those three modules. Phase 2
could even begin on the constant 60% (`primitives.py`, `system_prompt.py`) while
the fork is resolved — see §6.

---

## 5. Risk list

Anything in PR #65 or the brief that, taken at face value, produces a
paper-inaccurate RLM or a broken Phase 2. Ordered by severity.

**R1 — Self-contradictory canonical spec (blocking).** PR #65's last commit
rewrites the brief to mandate `rlms`; the same PR ships a skeleton, a mapping
doc, and an issue (#59) mandating the opposite. Building Phase 2 against either
"at face value" builds against a spec the team's own newest document contradicts.
*Not* a paper-accuracy defect — a spec-integrity defect — but it is risk #1, and
it must be resolved before any Phase 2 code merges. *(Brief §3/§5 ⇄ issue #59 ⇄
mapping doc.)*

**R2 — `FINAL_VAR` regex false-positives → premature/incorrect termination.**
Verified: `FINAL_VAR_TAG_RE` matches inside comments and string literals. Taken
at face value, the skeleton's `parse_final_tag` ends a run the moment the root
model writes the literal text `FINAL_VAR(...)` anywhere — including while building
a prompt or echoing instructions — and reads out whatever variable name was
captured. The paper already calls tag-based termination "brittle." *(Paper
Appendix B; brief §8.4 / FM#3.)*

**R3 — Three termination models in flight.** Skeleton/mapping: `FINAL_VAR(name)`
tag. Skeleton also: a `set_final` primitive. Rewritten brief: the reserved
`answer` variable. All three are "output-in-a-variable" (paper §2 property 2), so
none is *inaccurate* in isolation — but shipping three invites a Phase 2 that
half-implements each, and the `set_final` ⇄ `FINAL_VAR` interaction is itself
unspecified (C11). Pick exactly one. *(Paper §2 property 2; brief §8.4.)*

**R4 — `_recursive_query` is not a recursive language model.** Issue #59 #2
reuses `rlm_query.py::_recursive_query` as the `sub_RLM` engine.
`_recursive_query` is a fixed chunk→select→recurse→aggregate summarizer with no
REPL and no code execution. The paper's `sub_RLM` (depth > 1) is itself an RLM —
it writes code and can spawn its own sub-calls. A depth-2 `sub_RLM` built on
`_recursive_query` satisfies "a nested call happened" but not "the nested call was
a recursive *language model*," weakening fidelity invariant §8.5 ("the root
actually recurses … else it is the old pipeline in a REPL") one level down.
Path B's `rlms.rlm_query` does not have this problem. *(Paper §2, Algorithm 1 /
programmatic recursion; brief §8.5.)*

**R5 — Hand-building Algorithm 1 maximizes the Algorithm-2 regression surface.**
The brief itself (§3) argues the faithful move is `from rlm import RLM`, "not our
re-derivation of the paradigm." A hand-written `root_loop.py` is judged against
the paper line by line; every one of brief §8's nine invariants becomes something
*our* code must not break. Path A is not paper-*inaccurate* by construction, but
it is the higher-variance choice for fidelity. *(Paper §2, Algorithm 1 vs
Algorithm 2; brief §3, §8.)*

**R6 — `rlms` is unverified (a Path-B delivery risk).** If Path B is chosen:
`rlms` is not installed, not in `backend/requirements*.txt`, and the brief §3
API claims (`custom_tools`, `max_depth`, `max_iterations`, `on_*` callbacks,
`other_backends`) cannot be checked against a paper dated May 2026. A Path-B plan
written now would be built on an unverified API. *Mitigation:* a spike task
(`pip install rlms`, introspect the real `RLM` signature and callbacks) must
precede any Path-B planning.

**R7 — Default root model is not paper-validated.** Correction #5 / brief §3: the
paper validates GPT-5 and Qwen3-Coder as RLM roots; Claude is only a baseline
coding agent. `_build_rlm_llm_client` (`orchestrator.py:532-540`) returns
`OpenAILlmClient` for the `openai` provider and `ClaudeLlmClient` otherwise — so a
default run uses **Claude as the RLM root**, unvalidated. Phase 2 must add the
`REPROLAB_RLM_ROOT_MODEL` knob defaulting to a validated root and emit a
`root_model_unvalidated` warning when Claude is used. *(Paper §3.2/§4.)*

**R8 — One system prompt is not safe across root models.** Correction #5 /
mapping §5: the paper's Qwen prompt needs an explicit anti-over-subcalling line.
`build_system_prompt` takes a `root_model` argument but the skeleton does not
populate per-model addenda. If Phase 2 ships one prompt and the root-model knob
swaps the model without the addendum, the root over-subcalls (cost) or
under-decomposes. *(Paper Appendix C.)*

**R9 — Only `Metadata(stdout)` may enter root history.** `ReplOutput` carries
full `stdout` (C10). If `RootLoop` ever appends the whole `ReplOutput` (or the
raw stdout) to history, that is FM#3 — raw `RUN` output into `hist` — and the
implementation has become Algorithm 2. The frozen `Metadata(stdout)` schema
(`{length, prefix≤200, has_traceback, var_assignments}`) is correct; the risk is
a careless append. *(Paper §2 — only metadata returns to the model; brief §8 /
FM#3.)*

---

## 6. Recommended next step — reconciliation

The architecture fork (R1 / D1) is an upstream decision. It belongs to
**armaanamatya** — the umbrella-issue owner, the author of the brief rewrite, and
the author of the skeleton — not to the Phase 2 assignee, who inherited the
contradiction and cannot resolve it unilaterally.

**Concrete escalation:** file a comment on umbrella issue #64 linking this
analysis (`docs/design/phase2-analysis.md`), state the fork in one line
(hand-build per #59 vs. `rlms` library per the brief), and ask the issue owner to
decide — and to realign whichever of issue #59 / the brief loses — before Phase 2
implementation begins.

Two coherent outcomes:

- **If hand-build wins:** amend brief §3/§5 (they currently forbid it); fix the
  mapping doc's stale section references (D4); keep the skeleton. Issue #59 stands
  as written. The constants (depth 2, 20 iters) already match.
- **If `rlms` wins:** re-scope issues #58/#59/#64 (3 of #59's 5 deliverables
  drop); delete the `repl_host.py` / `root_loop.py` / `sub_call.py` skeleton; add
  `rlms` to `backend/requirements.txt`; add a spike task to verify the API (R6);
  rewrite mapping doc §3/§4/§6. Phase 2 becomes "primitives + system prompt +
  `run.py` + SSE bridge."

Either way, also fix: the brief↔issue phase-numbering (D2), the `rlm_query.py`
disposition (D3), and `CLAUDE.md`'s Python version (D7) and stale file ref (D8).
And — independent of the fork — fix the `FINAL_VAR`/`FINAL` parser (R2) and pick a
single termination model (R3).

Once the fork is resolved, the task-by-task checkbox implementation plan
(deliverable #2 of the original task) can be written immediately — it is roughly
12–20 TDD tasks against whichever spec wins, and §4.2–§4.5 above already supply
the contracts, the hard-problem resolutions, the test surface, and the build
order. Per §4.6, Phase 2 can also begin now on the architecture-independent ~60%
(`primitives.py` registry + wrapper, `system_prompt.py`) without waiting for the
decision.
