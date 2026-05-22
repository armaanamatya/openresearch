# RLM Pivot Phase 6 — Cleanup & Cutover Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete the old 14-stage `PipelineStage` pipeline (backend + frontend),
make the RLM path the only path, emit the 3 missing exploration-tree SSE events,
and rewrite the architecture docs.

**Architecture:** Subtractive cutover. Delete top-down through the import graph
(entry points → orchestration → stage agents) so every task ends with a green
build. Distinguish dead-from-the-pivot code from shared infrastructure the RLM
path reuses, per the dead-vs-shared map in the design spec.

**Tech Stack:** Python 3.14 / FastAPI / pytest (backend); Next.js 16 / TypeScript
/ vitest / Playwright (frontend).

**Design spec:** `docs/superpowers/specs/2026-05-22-rlm-phase6-cleanup-design.md`.
**Tree-events contract:** `docs/superpowers/specs/2026-05-21-rlm-phase4-backend-events-handoff.md`.

**Worktree:** `/Volumes/CS_Stuff/openresearch/.claude/worktrees/feat+rlm-phase6-cleanup`,
branch `feat/rlm-phase6-cleanup`. Run all git ops with `git -C <absolute-worktree-path>`.

**Commit convention:** NO `Co-Authored-By` / AI-attribution trailer (maintainer
instruction). One commit per task; squash per-task fix commits.

**Deletion discipline (every deletion task):** before removing a file/symbol,
`grep -rn` for every reference across `backend/` + `frontend/src/` + `tests/`.
Delete, then fix all fallout in the same task. End with no dangling import and
no unreachable code. Re-run the full suite after every task.

---

## Task 1: Environment setup & green baseline

**Files:** none modified.

- [ ] **Step 1: Set up the Python environment**

The worktree has no `.venv`. Verify the main checkout's venv works from the
worktree cwd:

```bash
cd /Volumes/CS_Stuff/openresearch/.claude/worktrees/feat+rlm-phase6-cleanup
/Volumes/CS_Stuff/openresearch/.venv/bin/python -c "import backend.agents.rlm.run; print('ok')"
```

Expected: `ok`. If it fails, create a fresh venv:
`python3.14 -m venv .venv && .venv/bin/pip install -r backend/requirements.txt -r backend/requirements-dev.txt`.
Record the working python path as `$PY` for all later tasks.

- [ ] **Step 2: Set up the frontend environment**

```bash
cd /Volumes/CS_Stuff/openresearch/.claude/worktrees/feat+rlm-phase6-cleanup/frontend
npm ci
```

Expected: clean install.

- [ ] **Step 3: Establish the backend baseline**

Run: `$PY -m pytest tests/ -q` from the worktree root.
Expected: a large pass count with at most the 2 known pre-existing failures
(`test_issue17_runtime`, `test_issue26_experiment_runner` — local-process
backend, `python: command not found`). Record the exact pass/fail/skip counts.

- [ ] **Step 4: Establish the frontend baseline**

```bash
cd frontend
npm run lint   ; npx tsc --noEmit ; npm test
```
Expected: lint reports the 4 known pre-existing errors
(`progress-strip.tsx` ×2, `telemetry-strip.tsx` ×1, `library-filters.tsx` ×1);
tsc clean; vitest green. Record counts.

- [ ] **Step 5: Confirm the RLM path is a complete replacement**

Run: `$PY -m pytest tests/rlm/ -q`. Expected: all green.
Confirm `backend/agents/rlm/run.py` calls `write_final_report_rlm` (writes
`final_report.{json,md}`) — grep: `grep -n write_final_report_rlm backend/agents/rlm/run.py`.
This is the pre-flight gate: the RLM path is mechanism-complete. No commit
(no files changed) — record the baseline numbers in the task report.

---

## Task 2: Collapse the CLI and live_runs to rlm-only

**Files:**
- Modify: `backend/cli.py`
- Modify: `backend/services/events/live_runs.py`
- Test: `tests/test_cli_provider_args.py` and any CLI/live-runs test referencing
  `offline`/`sdk` modes or the `eval` command.

- [ ] **Step 1: Find every reference**

```bash
grep -rn "run_pipeline_sdk\|run_pipeline_offline\|cmd_eval\|\"eval\"\|'eval'\|--mode\|mode == \"sdk\"\|mode == \"offline\"" backend/cli.py backend/services/events/live_runs.py tests/
```

- [ ] **Step 2: Edit `backend/cli.py`**

In `cmd_reproduce` (around lines 540–595): delete the `offline` branch
(`run_pipeline_offline`, ~547–561) and the `sdk` branch (`run_pipeline_sdk`,
~579–626); keep only the `rlm` branch. Remove the `--mode` argument from the
`reproduce` subparser, or make it accept only `rlm` with `default="rlm"` for
back-compat. Delete `cmd_eval` (line 294) and the `eval` subparser
(`evaluate = sub.add_parser("eval", …)` … `evaluate.set_defaults`, lines
702–710). Remove now-unused imports (`PipelineState` at line 296, the
`backend.evals` imports inside `cmd_eval`).

- [ ] **Step 3: Edit `backend/services/events/live_runs.py`**

Line 1225: change `from backend.agents.pipeline import run_pipeline_offline,
run_pipeline_sdk, run_pipeline_rlm` to import only `run_pipeline_rlm`. In the
run-mode dispatch (~1355–1392): delete the `sdk` branch (`run_pipeline_sdk`,
~1355) and the `offline` branch (`run_pipeline_offline`, ~1389); keep only the
`rlm` branch. Simplify `_run_mode_label` (line 1558) and any `run_mode in
("sdk", "rlm")` conditionals (lines 1339–1340) to the rlm-only reality.

- [ ] **Step 4: Delete/adjust dead tests**

Delete tests that exercise offline/sdk CLI modes or the `eval` command. Adjust
`tests/test_cli_provider_args.py` if it asserts on `--mode` choices.

- [ ] **Step 5: Run the suite**

Run: `$PY -m pytest tests/ -q`. Expected: green (minus known pre-existing).
Run: `$PY -m backend.cli reproduce --help` — expected: no `--mode offline/sdk`.

- [ ] **Step 6: Commit**

```bash
git -C <worktree> add -A && git -C <worktree> commit -m "refactor(rlm): collapse CLI + live_runs run-mode dispatch to rlm-only"
```

---

## Task 3: Delete the PipelineState-coupled evals

**Files:**
- Delete: `backend/evals/reproduction.py`, `backend/evals/innovation.py`,
  `backend/evals/runner.py`, `backend/evals/paperbench/runner.py`
- Modify: `backend/evals/__init__.py`
- Delete: `tests/` files covering reproduction/innovation eval and the
  paperbench pipeline runner.

- [ ] **Step 1: Find every reference**

```bash
grep -rn "evals.runner\|evals.reproduction\|evals.innovation\|EvalRunner\|evaluate_reproduction\|evaluate_innovation\|paperbench.runner\|evals import" backend/ tests/ scripts/
```

- [ ] **Step 2: Verify the keep-set is untouched**

Confirm `backend/evals/paperbench/leaf_scorer.py`, `score.py`, `bundle.py`,
`submission.py`, `evals/store.py`, `evals/schemas.py`, `evals/elo.py`,
`evals/ab_testing.py`, `evals/sources.py` have NO importer being deleted in
this task (grep above). These stay.

- [ ] **Step 3: Delete the files**

Delete `evals/reproduction.py`, `evals/innovation.py`, `evals/runner.py`,
`evals/paperbench/runner.py`. If `evals/paperbench/runner.py` is the only
`run_pipeline_sdk` consumer left, this also clears that dependency before
Task 4.

- [ ] **Step 4: Fix `backend/evals/__init__.py`**

Remove exports of `EvalRunner` and any reproduction/innovation symbols; keep
`EvalStore` and the paperbench/scoring exports.

- [ ] **Step 5: Delete dead tests**

Delete test files importing the deleted modules (e.g. eval-runner tests,
paperbench-pipeline-runner tests). Keep `leaf_scorer`/`score` tests.

- [ ] **Step 6: Run the suite & commit**

Run: `$PY -m pytest tests/ -q`. Expected: green.
```bash
git -C <worktree> add -A && git -C <worktree> commit -m "refactor(rlm): delete PipelineState-coupled evals (keep leaf scorer + paperbench scoring)"
```

---

## Task 4: Reduce `pipeline.py` to the RLM shim

**Files:**
- Modify: `backend/agents/pipeline.py`
- Modify: `backend/agents/__init__.py` (if it re-exports old-pipeline symbols)
- Delete: `tests/test_issue29_e2e_pipeline.py` and other tests of
  `run_pipeline_offline`/`run_pipeline_sdk`.

- [ ] **Step 1: Find every reference**

```bash
grep -rn "run_pipeline_sdk\|run_pipeline_offline\|from backend.agents.pipeline" backend/ tests/ scripts/
```
Expected production callers of sdk/offline: none (Tasks 2–3 removed them).

- [ ] **Step 2: Edit `pipeline.py`**

Delete `run_pipeline_sdk` and `run_pipeline_offline` and their helpers
(`_write_workspace_claim_map`, `_truncate_excerpt`, `_enrich` if only used by
the offline path — grep to confirm). Keep `run_pipeline_rlm`. Remove the
top-level `from backend.agents.orchestrator import PipelineStage, PipelineState`
(line 28) and the `report_generator`/`dashboard_emitter` imports if now unused.

- [ ] **Step 3: Delete dead tests**

Delete `tests/test_issue29_e2e_pipeline.py` and any other test of the deleted
functions.

- [ ] **Step 4: Run the suite & commit**

Run: `$PY -m pytest tests/ -q`. Expected: green.
```bash
git -C <worktree> add -A && git -C <worktree> commit -m "refactor(rlm): reduce pipeline.py to the run_pipeline_rlm shim"
```

---

## Task 5: Delete `orchestrator.py` (the 14-stage state machine)

**Files:**
- Delete: `backend/agents/orchestrator.py`
- Modify: `backend/agents/__init__.py`, `backend/cli.py` (residual
  `PipelineState` import if any survived Task 2)
- Delete: orchestrator-coupled tests.

- [ ] **Step 1: Find every reference**

```bash
grep -rn "backend.agents.orchestrator\|ReproLabOrchestrator\|PipelineStage\|PipelineState" backend/ tests/ scripts/
```
Expected: only `backend/agents/__init__.py` (production) plus test files.

- [ ] **Step 2: Delete the file and fix `__init__.py`**

Delete `backend/agents/orchestrator.py`. In `backend/agents/__init__.py`
(line ~23) remove `ReproLabOrchestrator`, `PipelineState` from imports/exports.

- [ ] **Step 3: Delete orchestrator-coupled tests**

Delete: `test_issue22_orchestrator.py`, `test_pipeline_state_persistence.py`,
`test_gate2_partial_continues_to_improvements.py`,
`test_track4_environment_build_repair.py`, `test_agent_sdk_scaffolding.py`,
`test_agent_runtime_orchestrator.py`, `test_hermes_audit_orchestrator.py`,
`test_hermes_audit_adapter.py`, `test_rlm_orchestrator_wiring.py` (tests
RLM-era methods on the *old* `ReproLabOrchestrator` — a retired bridge).
For `test_rubric_verifier.py`: it has deferred imports of `PipelineState`/
`ReproLabOrchestrator` mixed with rubric-helper tests — grep it; delete the
orchestrator-coupled test functions, keep any pure rubric tests, or delete the
file if it is entirely orchestrator-coupled.

- [ ] **Step 4: Run the suite & commit**

Run: `$PY -m pytest tests/ -q`. Expected: green.
```bash
git -C <worktree> add -A && git -C <worktree> commit -m "refactor(rlm): delete the 14-stage orchestrator state machine"
```

---

## Task 6: Delete the dead stage agents, dead prompts, trim the registry

**Files:**
- Delete: `backend/agents/dependency_verifier.py`, `experiment_runner.py`,
  `improvement.py`, `verification.py`, `report_generator.py`,
  `rubric_source.py`, `structured_output.py`
- Delete: dead `backend/agents/prompts/` files (see Step 2)
- Modify: `backend/agents/registry.py`, `backend/agents/__init__.py`
- Delete: the stage-agent tests.

- [ ] **Step 1: Find every reference for each module**

```bash
for m in dependency_verifier experiment_runner improvement verification report_generator rubric_source structured_output; do echo "== $m =="; grep -rn "agents.$m\|agents import.*$m" backend/ tests/ scripts/; done
```
Expected: no `backend/agents/rlm/` importer for any of them.

- [ ] **Step 2: Identify dead prompt files**

KEEP `backend/agents/prompts/rubric_verifier.py` and `prompts/improvement.py`
(imported by `rlm/primitives.py`). Candidates to DELETE (verify each has no
`rlm/` importer): `prompts/paper_understanding.py`,
`prompts/environment_detective.py`, `prompts/baseline_implementation.py`,
`prompts/experiment_runner.py`, `prompts/verifiers.py`,
`prompts/artifact_discovery.py`, `prompts/reproduction_planner.py`,
`prompts/_sandbox_contract.py`. Grep before deleting each.

- [ ] **Step 3: Delete the files**

Delete the 7 stage-agent modules and the confirmed-dead prompt files.

- [ ] **Step 4: Trim `registry.py` and `agents/__init__.py`**

`registry.py` registers stage agents/prompts. Remove every registration whose
target was deleted. Keep `AGENT_REGISTRY` and entries the RLM path / runtime
need (`rubric_verifier`, `improvement`). Fix `agents/__init__.py` exports.
Run `$PY -c "import backend.agents.registry"` to confirm it imports clean.

- [ ] **Step 5: Delete stage-agent tests**

Delete: `test_dependency_verifier.py`, `test_issue26_experiment_runner.py`
(this also removes one of the 2 pre-existing baseline failures),
`test_issue28_improvement.py`, `test_issue27_verification.py`,
`test_report_generator.py`, `test_partial_final_report.py`. For
`test_issue23_paper_understanding.py` / `test_issue24_environment_detective.py`
/ `test_issue25_baseline.py` — these cover modules KEPT as shared
(`paper_understanding.py`, `environment_detective.py`,
`baseline_implementation.py`); KEEP the tests but verify they still pass (they
test the core functions the primitives reuse).

- [ ] **Step 6: Run the suite & commit**

Run: `$PY -m pytest tests/ -q`. Expected: green; baseline failures now 1
(only `test_issue17_runtime`).
```bash
git -C <worktree> add -A && git -C <worktree> commit -m "refactor(rlm): delete dead stage agents + prompts; trim the agent registry"
```

---

## Task 7: Delete `topology.py` and the `/pipeline/topology` endpoint

**Files:**
- Delete: `backend/agents/topology.py`
- Modify: `backend/app.py`
- Delete: `tests/test_pipeline_topology_api.py`

- [ ] **Step 1: Find every reference**

```bash
grep -rn "topology\|PipelineTopology\|default_topology\|pipeline/topology" backend/ frontend/src/ tests/
```

- [ ] **Step 2: Delete and fix**

Delete `backend/agents/topology.py`. In `backend/app.py` remove the
`from backend.agents.topology import …` import (line ~16) and the
`GET /pipeline/topology` route handler. If the frontend has a now-dead consumer
of that endpoint (`frontend/src/lib/pipeline/layout.ts` and its callers), note
it for Task 10 — do not edit frontend here.

- [ ] **Step 3: Delete the test & run the suite**

Delete `tests/test_pipeline_topology_api.py`.
Run: `$PY -m pytest tests/ -q`. Expected: green.

- [ ] **Step 4: Commit**

```bash
git -C <worktree> add -A && git -C <worktree> commit -m "refactor(rlm): remove the 14-stage pipeline topology endpoint"
```

---

## Task 8: Simplify `WorkflowView` — the RLM lab becomes unconditional

**Files:**
- Modify: `frontend/src/components/lab/lab-shell.tsx`
- Modify: `frontend/src/components/lab/lab-shell.test.tsx`

- [ ] **Step 1: Read the current branch**

`WorkflowView` (`lab-shell.tsx` ~514–614) branches at `if (run.runMode ===
"rlm") { … return <RlmLab …/>; }` then runs the 14-stage path. Read the whole
function and note every import used only by the old branch.

- [ ] **Step 2: Make `<RlmLab>` unconditional**

Delete the `if (run.runMode === "rlm")` guard and the entire old-pipeline
branch after it (the pipeline header, `ResizableSplit`, `PanWrap`,
`RightPanel`, `TelemetryStrip`, etc.). `WorkflowView` returns `<RlmLab …/>`
unconditionally. Remove now-unused imports at the top of `lab-shell.tsx`
(`AgentTimelineRail`, `NodeCard`, `GateChips`, `FloatingAgentWindow`,
`ResizableSplit`, `PanWrap`, `TelemetryStrip`, `progress-strip`,
`agent-info-panel`, `script-panel`, `status`, `node-config`, `failure-summary`,
the `useRouter` if unused, etc. — the lint warnings list these).

- [ ] **Step 3: Update `lab-shell.test.tsx`**

Delete the `runMode: "sdk"` 14-stage tests; keep the
`"renders RlmLab when runMode is rlm"` test (now the unconditional case — drop
the conditional framing). Keep tests for the kept shell pieces (upload view,
command palette).

- [ ] **Step 4: Verify & commit**

```bash
cd frontend && npx tsc --noEmit && npm test -- lab-shell && npm run lint
```
Expected: tsc clean; lab-shell tests green; lint no longer warns on the removed
imports. tsc/lint will still flag the not-yet-deleted dead files — that is
expected until Task 9.
```bash
git -C <worktree> add -A && git -C <worktree> commit -m "refactor(rlm-ui): WorkflowView renders the RLM lab unconditionally"
```

---

## Task 9: Delete the dead 14-stage lab components

**Files:**
- Delete (with `.css`/`.test` siblings): `progress-strip`, `telemetry-strip`,
  `gate-chips`, `lab-canvas`, `node-card`, `node-config`, `pan-wrap`,
  `resizable-split`, `script-panel`, `agent-info-panel`, `agent-info-helpers`,
  `hermes-audit-panel`, `floating-agent-window`, `status`, `timeline-panel`,
  `failure-summary` — all under `frontend/src/components/lab/`.
- Modify: `frontend/src/components/lab/agent-timeline-rail.tsx`,
  `frontend/src/hooks/use-run.ts`.

- [ ] **Step 1: Extract the `DashboardLiveEvent` type**

`agent-timeline-rail.tsx` exports the `DashboardLiveEvent` type, used by
`use-run.ts`. Move that type definition into `use-run.ts` (or a small
`frontend/src/lib/events/dashboard-live-event.ts`) and repoint `use-run.ts`'s
import. Then `agent-timeline-rail.tsx` (component) is fully dead.

- [ ] **Step 2: Grep each file before deleting**

```bash
cd frontend && for f in progress-strip telemetry-strip gate-chips lab-canvas node-card node-config pan-wrap resizable-split script-panel agent-info-panel agent-info-helpers hermes-audit-panel floating-agent-window status timeline-panel failure-summary agent-timeline-rail; do echo "== $f =="; grep -rn "lab/$f\b\|/$f'" src/ ; done
```
Expected: no importer outside the dead set itself. `NODE_W` from `node-card.tsx`
is used by `demo/demo-overlay.tsx` — inline the constant value into
`demo-overlay.tsx` before deleting `node-card.tsx`.

- [ ] **Step 3: Delete the files**

Delete all listed components and their `.css`/`.module.css`/`.test.tsx`/
`.test.ts` siblings.

- [ ] **Step 4: Verify & commit**

```bash
cd frontend && npx tsc --noEmit && npm run lint && npm test
```
Expected: tsc clean; lint clears the `progress-strip`/`telemetry-strip` errors
(2 of 4 gone); vitest green.
```bash
git -C <worktree> add -A && git -C <worktree> commit -m "refactor(rlm-ui): delete the dead 14-stage lab components"
```

---

## Task 10: Delete the old event contract & collapse `DemoRunMode`

**Files:**
- Delete: `frontend/src/lib/events/contract.ts`,
  `frontend/src/lib/demo/pipeline-dashboard.ts` (+ `.test.ts`)
- Modify: `frontend/src/lib/demo/demo-run-types.ts`, `node-runner.ts`,
  `server-fs.ts`, `server-payload.ts`, `frontend/src/app/api/demo/route.ts`,
  `frontend/src/app/api/demo/events/route.ts`, `frontend/src/hooks/use-run.ts`,
  `frontend/src/lib/pipeline/layout.ts` (if dead).

- [ ] **Step 1: Grep the consumers**

```bash
cd frontend && grep -rn "events/contract\|pipeline-dashboard\|DemoRunMode\|runMode\|\"run_state\"\|\"agent_log\"\|\"dashboard_event\"" src/
```

- [ ] **Step 2: Delete `contract.ts` and `pipeline-dashboard.ts`**

Delete both (and `pipeline-dashboard.test.ts`). `pipeline-dashboard.ts` is the
14-stage server-side dashboard builder; trace its exports
(`node-runner.ts`/`server-fs.ts`/`server-payload.ts`) and delete the dead
server-side dashboard code paths that fed the old UI.

- [ ] **Step 3: Collapse `DemoRunMode`**

In `demo-run-types.ts` the union is `"offline" | "sdk" | "rlm"`. Collapse to
just `"rlm"` (keep the named type for clarity). Update `node-runner.ts`,
`server-fs.ts`, `route.ts` — the Python process is always the rlm run now;
remove offline/sdk spawn branches. Keep the `dashboard_event` and `run_state`
SSE frames in `events/route.ts` and `use-run.ts` (shared transport — the RLM
lab rides on `dashboard_event`); remove `agent_log` handling only if confirmed
old-pipeline-only.

- [ ] **Step 4: Handle `lib/pipeline/layout.ts`**

If `layout.ts` (the 14-stage graph layout, flagged by the `PipelineEdge` lint
warning) has no remaining consumer after Task 7 + this task, delete it and its
test; otherwise leave it.

- [ ] **Step 5: Verify & commit**

```bash
cd frontend && npx tsc --noEmit && npm run lint && npm test
```
Expected: tsc clean; vitest green.
```bash
git -C <worktree> add -A && git -C <worktree> commit -m "refactor(rlm-ui): delete the old event contract; collapse runMode to rlm-only"
```

---

## Task 11: Delete old e2e specs & fix the `library-filters` lint error

**Files:**
- Delete: `frontend/e2e/lab-smoke.spec.ts`, `frontend/e2e/lab-e2e-full.spec.ts`
- Modify: `frontend/src/components/library/library-filters.tsx`

- [ ] **Step 1: Delete the old Playwright specs**

Delete `lab-smoke.spec.ts` and `lab-e2e-full.spec.ts` (both drive the 14-stage
`WorkflowView`). Keep `rlm-lab.spec.ts` and `lab-smoke-interactive.spec.ts`.

- [ ] **Step 2: Fix the `library-filters.tsx` lint error**

The error is `react-hooks/set-state-in-effect` at `library-filters.tsx:39` — a
`setState` called synchronously inside a `useEffect`. Fix it properly (derive
the value during render, or guard the effect) — `library-filters.tsx` is a live
`/library` route, not dead code.

- [ ] **Step 3: Verify & commit**

```bash
cd frontend && npm run lint && npx playwright test rlm-lab.spec.ts lab-smoke-interactive.spec.ts
```
Expected: `npm run lint` **fully clean** (0 errors, 0 warnings); the 2 kept e2e
specs green.
```bash
git -C <worktree> add -A && git -C <worktree> commit -m "refactor(rlm-ui): delete old e2e specs; fix library-filters lint error"
```

---

## Task 12: Tree events — schema, RunContext, and `sse_bridge` builders

**Files:**
- Modify: `backend/agents/schemas.py`, `backend/agents/rlm/context.py`,
  `backend/agents/rlm/sse_bridge.py`
- Test: `tests/rlm/test_sse_bridge.py`, `tests/rlm/test_context.py`

Implements §2–§4 of the tree-events handoff doc. Follow
`docs/superpowers/specs/2026-05-21-rlm-phase4-backend-events-handoff.md` — the
field names there are the wire contract (authoritative source
`frontend/src/lib/events/rlm-events.ts`).

- [ ] **Step 1: Write failing tests for the event builders**

In `tests/rlm/test_sse_bridge.py` add tests for three new builders —
`build_candidate_proposed_event`, `build_candidate_outcome_event`,
`build_rubric_score_event` — asserting the exact field shapes from handoff
§2.1/§2.2/§2.3 (e.g. `candidate_proposed` → `{event, timestamp, iteration,
round, parent_id?, candidate:{id,title,category,description,reasoning}}`).
Add a `test_context.py` test that `RunContext` has `current_iteration: int = 0`
and `propose_round: int = 0`.

- [ ] **Step 2: Run the tests, verify they fail**

Run: `$PY -m pytest tests/rlm/test_sse_bridge.py tests/rlm/test_context.py -q`.
Expected: FAIL (builders / fields undefined).

- [ ] **Step 3: Add the schema and context fields**

In `schemas.py`, add `title: str = ""` to `ImprovementHypothesis`. In
`context.py`, add `current_iteration: int = 0` and `propose_round: int = 0` to
`RunContext`.

- [ ] **Step 4: Implement the three event builders in `sse_bridge.py`**

Add `build_candidate_proposed_event`, `build_candidate_outcome_event`,
`build_rubric_score_event`, following the existing `build_run_complete_event`
pattern. For `rubric_score`, derive `areas[].status` from thresholds
(`score >= 0.7` → `pass`, `>= 0.4` → `partial`, else `fail`) — define the
thresholds as named constants. All builders produce plain dicts routed through
`make_emit`.

- [ ] **Step 5: Run the tests, verify they pass; commit**

Run: `$PY -m pytest tests/rlm/ -q`. Expected: green.
```bash
git -C <worktree> add -A && git -C <worktree> commit -m "feat(rlm): tree-event schemas, RunContext counters, sse_bridge builders"
```

---

## Task 13: Tree events — emission wiring

**Files:**
- Modify: `backend/agents/rlm/binding.py`, `backend/agents/rlm/primitives.py`,
  `backend/agents/rlm/run.py`, `backend/agents/rlm/system_prompt.py`,
  `backend/agents/rlm/__init__.py`, `frontend_integration.md`
- Test: `tests/rlm/test_binding.py`, `tests/rlm/test_propose_improvements.py`

Implements handoff §3 (`candidate_proposed`), §4 (`rubric_score`), §5 Option B
(`candidate_outcome` via a `record_candidate_outcome` primitive), §6 (`parent_id`).

- [ ] **Step 1: Write failing tests**

In `test_binding.py`: after a wrapped `propose_improvements` returns N
hypotheses, `wrap_primitive` emits N `candidate_proposed` events (mapping
`path_id→id`, `title→title`, `category→category`, `hypothesis→description`,
`rationale→reasoning`) and increments `ctx.propose_round`. After a successful
`verify_against_rubric`, it emits one `rubric_score` event; after a *failed*
(`success=False`) verification it emits none. Add a test that the new
`record_candidate_outcome` primitive emits a `candidate_outcome` event with the
given `candidate_id`/`outcome`/`parent_id`.

- [ ] **Step 2: Run the tests, verify they fail**

Run: `$PY -m pytest tests/rlm/test_binding.py tests/rlm/test_propose_improvements.py -q`.
Expected: FAIL.

- [ ] **Step 3: Wire iteration plumbing**

Per handoff §3: `ReproLabRLMLogger.log` (in `sse_bridge.py`/`run.py`) sets
`ctx.current_iteration` before each wrapped call fires. Plumb a locked `emit`
callable onto `RunContext` (or reuse `make_emit`) so `wrap_primitive` emits
through the single egress chokepoint, not `dashboard._emit`.

- [ ] **Step 4: Emit `candidate_proposed` and `rubric_score`**

In `binding.py`'s `wrap_primitive`: after a successful `propose_improvements`,
increment `ctx.propose_round` and emit one `candidate_proposed` per hypothesis.
After a successful `verify_against_rubric`, emit one `rubric_score`. Update the
`propose_improvements` LLM prompt (`prompts/improvement.py` or the inline prompt
in `primitives.py`) to supply the new `title` field.

- [ ] **Step 5: Add the `record_candidate_outcome` primitive (Option B)**

Add `record_candidate_outcome(candidate_id, outcome, parent_id=None)` to
`primitives.py` + `PRIMITIVE_REGISTRY` + `PRIMITIVE_DESCRIPTIONS` — a no-op
computation whose `wrap_primitive` wrapper emits a `candidate_outcome` event.
Add one line to `system_prompt.py` instructing the root to call it after
evaluating each candidate.

- [ ] **Step 6: Run tests, update the contract doc, commit**

Run: `$PY -m pytest tests/rlm/ -q`. Expected: green.
Add the 3 new rows to `frontend_integration.md`'s SSE event table.
```bash
git -C <worktree> add -A && git -C <worktree> commit -m "feat(rlm): emit candidate_proposed / candidate_outcome / rubric_score"
```

---

## Task 14: Rewrite the architecture docs

**Files:**
- Modify: `README.md`, `system_overview.md`, `CLAUDE.md`,
  `docs/design/rlm-pivot-brief.md`, `frontend_integration.md`
- Move: stray root screenshots/PDFs → `docs/`

- [ ] **Step 1: Find stray root files**

```bash
ls -1 *.png *.jpg *.jpeg *.pdf *.gif 2>/dev/null
```
Move any found into `docs/` (or `docs/assets/`); update references to them.

- [ ] **Step 2: Rewrite `CLAUDE.md`**

Remove the "⚠ Architecture pivot in progress" banner. Replace the "14-stage
pipeline state machine" / "Three verification gates" / `PipelineStage`
architecture sections with the RLM orchestrator description: `backend/agents/rlm/`
(`run.py` builds `rlm.RLM(...)`, `.completion()`, `write_final_report_rlm`), the
9 domain primitives, `system_prompt.py`, the `--mode rlm`-only CLI, the RLM lab.
Update "Common commands" (`--mode` no longer has `offline`/`sdk`; no `eval`
command). Update "Where to look first".

- [ ] **Step 3: Rewrite `README.md`**

Lead with what the system does (reproduces papers, RLM-based), then how to run
it (`docker compose up`, the CLI), then one architecture paragraph (the
`rlms`-library + domain-primitives design). Remove 14-stage framing.

- [ ] **Step 4: Rewrite `system_overview.md`**

Replace the 14-stage "how it fits together" with the RLM architecture: the root
model treats the paper as a REPL `context` variable, calls domain primitives,
the 8-event SSE model, the RLM lab UI.

- [ ] **Step 5: De-banner `rlm-pivot-brief.md`**

Remove the "pivot in progress" / "current (pre-pivot) code" framing from the
header and §1; it becomes the architecture reference doc, not a migration plan.
Do not rewrite its technical content.

- [ ] **Step 6: Commit**

```bash
git -C <worktree> add -A && git -C <worktree> commit -m "docs: rewrite README / system_overview / CLAUDE.md for the RLM architecture"
```

---

## Task 15: Final full-suite verification

**Files:** none modified (verification only; small fixes if anything is red).

- [ ] **Step 1: Backend suite**

Run: `$PY -m pytest tests/ -q`. Expected: green; the only acceptable
pre-existing failure is `test_issue17_runtime` (`test_issue26_experiment_runner`
was deleted in Task 6). If anything else is red, fix it.

- [ ] **Step 2: Frontend checks**

```bash
cd frontend && npm run lint && npx tsc --noEmit && npm test
```
Expected: lint **0 errors 0 warnings**; tsc clean; vitest green.

- [ ] **Step 3: Playwright e2e**

Run: `cd frontend && npx playwright test`. Expected: `rlm-lab.spec.ts` and
`lab-smoke-interactive.spec.ts` green; no old-pipeline specs remain.

- [ ] **Step 4: Dead-reference sweep**

```bash
grep -rn "PipelineStage\|ReproLabOrchestrator\|run_pipeline_sdk\|run_pipeline_offline\|progress-strip\|gate-chips\|events/contract" backend/ frontend/src/ tests/
```
Expected: no production hits (docs/historical specs may mention them — that is
fine).

- [ ] **Step 5: Commit any fixes**

If Steps 1–4 required fixes, commit them:
```bash
git -C <worktree> add -A && git -C <worktree> commit -m "test(rlm): final Phase 6 verification fixes"
```
Otherwise no commit.

---

## After the plan: open the PR

Open one PR with `superpowers:finishing-a-development-branch` — base `main`,
head `feat/rlm-phase6-cleanup`. The PR stacks on PR #70; note in the body that
#70 must merge first. **Do not merge** — the maintainer signs off. No
AI-attribution trailer in the PR body.
