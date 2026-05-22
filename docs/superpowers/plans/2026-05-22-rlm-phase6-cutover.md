# RLM Phase 6 — Cutover & Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retire the pre-pivot 14-stage `PipelineStage` state machine and its UI, leaving the `rlms`-based RLM orchestrator as the sole code path — backend, frontend, CLI, and docs.

**Architecture:** Incremental deletion. The old pipeline and the RLM path currently coexist; the RLM path imports a set of *shared* leaf modules but nothing from the old *orchestration* layer. Deletion proceeds importer-first (cut every consumer of a module before deleting the module) so the test suite stays green at every commit. One additive feature (live exploration-tree SSE events) and a docs rewrite finish the cutover.

**Tech Stack:** Python 3.14 / FastAPI / pytest (backend); Next.js 16 / TypeScript / vitest / Playwright (frontend); `rlms` 0.1.1.

---

## Branch, worktree & PR

- Create an isolated worktree branched off **`feat/rlm-phase4-frontend`'s head** (the RLM lab + Phase 4/5 work lives there; the current `feat/rlm-phase2-foundation` tree does **not** have it). Use the `superpowers:using-git-worktrees` skill.
- Branch name: **`feat/rlm-phase6-cleanup`**.
- One PR, base `main`. It **stacks on #70** — until #70 merges the PR diff also shows Phase 4 commits. Open it as **draft** until #70 lands, or hold until then, so reviewers see a clean diff.
- **No AI-attribution trailer** on commits or the PR body (session-3 handoff rule).

## Verification gate (run after every task)

```bash
# from repo root
.venv/bin/python -m pytest tests/ -q
cd frontend && npm run lint && npx tsc --noEmit && npm test && npx playwright test ; cd ..
```

All green is required to commit a task. **Known-acceptable exception:** the 2 pre-existing backend failures caused by `python: command not found` (the suite shells out to bare `python`). Anything else red blocks the commit — fix or revert before moving on.

**Per-task discipline (non-negotiable):** before deleting any file or symbol, `git grep -n "<symbol>"` across the *whole* repo (backend + `frontend/`) to confirm no live importer remains. This grep step *is* the re-verification of this plan against the phase-4 head — the codebase exploration behind this plan was partly done against an older branch, so trust the grep, not the prose, when they disagree.

---

# Part A — Backend deletion

The old orchestration layer is one mutually-referencing cluster: `orchestrator.py` imports every stage agent plus `structured_output.py` and `rubric_source.py`; `cli.py`, `live_runs.py`, `backend/agents/__init__.py`, and `evals/` import *into* that cluster. **Deletion must therefore be importer-first:** cut the entry points (CLI, live_runs, `__init__`, evals) → delete `orchestrator.py` + `pipeline.py` → delete the now-leaf stage agents. Deleting a leaf agent before `orchestrator.py` leaves `orchestrator.py` with a broken import; deleting `orchestrator.py` before its importers leaves them broken. Follow the task order.

---

## Task 1: Pre-flight gate — pin the RLM run artifact contract

**Files:**
- Read-only: `backend/agents/rlm/`, `backend/agents/run.py` (or wherever `write_final_report_rlm` lives), `backend/services/events/live_runs.py`, `backend/cli.py`
- Create: `tests/rlm/test_run_artifact_contract.py` (if no equivalent exists)

This gate replaces a paid real-paper run. It must prove `--mode rlm` is a *complete* replacement before any deletion.

- [ ] **Step 1: Confirm `tests/rlm/` is green**

Run: `.venv/bin/python -m pytest tests/rlm/ -q`
Expected: PASS (modulo the `python: command not found` exception).

- [ ] **Step 2: Establish the on-disk artifact contract**

`git grep -n` for each artifact filename and record which code writes it under an RLM run:
- `final_report.json` / `final_report.md` — confirm `write_final_report_rlm` (or its caller) writes both.
- `demo_status.json` — the UI status snapshot; confirm the RLM path writes it (the frontend SSE bridge and `live_runs.py` read it).
- `*.jsonl` event logs — confirm the RLM path writes the agent-event log that the SSE stream sources from. Part C (tree events) rides on this stream.
- `pipeline_state.json` — confirm whether the RLM path writes it. **Expected: it does NOT** (the RLM run produces `RLMRunResult`, not `PipelineState` — see `cli.py` `# RLMRunResult — no PipelineState fields`). Record the finding; it justifies the `PipelineState` deletion in Task 6.

- [ ] **Step 3: Write a test that pins the contract**

```python
# tests/rlm/test_run_artifact_contract.py
"""An RLM run must produce the artifacts the UI/SSE layer depends on."""
from pathlib import Path

def test_rlm_run_writes_required_artifacts(tmp_path, rlm_offline_run):
    # rlm_offline_run: existing fixture/helper that runs `--mode rlm` deterministically
    # against a fixture paper into `tmp_path`. Reuse the harness tests/rlm/ already uses;
    # do not invent a new one.
    run_dir = rlm_offline_run(tmp_path)
    for name in ("final_report.json", "final_report.md", "demo_status.json"):
        assert (run_dir / name).is_file(), f"RLM run did not write {name}"
    assert list(run_dir.glob("*.jsonl")), "RLM run wrote no agent-event log"
```

If `tests/rlm/` already proves this end-to-end, skip the new file and instead add a comment in this plan's PR description pointing at the existing test. Do not duplicate coverage.

- [ ] **Step 4: Run it**

Run: `.venv/bin/python -m pytest tests/rlm/test_run_artifact_contract.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/rlm/test_run_artifact_contract.py
git commit -m "test(rlm): pin the RLM run on-disk artifact contract"
```

**Gate decision:** if Step 2 finds the RLM path does *not* write `demo_status.json` or the `*.jsonl` log, **stop** — the cutover is not safe, and the missing writer must be added (separate task, out of this plan's deletion scope) before continuing.

---

## Task 2: Collapse the CLI to RLM-only

**Files:**
- Modify: `backend/cli.py`

The CLI currently supports `--mode {offline,sdk,rlm}` and a `cmd_eval` / `eval` subparser. Both reach the old pipeline. Collapsing them removes `cli.py` as a consumer of `orchestrator.py` / old `pipeline.py` / old `evals/`.

- [ ] **Step 1: Inventory `cli.py`'s coupling to the old path**

Run: `git grep -n -E "PipelineState|run_pipeline_sdk|run_pipeline_offline|cmd_eval|from backend.evals" backend/cli.py`
Record every hit — these are what this task removes.

- [ ] **Step 2: Remove the `eval` subparser and `cmd_eval`**

Delete the `cmd_eval` function and its subparser registration. Delete `cli.py`'s `from backend.agents.orchestrator import PipelineState` and the `PipelineState.load_checkpoint(...)` call (the resume/eval path). `final_report` scoring stays available via `evals/paperbench/` (kept — see Task 5), not via `cmd_eval`.

- [ ] **Step 3: Collapse `--mode`**

Remove `offline` and `sdk` from the `--mode` choices and any branch that dispatches to `run_pipeline_sdk` / `run_pipeline_offline`. `reproduce` now always runs the RLM path. Keep the `--mode` flag accepting only `rlm` (or drop the flag entirely if nothing else reads it — `git grep -n "args.mode"` to decide).

- [ ] **Step 4: Run the verification gate**

Run the full gate (see top of plan). Expected: green. CLI tests that asserted `offline`/`sdk` behaviour will fail — update or delete them as part of this task (`git grep -ln "mode.*offline" tests/`).

- [ ] **Step 5: Commit**

```bash
git add backend/cli.py tests/
git commit -m "refactor(cli): collapse run modes to RLM-only; drop eval subcommand"
```

---

## Task 3: Relocate `run_pipeline_rlm` out of `pipeline.py`

**Files:**
- Create: `backend/agents/rlm/run.py` (or extend an existing `backend/agents/rlm/` module if `run_pipeline_rlm` has a natural home there — check first)
- Modify: `backend/agents/pipeline.py`, `backend/services/events/live_runs.py`, `backend/cli.py`

`pipeline.py` has a **module-level** `from backend.agents.improvement import ...`. While `run_pipeline_rlm` lives in `pipeline.py`, that import keeps the old `improvement.py` agent alive and `pipeline.py` undeletable. Move `run_pipeline_rlm` to the `rlm` package so `pipeline.py` becomes fully dead.

- [ ] **Step 1: Move the function**

Cut `run_pipeline_rlm` (and any private helpers used *only* by it) from `pipeline.py` into `backend/agents/rlm/run.py`. Do **not** carry over `pipeline.py`'s module-level imports of `improvement.py`, `orchestrator.py`, etc. — `run_pipeline_rlm` does not need them; if a name turns out to be needed, import it function-locally from a *kept* module.

- [ ] **Step 2: Rewire callers**

`git grep -n "run_pipeline_rlm"` → update every importer (`live_runs.py`, `cli.py`, tests) to `from backend.agents.rlm.run import run_pipeline_rlm`.

- [ ] **Step 3: Simplify `live_runs.py` dispatch**

In `live_runs.py`, remove the run-mode branch that selects between `run_pipeline_sdk` / `run_pipeline_offline` / `run_pipeline_rlm`; the RLM function is now the only target. `git grep -n -E "run_pipeline_(sdk|offline)" backend/services/events/live_runs.py` must return nothing after this step.

- [ ] **Step 4: Run the verification gate**

Expected: green.

- [ ] **Step 5: Commit**

```bash
git add backend/agents/rlm/run.py backend/agents/pipeline.py backend/services/events/live_runs.py backend/cli.py tests/
git commit -m "refactor(rlm): move run_pipeline_rlm into the rlm package; rlm-only dispatch"
```

---

## Task 4: Trim `backend/agents/__init__.py`

**Files:**
- Modify: `backend/agents/__init__.py`

- [ ] **Step 1: Find dead re-exports**

Run: `git grep -n -E "ReproLabOrchestrator|PipelineState|PipelineStage" backend/agents/__init__.py`

- [ ] **Step 2: Remove them**

Delete the `from backend.agents.orchestrator import ...` re-export line(s) and any `__all__` entries for `ReproLabOrchestrator`, `PipelineState`, `PipelineStage`. Then `git grep -n -E "from backend.agents import (ReproLabOrchestrator|PipelineState|PipelineStage)"` across the repo — every hit must already be dead (handled in Tasks 2–3, 5). If one is live, stop and resolve it before continuing.

- [ ] **Step 3: Run the verification gate**

Expected: green.

- [ ] **Step 4: Commit**

```bash
git add backend/agents/__init__.py
git commit -m "refactor(agents): drop dead orchestrator re-exports from package init"
```

---

## Task 5: Delete the PipelineState-coupled `evals/` parts

**Files:**
- Delete: `backend/evals/reproduction.py`, `backend/evals/innovation.py`
- Modify: `backend/evals/runner.py`, `backend/evals/__init__.py`
- **Keep:** `backend/evals/paperbench/` entirely (`leaf_scorer.py`, `score.py`, `bundle.py` — Phase 5 scoring)

- [ ] **Step 1: Confirm the split**

Run: `git grep -n -E "PipelineState|reproduction|innovation" backend/evals/`
The reproduction/innovation eval and `EvalRunner`'s reproduction/innovation methods are PipelineState-coupled and dead. `paperbench/` is not — confirm `paperbench/` has zero `PipelineState` references.

- [ ] **Step 2: Delete and trim**

Delete `evals/reproduction.py` and `evals/innovation.py`. In `evals/runner.py` remove the reproduction/innovation `EvalRunner` methods and the `from backend.agents.orchestrator import PipelineState` import. In `evals/__init__.py` remove re-exports of the deleted symbols. Keep whatever `paperbench/` needs.

- [ ] **Step 3: Run the verification gate**

Expected: green. Delete or update eval tests that exercised the removed paths (`git grep -ln "reproduction\|innovation" tests/`); keep paperbench scoring tests.

- [ ] **Step 4: Commit**

```bash
git add backend/evals/ tests/
git commit -m "refactor(evals): remove PipelineState-coupled reproduction/innovation eval; keep paperbench"
```

---

## Task 6: Delete `orchestrator.py` and `pipeline.py`

**Files:**
- Delete: `backend/agents/orchestrator.py`, `backend/agents/pipeline.py`

After Tasks 2–5, both files have **no live importers**. `PipelineState` requires no extraction — Task 1 confirmed the RLM run path uses `RLMRunResult`, never `PipelineState`, and never writes `pipeline_state.json`.

- [ ] **Step 1: Prove both are dead**

```bash
git grep -n -E "from backend.agents.orchestrator import|import orchestrator|from backend.agents.pipeline import|import pipeline" -- backend/ tests/ frontend/
```
Expected: no hits outside `orchestrator.py`/`pipeline.py` themselves. **If any hit remains, stop** — trace it and cut that importer first; do not delete with a live reference.

- [ ] **Step 2: Delete**

```bash
git rm backend/agents/orchestrator.py backend/agents/pipeline.py
```

- [ ] **Step 3: Run the verification gate**

Expected: green. Delete orchestrator/pipeline-specific test files now failing on `ImportError` (`tests/test_issue22_orchestrator.py` and any `tests/...pipeline...` that targeted the old machine). `git grep -ln "orchestrator\|run_pipeline_sdk" tests/` to find them.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor(agents): delete the retired 14-stage orchestrator and pipeline"
```

---

## Task 7: Delete the now-leaf stage agents, topology, and orphaned helpers

**Files:**
- Delete: `backend/agents/dependency_verifier.py`, `backend/agents/experiment_runner.py`, `backend/agents/improvement.py`, `backend/agents/report_generator.py`, `backend/agents/verification.py`, `backend/agents/rubric_source.py`, `backend/agents/structured_output.py`, `backend/agents/topology.py`
- Modify: `backend/app.py` (remove the `/pipeline/topology` endpoint)

With `orchestrator.py`/`pipeline.py` gone these are leaves. `topology.py` is now deletable because Task 12 deletes the `/demo` route — its only remaining consumer (per the cutover decision: delete `/demo` entirely).

- [ ] **Step 1: Prove each file is dead**

For every file above, run `git grep -n "<modulename>"` across `backend/` and `tests/`. Each must have no importer other than itself and the (already-deleted) old cluster. **Pay special attention to `improvement.py`** — confirm it is the *agent* module, distinct from `backend/agents/prompts/improvement.py` (the prompt string), which is KEPT and used by `rlm/primitives.py`. Their grep hits must not be conflated.

- [ ] **Step 2: Remove the topology endpoint**

In `backend/app.py` delete the `@app.get("/pipeline/topology")` handler and the `from backend.agents.topology import PipelineTopology, default_topology` import.

- [ ] **Step 3: Delete**

```bash
git rm backend/agents/dependency_verifier.py backend/agents/experiment_runner.py \
       backend/agents/improvement.py backend/agents/report_generator.py \
       backend/agents/verification.py backend/agents/rubric_source.py \
       backend/agents/structured_output.py backend/agents/topology.py
```

- [ ] **Step 4: Run the verification gate**

Expected: green. Delete now-broken tests for these modules.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor(agents): delete retired stage agents, topology, and orphaned helpers"
```

---

## Task 8: Delete dead prompt files — and explicitly DO NOT trim `registry.py`

**Files:**
- Delete: only prompt files in `backend/agents/prompts/` confirmed referenced by neither `registry.py` nor `rlm/primitives.py`
- **Do not modify:** `backend/agents/registry.py`

**Decision — `registry.py` is left intact this PR.** Rationale: `runtime/invoke.py` does `AGENT_REGISTRY[agent_id].to_runtime_spec(...)`, and `runtime/` is kept and reachable from the RLM path via `ctx.runtime`. Whether the RLM path actually performs a registry lookup or only imports implementation functions directly (`run_with_sdk`, `run_offline`) is not fully pinned. Either way the cost/benefit is one-sided: `registry.py` is small static data, stale entries are harmless, and a wrong trim is a `KeyError` at agent-invocation time that the test suite will not necessarily catch. Trimming it is **cosmetic with non-zero blast radius** — defer it to a follow-up that first proves which `agent_id`s the RLM path resolves.

- [ ] **Step 1: Classify each prompt file**

For every file in `backend/agents/prompts/`, run `git grep -n "<PROMPT_CONST>"`. A file is dead **only if** referenced by neither `registry.py` nor anything under `backend/agents/rlm/`. Known-live (do **not** delete): `prompts/rubric_verifier.py` (`RUBRIC_VERIFIER_PROMPT`, used by `rlm/primitives.py`), `prompts/improvement.py` (`IMPROVEMENT_ORCHESTRATOR_PROMPT`, used by `rlm/primitives.py`), and every prompt still imported by `registry.py`.

- [ ] **Step 2: Delete only the confirmed-dead prompt files**

`git rm` the files that passed Step 1's test. If a prompt is imported by `registry.py` but its agent is otherwise unused, leave it — `registry.py` is not being trimmed, so its imports must stay resolvable.

- [ ] **Step 3: Run the verification gate**

Expected: green.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor(prompts): delete prompt files with no remaining references"
```

---

# Part B — Frontend deletion

Delete the old 14-stage lab UI and the `/demo` guided-tour route; keep the RLM lab (`frontend/src/components/lab/rlm/`). Extraction-before-deletion for the two shared symbols.

---

## Task 9: Extract shared symbols out of doomed files

**Files:**
- Create: `frontend/src/lib/events/dashboard-event.ts`
- Modify: importers of `DashboardLiveEvent`

`agent-timeline-rail.tsx` is being deleted but exports the `DashboardLiveEvent` type, which **kept** code (`useRun` hook, `lab-shell.tsx`) imports. `NODE_W` from `node-card.tsx` is consumed only by `demo-overlay.tsx`, which Task 12 deletes — so `NODE_W` needs no extraction; verify that and move on.

- [ ] **Step 1: Move the type**

Create `frontend/src/lib/events/dashboard-event.ts` containing the `DashboardLiveEvent` type definition (copy it verbatim from `agent-timeline-rail.tsx`). Re-verify the exact shape on the phase-4 branch before copying — `git show feat/rlm-phase4-frontend:frontend/src/components/lab/agent-timeline-rail.tsx`.

- [ ] **Step 2: Rewire importers**

`git grep -n "DashboardLiveEvent"` → repoint every importer to `@/lib/events/dashboard-event`. Importers in files Task 11 deletes can be ignored (they vanish); importers in **kept** files (`use-run.ts`, `lab-shell.tsx`, anything under `components/lab/rlm/`) must be updated now.

- [ ] **Step 3: Confirm `NODE_W` needs no rescue**

`git grep -n "NODE_W"` — every importer must be a file slated for deletion (`demo-overlay.tsx`, old lab files). If a kept file imports it, extract it alongside `DashboardLiveEvent`; otherwise do nothing.

- [ ] **Step 4: Run the verification gate**

Expected: green.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/lib/events/dashboard-event.ts frontend/src/
git commit -m "refactor(frontend): extract DashboardLiveEvent type to a shared module"
```

---

## Task 10: Simplify `lab-shell.tsx` to render only the RLM lab

**Files:**
- Modify: `frontend/src/components/lab/lab-shell.tsx`

- [ ] **Step 1: Read the current file on the working branch and locate the branch**

Find the `runMode === "rlm"` guard and the `WorkflowView` function. `WorkflowView` is the old-lab renderer (`PanWrap`, `ResizableSplit`, `TelemetryStrip`, …); the RLM branch renders the RLM lab.

- [ ] **Step 2: Delete the old branch**

Remove the `WorkflowView` function entirely and the `runMode === "rlm"` conditional — `lab-shell` now *unconditionally* renders the RLM lab component for an active run. Remove the now-unused imports (`PipelineTopology`/`TopologyProvider`, the old lab components, `topology`-related props). Keep the `UploadView` path for the no-active-run state.

- [ ] **Step 3: Run the verification gate**

Expected: green. Update `lab-shell.test.tsx` to drop assertions about the old `WorkflowView`.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/lab/lab-shell.tsx frontend/src/components/lab/lab-shell.test.tsx
git commit -m "refactor(lab): render the RLM lab unconditionally; drop the 14-stage WorkflowView"
```

---

## Task 11: Delete the old lab components

**Files (delete each with its `.css` and `.test`/`.spec` siblings):**
`progress-strip`, `telemetry-strip`, `gate-chips`, `lab-canvas`, `node-card`, `node-config`, `pan-wrap`, `resizable-split`, `script-panel`, `agent-info-panel`, `agent-info-helpers`, `hermes-audit-panel`, `floating-agent-window`, `status`, `timeline-panel`, `failure-summary`, `agent-timeline-rail` — all under `frontend/src/components/lab/`.

- [ ] **Step 1: Prove no kept file imports any of them**

For each component, `git grep -n "<component-name>"` across `frontend/src`. Every importer must itself be in this delete list or already deleted. The only legitimate cross-references were `DashboardLiveEvent` (Task 9) and `NODE_W` (Task 9 verified) — both handled.

- [ ] **Step 2: Delete**

`git rm` each `.tsx`/`.ts` plus its `.css` and any `.test.tsx`/`.test.ts` sibling. **Do not delete** anything under `frontend/src/components/lab/rlm/`, nor the kept shells: `lab-sidebar`, `command-palette`, `shortcut-overlay`, `upload-view`, `icons`, `shared-helpers`.

- [ ] **Step 3: Run the verification gate**

Expected: green. Deleting `progress-strip` + `telemetry-strip` clears 2 of the 4 known eslint errors automatically.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor(lab): delete the retired 14-stage lab component tree"
```

---

## Task 12: Delete the `/demo` route and dead dashboard plumbing

**Files:**
- Delete: `frontend/src/app/demo/`, `frontend/src/components/demo/` (`demo-overlay.tsx` + `.css`), `frontend/src/app/api/pipeline/topology/route.ts`, `frontend/src/lib/pipeline/` topology files (`topology.ts`, `server-fetch.ts`, `topology-context.tsx`, `topology-helpers.ts`), `frontend/src/hooks/use-topology.ts`, `frontend/src/lib/events/contract.ts`, `frontend/src/lib/demo/pipeline-dashboard.ts`, `frontend/e2e/lab-smoke.spec.ts`, `frontend/e2e/lab-e2e-full.spec.ts`
- Modify: any kept file that imported the above

Per the cutover decision, **`/demo` is deleted entirely** — the guided-tour demo is removed from the product. That makes the whole topology client stack dead.

- [ ] **Step 1: Map the blast radius**

```bash
git grep -n -E "pipeline/topology|use-topology|topology-context|topology-helpers|pipeline-dashboard|events/contract|demo-overlay" -- frontend/
```
Record every importer. Files **in** the delete set are fine. For any **kept** file that imports a doomed module, that import must be removed (it will be dead UI — e.g. a topology provider wrapping the kept lab; remove the wrapper).

- [ ] **Step 2: Delete the `/demo` route and topology stack**

`git rm` the route (`app/demo/`), the API proxy (`app/api/pipeline/topology/route.ts`), `components/demo/`, the `lib/pipeline/` topology files, and `hooks/use-topology.ts`.

- [ ] **Step 3: Delete dead dashboard type plumbing**

`pipeline-dashboard.ts` (server-side 14-stage builder) and `contract.ts` (old event type defs) are dead once the old lab and `/demo` are gone — **but** verify each `lib/demo/` neighbour (`server-payload.ts`, `demo-run-types.ts`, `server-fs.ts`, `node-runner.ts`) is itself dead before deleting it. The RLM lab rides on the shared `dashboard_event` SSE frame (filtered by `isRlmEvent`), **not** on `contract.ts` types — confirm with `git grep -n "isRlmEvent"` that the RLM path's event typing is self-contained. Delete only the confirmed-dead files; if a `lib/demo/` file is still imported by a kept route, leave it and note it.

- [ ] **Step 4: Delete the old e2e specs**

`git rm frontend/e2e/lab-smoke.spec.ts frontend/e2e/lab-e2e-full.spec.ts`. Keep `frontend/e2e/rlm-lab.spec.ts`.

- [ ] **Step 5: Run the verification gate**

Expected: green.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor(frontend): delete the /demo route and the dead topology/dashboard stack"
```

---

## Task 13: Collapse `runMode` and fix the live eslint error

**Files:**
- Modify: `frontend/src/components/library/library-filters.tsx`; the `runMode`/`DemoRunMode` type definition site

- [ ] **Step 1: Collapse the `runMode` union**

`git grep -n -E "DemoRunMode|runMode"` → the union over `{offline, sdk, rlm}` is now RLM-only. Replace the union type with the single literal (or remove the discriminator entirely if nothing branches on it anymore) and simplify every consumer.

- [ ] **Step 2: Fix the `library-filters.tsx` eslint error**

This is the remaining live (non-lab) eslint error — Task 11 cleared the other two. Run `cd frontend && npm run lint` to see the exact rule and line, fix it directly in `library-filters.tsx` (this file is in the live `/library` route, not dead lab code).

- [ ] **Step 3: Run the verification gate**

Expected: green — `npm run lint` now fully clean (0 errors).

- [ ] **Step 4: Commit**

```bash
git add frontend/src/
git commit -m "refactor(frontend): collapse runMode to rlm-only; fix library-filters lint error"
```

---

# Part C — Live exploration-tree SSE events (additive)

## Task 14: Emit `candidate_proposed` / `candidate_outcome` / `rubric_score` events

**Files:**
- Modify: backend RLM path (per the handoff spec) and `dashboard_emitter.py`
- Reference: `docs/superpowers/specs/2026-05-21-rlm-phase4-backend-events-handoff.md`

The exploration tree currently renders only from fixtures. Emit these three SSE events from the backend during a real RLM run so the tree is live. **Honesty rule:** emit only real values produced by the run — never fabricate a node, score, or slot to fill the tree.

- [ ] **Step 1: Read the handoff spec**

Open `docs/superpowers/specs/2026-05-21-rlm-phase4-backend-events-handoff.md` and follow its event payload schemas exactly. Do not improvise field names — the frontend `isRlmEvent` filter and the RLM lab's tree builder expect the spec's shapes.

- [ ] **Step 2: Write failing tests**

Add tests under `tests/rlm/` asserting that an RLM run (offline/deterministic harness from Task 1) emits `candidate_proposed` when a candidate is proposed, `candidate_outcome` when it resolves, and `rubric_score` when the rubric verifier scores. Assert payloads match the handoff spec's schema.

- [ ] **Step 3: Run tests to confirm they fail**

Run: `.venv/bin/python -m pytest tests/rlm/ -k "candidate or rubric_score" -v`
Expected: FAIL (events not emitted yet).

- [ ] **Step 4: Emit the events**

Wire emission into the RLM primitives at the points the handoff spec identifies — candidate proposal, candidate outcome, rubric scoring — routing through the existing `dashboard_emitter.py` so they land on the shared `dashboard_event` SSE frame.

- [ ] **Step 5: Run tests to confirm they pass**

Run: `.venv/bin/python -m pytest tests/rlm/ -k "candidate or rubric_score" -v`
Expected: PASS.

- [ ] **Step 6: Run the full verification gate**

Expected: green.

- [ ] **Step 7: Commit**

```bash
git add backend/ tests/rlm/
git commit -m "feat(rlm): emit live candidate/rubric SSE events for the exploration tree"
```

---

# Part D — Docs

## Task 15: Rewrite the docs to make the RLM orchestrator the present tense

**Files:**
- Modify: `README.md`, `system_overview.md`, `CLAUDE.md`, `docs/design/rlm-pivot-brief.md`, `docs/frontend_integration.md`
- Move: stray root screenshots / PDFs → `docs/`

- [ ] **Step 1: `CLAUDE.md`**

Delete the "Architecture pivot in progress" banner. Delete the "14-stage pipeline state machine", "Three verification gates", and topology sections. Replace "Where to look first" entries (`orchestrator.py`, `pipeline.py`) with the RLM path (`backend/agents/rlm/`, `backend/agents/rlm/run.py`). Update CLI docs: `--mode` is RLM-only; `cmd_eval` is gone. Keep sandbox, demo-gate, and SSE-frame sections that still apply.

- [ ] **Step 2: `system_overview.md` and `README.md`**

Rewrite so the RLM orchestrator is described as the current architecture, not a future pivot. Remove 14-stage / 5-path / gate descriptions and the old lab UI.

- [ ] **Step 3: `docs/design/rlm-pivot-brief.md`**

Drop the "pivot in progress" framing — the pivot is done. Keep it as the architecture reference for the RLM design.

- [ ] **Step 4: `docs/frontend_integration.md`**

Refresh for the RLM lab: the `dashboard_event` SSE frame + `isRlmEvent` filter, the exploration tree, the kept component set. Remove old-lab integration notes.

- [ ] **Step 5: Move stray root assets**

`git mv` any screenshots / PDFs sitting in the repo root into `docs/`.

- [ ] **Step 6: Run the verification gate**

Expected: green (docs-only, but run it — Markdown-linked snippets occasionally break builds).

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "docs: retire the pivot framing; document the RLM orchestrator as current"
```

---

## Task 16: Final full-suite pass and PR

- [ ] **Step 1: Clean run of the full gate**

```bash
.venv/bin/python -m pytest tests/ -q
cd frontend && npm run lint && npx tsc --noEmit && npm test && npx playwright test ; cd ..
```
Expected: all green except the 2 known `python: command not found` backend failures. `npm run lint` must be 0 errors.

- [ ] **Step 2: Dead-reference sweep**

```bash
git grep -n -E "PipelineStage|ReproLabOrchestrator|run_pipeline_sdk|run_pipeline_offline|pipeline/topology|WorkflowView" -- backend/ frontend/src/ docs/
```
Expected: no hits in live code (docs may mention them only historically — acceptable). Any live hit is a missed deletion — fix before opening the PR.

- [ ] **Step 3: Open the PR**

Base `main`, branch `feat/rlm-phase6-cleanup`, **draft** until #70 merges. PR body: summarize Parts A–D, note the stack on #70, note `registry.py` was deliberately *not* trimmed (deferred — see Task 8). **No AI-attribution trailer.**

---

## Self-review notes (carried into execution)

- **`registry.py` is intentionally untouched** — not an omission. Trimming is a deferred follow-up (Task 8 rationale).
- **`PipelineState` is deleted, not extracted** — Task 1 proves the RLM path never uses it.
- **Deletion order is load-bearing** — Tasks 2–7 are importer-first; do not reorder. Each task's Step 1 grep is the safety check; a live hit means stop.
- **Open question resolved:** `/demo` is deleted entirely (Task 12), which is what makes `topology.py` deletable (Task 7).
- **Re-verification:** this plan was authored partly against an older branch. Every task greps before it deletes — when the grep contradicts this prose, trust the grep.
