# RLM Pivot Phase 6 — Cleanup, Docs & Cutover — Design

> Spec for the final phase of the RLM pivot (issue #63). Phase 6 deletes the old
> 14-stage `PipelineStage` pipeline — backend and frontend — makes the RLM path
> the only path, implements the 3 missing exploration-tree events, and rewrites
> the architecture docs. Canonical pivot plan: `docs/design/rlm-pivot-brief.md`.
> Written 2026-05-22.

## 1. Context

The repo is being re-architected from a fixed 14-stage `PipelineStage` state
machine to an RLM orchestrator (`rlms` library, arXiv 2512.24601). Phases 1–5
built the RLM path *alongside* the old pipeline:

- Phases 1–3 + Phase 5 backend are on `main`: `backend/agents/rlm/` (`run.py`,
  `system_prompt.py`, `sse_bridge.py`, `primitives.py`, `binding.py`,
  `checkpoint.py`, `report.py`, …), the `--mode rlm` CLI path, `run_pipeline_rlm`,
  and the vendored PaperBench bundles (`ftrl`, `mechanistic-understanding`,
  `sequential-neural-score-estimation`).
- Phase 4 (RLM lab frontend) is **PR #70** — open, mergeable, reviewed
  "READY TO MERGE", **not yet merged**.

Phase 6 is the cutover: the old pipeline is deleted, the RLM path becomes the
only path, and the docs are rewritten to describe the RLM architecture as the
present.

### Reconciliation note (verified 2026-05-22)

The Phase 6 kickoff brief stated "Phase 5 is on `main`" — true for the backend
*mechanism* (`run.py` writes `final_report.{json,md}` via `write_final_report_rlm`,
verified). Phase 5's *demo deliverable* (≥2 PaperBench papers with real scores on
disk) is **incomplete** — the session-3 handoff records "1 of ≥2" (SNSE scored
0.4042, honest `failed` verdict; `mechanistic-understanding` incomplete; `ftrl`
unrun). Per maintainer decision, Phase 6 proceeds now: the RLM path is a
*mechanism-complete* replacement, and the remaining scored-paper runs do not
require the old pipeline to be alive — they can happen post-cutover.

## 2. Decisions (maintainer-confirmed)

1. **Sequencing** — Phase 6 proceeds now. The RLM path is a mechanism-complete
   replacement; "≥2 scored papers" is a Phase 5 demo deliverable, not a
   code-architecture gate.
2. **Branch** — Phase 6 branches off `feat/rlm-phase4-frontend`'s head (so it
   stacks on PR #70). Worktree: `.claude/worktrees/feat+rlm-phase6-cleanup`,
   branch `feat/rlm-phase6-cleanup`. One PR; **do not merge** — maintainer signs
   off. PR #70 must merge before this PR.
3. **Commit trailers** — **no** AI-attribution / `Co-Authored-By` trailer on
   commits or the PR body (the more recent maintainer instruction in the Phase 5
   session-3 handoff overrides the kickoff brief).
4. **Run modes** — collapse `--mode {offline,sdk,rlm}` to **rlm-only**. Delete
   `run_pipeline_sdk` and `run_pipeline_offline`. The no-LLM `offline` mode is
   not preserved (pytest covers determinism; `stub_primitives.py` can be wired
   into a stub mode later if needed).
5. **Evals** — delete the `PipelineState`-coupled parts (`EvalRunner`
   reproduction/innovation eval, `evals/reproduction.py`, `evals/innovation.py`,
   the `cmd_eval` CLI command + `eval` subparser). **Keep** `evals/paperbench/`
   (`leaf_scorer.py`, `score.py`, `bundle.py`) — Phase 5 scoring.
6. **Tree events** — the 3 missing SSE events (`candidate_proposed`,
   `candidate_outcome`, `rubric_score`) **are in Phase 6 scope**; implement per
   `docs/superpowers/specs/2026-05-21-rlm-phase4-backend-events-handoff.md`.

## 3. The dead-vs-shared map

The hard part of Phase 6 is distinguishing dead-from-the-pivot code from shared
infrastructure the RLM path reuses. This map was built by grepping every
`import`/reference against the `feat/rlm-phase4-frontend` worktree.

### 3.1 Backend — DELETE (dead; only the old pipeline reaches them)

| Target | Evidence |
|---|---|
| `backend/agents/orchestrator.py` (whole file) | `ReproLabOrchestrator`, `PipelineStage`, `PipelineState`, Gate 1/2/3 control-flow. Not imported by any `backend/agents/rlm/` file. |
| `backend/agents/pipeline.py` → `run_pipeline_sdk`, `run_pipeline_offline` | Old-path entry points. Keep only `run_pipeline_rlm` (a thin shim to `rlm.run`). |
| `backend/agents/dependency_verifier.py` | Zero production importers. |
| `backend/agents/experiment_runner.py` | Only `orchestrator.py` + `pipeline.py` old path. |
| `backend/agents/improvement.py` | Only `pipeline.py` old path. |
| `backend/agents/report_generator.py` | Only `orchestrator.py` + `pipeline.py`. RLM builds its report in `rlm/report.py`. |
| `backend/agents/verification.py` | Only `pipeline.py` old path. RLM verification is the `verify_against_rubric` primitive. |
| `backend/agents/rubric_source.py` | Only `orchestrator.py`. |
| `backend/agents/structured_output.py` | Only `orchestrator.py`. |
| `backend/agents/topology.py` + `GET /pipeline/topology` in `app.py` | The hardcoded 14-stage / 5-path graph. Only `app.py` imports `topology.py`. |
| Dead `backend/agents/prompts/` files | Stage-agent prompts not used by primitives (`paper_understanding`, `environment_detective`, `baseline_implementation`, `experiment_runner`, `verifiers`, `artifact_discovery`, `reproduction_planner`, `_sandbox_contract`). Reached only via `registry.py` → old pipeline. |
| `backend/evals/` PipelineState-coupled parts | `EvalRunner` reproduction/innovation eval, `evals/reproduction.py`, `evals/innovation.py`, `cmd_eval` + `eval` subparser in `cli.py`. |

### 3.2 Backend — KEEP (shared; the RLM path imports these)

`baseline_implementation.py`, `environment_detective.py`, `paper_understanding.py`
(core functions imported by `rlm/primitives.py`); `execution.py`, `schemas.py`,
`dashboard_emitter.py`, `telemetry.py`, `registry.py` (trim dead registrations);
`resilience/`, `runtime/` (all); `prompts/rubric_verifier.py`,
`prompts/improvement.py`; all of `backend/agents/rlm/`; `evals/paperbench/`
(`leaf_scorer.py`, `score.py`, `bundle.py`, `store.py`, `schemas.py`, `sources.py`).

Shared infra — KEEP: SSE transport (`services/events/live_runs.py`), the sandbox
layer (`services/runtime/`), the cost ledger (`resilience/cost.py`), the SQLite
event store (`eventstore/`), `messaging/`, and the **Hermes audit chain**
(`hermes_audit/` — confirmed still used by `dashboard_emitter.py`, `live_runs.py`,
and `services/ingestion/parser/vision.py`, all shared; **not** orphaned).

### 3.3 Frontend — DELETE (dead 14-stage lab)

Components (with their `.css`/`.test` siblings): `progress-strip`,
`telemetry-strip`, `gate-chips`, `lab-canvas`, `node-card`, `node-config`,
`pan-wrap`, `resizable-split`, `script-panel`, `agent-info-panel`,
`agent-info-helpers`, `hermes-audit-panel`, `floating-agent-window`, `status`,
`timeline-panel`, `failure-summary`.

Event layer: `frontend/src/lib/events/contract.ts` (old 14-stage event type
definitions — its own header says "Removal of contract.ts is Phase 6"); the
14-stage server-side dashboard builder `pipeline-dashboard.ts` and the dead
server-side dashboard code it feeds.

e2e: `frontend/e2e/lab-smoke.spec.ts`, `frontend/e2e/lab-e2e-full.spec.ts`.

### 3.4 Frontend — KEEP (shell + RLM lab + upload flow)

`lab-shell.tsx` (**simplified** — see §4.2), `lab-sidebar`, `command-palette`,
`shortcut-overlay`, `upload-view`, `icons`, `shared-helpers`, `useRun`
(`hooks/use-run.ts` — still the upstream SSE feed for the RLM lab); all of
`frontend/src/components/lab/rlm/`, `hooks/use-rlm-run.ts`,
`lib/events/rlm-events.ts`, `e2e/rlm-lab.spec.ts`,
`e2e/lab-smoke-interactive.spec.ts`.

`agent-timeline-rail.tsx` is MIXED — the `DashboardLiveEvent` type it exports is
used by `use-run.ts` (KEEP); the `AgentTimelineRail` *component* is dead. Extract
the type to a kept module (or inline it into `use-run.ts`), delete the component.

### 3.5 SSE frame names — a correction to issue #63's wording

Issue #63 says "delete the old SSE event types `run_state`, `agent_log`,
`dashboard_event`." This is imprecise. `run_state` and `dashboard_event` are
**shared SSE transport frames** — the RLM lab rides on `dashboard_event`
(filtered by `isRlmEvent`) and consumes run metadata from `run_state`. What is
actually dead is the `contract.ts` *type definitions* and `pipeline-dashboard.ts`,
**not** those frame names. The `agent_log` frame handling may be removed if it is
old-pipeline-only after the cutover. **Keep `run_state` and `dashboard_event`.**

## 4. Work breakdown

### 4.0 Pre-flight gate

Before any backend deletion, confirm `--mode rlm` is a complete replacement —
**not** by a paid real-paper run (a maintainer-cost decision), but by: `pytest
tests/rlm/` green, and a verified code-path that `run.py` → `write_final_report_rlm`
writes `final_report.{json,md}`. Establish the green baseline (backend pytest,
frontend lint/tsc/vitest) and record it.

### 4.1 Backend deletion (incremental)

Delete §3.1 targets. After each module deletion, grep for every remaining
reference and fix the fallout — no dangling imports, no dead references. Trim
`registry.py` to only the registrations the RLM path needs. Collapse `cli.py`'s
`--mode` to rlm-only and delete the `eval` subparser. Simplify
`live_runs.py`'s run-mode dispatch to rlm-only. Reduce `pipeline.py` to the
`run_pipeline_rlm` shim (or delete it and repoint callers at `rlm.run`).

### 4.2 Frontend deletion (incremental)

Delete §3.3 targets. In `lab-shell.tsx`, `WorkflowView` loses its
`if (run.runMode === "rlm")` guard and unconditionally renders `<RlmLab>`; the
entire old-pipeline branch is deleted. Collapse the `DemoRunMode` union
(`"offline" | "sdk" | "rlm"`) to rlm-only across `demo-run-types.ts`,
`pipeline-dashboard.ts` successors, `node-runner.ts`, `server-fs.ts`,
`route.ts`. Fix the real `library-filters.tsx` eslint error (a live route — not
dead). Result: a fully clean `npm run lint` (the `progress-strip`/`telemetry-strip`
errors vanish with the files).

### 4.3 Tree events (additive)

Implement `candidate_proposed`, `candidate_outcome`, `rubric_score` SSE emission
per `2026-05-21-rlm-phase4-backend-events-handoff.md`: add `title` to
`ImprovementHypothesis`, `current_iteration`/`propose_round` to `RunContext`,
event builders in `sse_bridge.py`, emission in `binding.py`'s `wrap_primitive`,
and `candidate_outcome` via the handoff's recommended **Option B** (a
`record_candidate_outcome` primitive — authoritative, no run-level inference).
Route every new event through `make_emit` (the single egress chokepoint).
Honesty rule: emit only real values, no fabricated slots.

### 4.4 Docs

Rewrite `README.md` (lead with what the system does → how to run it → one
architecture paragraph), `system_overview.md` (RLM architecture), and `CLAUDE.md`
(remove the "⚠ Architecture pivot in progress" banner; replace the 14-stage
sections with the RLM orchestrator description as the present; update the
`--mode` collapse). `rlm-pivot-brief.md` loses its "pivot in progress" framing
and becomes the architecture doc. Update `frontend_integration.md`'s SSE event
table with the 3 new rows. Move stray root screenshots/PDFs into `docs/`.

## 5. Ordering & execution

Executed via `superpowers:subagent-driven-development` — fresh subagent per task,
spec + quality review per task, squashed to one commit per task. Order:

1. Pre-flight gate + baseline (§4.0).
2. Backend deletion, incremental (§4.1) — one module group per task, full suite
   after each.
3. Frontend deletion, incremental (§4.2).
4. Tree events (§4.3).
5. Docs (§4.4).
6. Final full-suite verification pass.

Deletion discipline: before removing any file/symbol, grep for *every*
reference; delete, then fix all fallout in the same task. Never leave a dangling
import or unreachable code.

## 6. Verification

After every task: `.venv/bin/python -m pytest tests/`, `npm run lint`,
`npx tsc --noEmit`, `npm test` (vitest), and the Playwright e2e — all green. The
2 pre-existing backend failures (`test_issue17_runtime`,
`test_issue26_experiment_runner` — local-process backend, `python: command not
found`) are unrelated; note that `test_issue26_experiment_runner` is **deleted**
with `experiment_runner.py`, so it leaves the suite entirely. Old-pipeline test
files (the `tests/test_issue2x_*` stage-agent tests, `test_issue22_orchestrator`,
`test_issue29_e2e_pipeline`, `test_pipeline_state_persistence`, the gate tests,
the old-lab vitest files) are deleted alongside the code they cover.

The RLM path must stay green throughout: `tests/rlm/`, `use-rlm-run.test.ts`,
`rlm-events.test.ts`, the `lab/rlm/` vitest suite, and `e2e/rlm-lab.spec.ts`.

## 7. Done condition

- The old 14-stage pipeline is gone — backend and frontend — with no dangling
  references, dead imports, or unreachable code.
- The lab is unconditionally the RLM lab; `WorkflowView` no longer branches on a
  removed `runMode`.
- `README.md`, `system_overview.md`, `CLAUDE.md` describe the RLM-only
  architecture; the "pivot in progress" banner is gone.
- The 3 exploration-tree events are emitted by the backend; the tree is live on
  real runs, not fixture-only.
- `npm run lint` fully clean; `npx tsc --noEmit` clean; `npm test` green; the
  Playwright e2e green; `pytest tests/` green (modulo the pre-existing
  unrelated failure).
- One Phase 6 PR, opened off a branch that includes Phase 4, not merged.

## 8. Out of scope

- Merging PR #70 (maintainer).
- Running real PaperBench papers to complete Phase 5's "≥2 scored papers"
  deliverable (post-cutover, maintainer-cost decision).
- Wiring `stub_primitives.py` into a no-LLM run mode (a possible future task).
- Refactoring `EvalRunner` to consume `final_report.json` (evals' PipelineState-
  coupled parts are deleted, not ported).
- The `environment="local"` host-RCE threat model (a Phase 3/5 gate, tracked
  separately).
