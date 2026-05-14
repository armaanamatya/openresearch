# Lab pipeline smoke test — results

Driver: Playwright 1.60.0 (chromium-headless), Node 22.14, frontend on
`http://localhost:3001`, FastAPI backend on `http://127.0.0.1:8001`. Spec at
`frontend/e2e/lab-smoke.spec.ts`. Per user direction, the suite is centered on
the **PDF upload flow** ("the one that used to not work"); the fixture-run path
is exercised once for parity in the shared session.

A root-cause bug was found and fixed mid-validation (see "The fix" below). The
table records the final state **after** the fix.

## Summary

| Section | Status | Notes |
|---|---|---|
| Pre-flight | PASS | tsc clean, vitest 34/34, backend pytest 7/7 (+ 5 new persistence tests). |
| **B1. PDF upload accepted** | **PASS** | 12 nodes render, projectId issued, workflow header reflects uploaded source. |
| **B2. Workflow nodes animate** | **PASS** | src done, Reader running, gate-1 chip visible, "Live agents" rail populates within 60 s. |
| **B3. ScriptPanel** | **PASS** | Source PDF preview + Download visible, hrefs include projectId, **proxy actually serves the PDF bytes (Content-Type: application/pdf)**, benchmark card + final-report link present. |
| **B4. Per-node panels (early pipeline)** | **PASS** | Paper + Reader open with structural content; the other 10 nodes were `upcoming` (opacity 0.4) — intentionally non-clickable until the pipeline advances. |
| **B5. Backend logs advancing** | **PASS** | runner.stderr.log shows stage marker `Starting: paper_understood`, no fatal exits. |
| **B6. Backend + proxy SSE** | **PASS** | Backend emits run_state + dashboard_event/agent_log/heartbeat; proxy passes them through. |
| **B7. SLOW — counter advances past 1** | **PASS** (4.2 min, was a 14-min timeout pre-fix) | Run `prj_833c80c0e7f4e011` wrote `pipeline_state.json` at `paper_understood` — the first stage transition — so the UI counter advanced past 1/12. Pre-fix this file was not written until `gate_1_passed` and the test timed out. |
| A1. Fixture workflow renders | PASS | Same structural checks as B2, on the fixture path. |
| C. Navigation | PASS w/ doc note | Brand resets, Library/Hermes nav works. Return-to-Lab does NOT auto-restore from FastAPI — see "C caveat". |
| D. Per-node panels (fixture, fast) | PASS | Same structural pattern as B4. |
| E. SSE health (live fixture) | PASS | Same as B6 against fixture run. |
| **E2. Bridge enrichment, deterministic** | **PASS** | Against `prj_1621776362bfa518` the proxy emits `payload.summary.stage = "gate_1_passed"` — confirms the bridge enrichment path itself. |

## The fix

**Root cause:** stage progression and checkpoint persistence were decoupled.
13 step functions did `state.stage = X` in memory; only 4 (the gates +
`complete`) also called `save_checkpoint`. The Next.js bridge reads
`pipeline_state.json` to populate `payload.summary.stage`, so for the entire
pre-gate-1 span (paper_understood → plan_created, 10-15 min of real LLM work)
the file did not exist and the UI counter was stranded at `1/12`.

**Solution** (`backend/agents/orchestrator.py`, `backend/agents/pipeline.py`):

1. **`PipelineState.advance_stage(stage, runs_root)`** — the single sanctioned
   transition path. Sets the stage and persists the checkpoint as one atomic
   operation, so the two can never desync. All 13 call sites in both the SDK
   orchestrator and the offline pipeline now go through it (gate sites collapse
   their former two-line `stage = … ; save_checkpoint(…)` into one call).
2. **Atomic checkpoint write** — `save_checkpoint` now writes to a `.tmp`
   sibling and `Path.replace()`s it into place. With ~14× more frequent writes,
   a concurrent bridge read must never observe a truncated file; write-then-
   rename guarantees that. (This also removes the race the bridge's LRU
   "last-good cache" workaround in `server-payload.ts` exists to paper over.)
3. **Enforced invariant** — `tests/test_pipeline_state_persistence.py` parses
   both modules with `ast` and fails if any `.stage =` assignment appears
   outside `advance_stage` (the setter) and `load_checkpoint` (deserialization).
   The bug class cannot silently return: a bare `state.stage = X` breaks CI.
   Plus behavioral tests for persist / round-trip-resume / atomic-write.

This is a root-level fix, not a patch: the invariant ("a stage transition *is*
a checkpoint write") is now expressed in one method and machine-enforced,
rather than being 13 places a developer must remember to keep in sync.

## Repro

```bash
# Backend
cd /home/abheekp/openresearch
.venv/bin/uvicorn backend.app:create_app --factory --host 127.0.0.1 --port 8001 &

# Frontend (point proxy at the backend port)
cd frontend
PATH=/home/abheekp/.nvm/versions/node/v22.14.0/bin:$PATH \
  REPROLAB_BACKEND_URL=http://127.0.0.1:8001 npm run dev &

# Tests — fast suite (B1-B6 + A1/D/C/E + E2)
PATH=/home/abheekp/.nvm/versions/node/v22.14.0/bin:$PATH \
  npx playwright test e2e/lab-smoke.spec.ts --grep-invert "(A2|B7|D2)\\."

# Tests — slow PDF gate-1 progression
PATH=/home/abheekp/.nvm/versions/node/v22.14.0/bin:$PATH \
  npx playwright test e2e/lab-smoke.spec.ts --grep "B7\\."
```

`LAB_BASE_URL` and `LAB_BACKEND_URL` env vars override the Playwright defaults
if your dev stack runs on different ports.

## Caveats and findings

### (FIXED) Counter advance was gate-bounded, not stage-bounded

This was the root bug behind the original "the lab graph is stuck at 1/12"
report. Tracing the chain:

- `stateMapForRun` (repro-lab-client.tsx:439) derives node states from
  `run.payload.summary.stage`.
- `payload.summary.stage` is populated by the Next.js enrichment layer
  (`server-payload.ts:enrichRunStateWithPayload`), which reads
  `runs/<id>/pipeline_state.json`.
- `pipeline_state.json` *was* written by `PipelineState.save_checkpoint`, but
  the orchestrator only called it at gate boundaries (gate_1/2/3 + complete) —
  never at the 9 intermediate stages.

So between `ingested` and `gate_1_passed` the file did not exist, enrichment
returned `payload = null`, and the UI counter stayed at `1/12` for the 10-15
min of real LLM work that span takes. The B7 smoke test reproduced this exactly
(14-min timeout).

**Fixed** — see "The fix" above. `advance_stage` makes every transition persist
atomically, an `ast`-based test enforces the invariant, and B7 now passes in
4.2 min. `docs/lab-ui-pipeline-bridge.md` has been corrected to describe the
new (now accurate) every-stage write cadence.

### (UNRELATED, pre-existing) Stale test `test_pipeline_stages_are_ordered`

`tests/test_issue22_orchestrator.py::test_pipeline_stages_are_ordered` fails on
`main` independent of this work — it expects a `composition_tested` stage that
no longer exists in the `PipelineStage` enum (nor is referenced anywhere in
`orchestrator.py` / `pipeline.py`). Verified failing identically with this
branch's changes stashed. The test is stale and should be updated to match the
current enum; left untouched here as out of scope.

### Return-to-Lab does not restore in-flight run

The smoke prompt expects: *"Return to Lab. The previously-running run should
still be active (state restored from FastAPI)."*

`frontend/src/app/lab/page.tsx` always renders `<ReproLabClient />` with no
`initialRun`. `ReproLabClient`'s state is in-memory only — no localStorage, no
URL `?projectId=` reader, no server prefetch on mount. After Library/Hermes
nav, returning to `/lab` shows the upload screen. The run continues in the
FastAPI backend, but the UI cannot reattach without a code change.

Smallest restorative fix:
- Persist the active `projectId` to `localStorage` on `setRun(next)`
- On `ReproLabClient` mount, read it and `GET /api/demo?projectId=…` to rehydrate
- Clear it in `resetToUpload` and on terminal status

### Synthetic SSE frames (`id: synth-N`) are correctly rare

The proxy emits `id: synth-N` run_state frames only when `stableEnrichedHash`
changes between emits. Two reasons they don't fire often:

1. The proxy's `lastState` is captured from the last backend `run_state` (which
   itself fires only on `demo_status.json` changes — rare).
2. Subsequent enrichments use the same `lastState` and re-read disk; if the
   enriched payload hasn't changed, the hash is identical and emission is
   suppressed.

So synth-N frames are rarely visible in an 8 s SSE sample. This is the
designed behavior for deduplication — not a bug. The deterministic E2 test
confirms enrichment itself works correctly on a populated `pipeline_state.json`.
Note: now that `pipeline_state.json` is written on every stage transition (see
"The fix"), synth-N frames will appear more often during an active run — each
real stage change flips the hash and triggers one synthetic enriched frame.

### Upcoming nodes ignore clicks (UX detail)

`NodeCard` makes upcoming nodes pointer-events-default and the parent
`onSelect` is a no-op for upcoming state (repro-lab-client.tsx:862-866 and
975-977). For an early-pipeline run, only Paper (src, done) and Reader (read,
running) are clickable. This matches user expectation but means Section D /
B4 can only deeply test 2 nodes early in the run. The other 10 nodes' panels
become testable as the pipeline advances.

## Files added/changed by this validation pass

App fix:
- `backend/agents/orchestrator.py` — added `PipelineState.advance_stage`; made
  `save_checkpoint` write atomically (`.tmp` + `Path.replace`); routed all SDK
  orchestrator stage transitions through `advance_stage`.
- `backend/agents/pipeline.py` — routed all offline-pipeline stage transitions
  through `advance_stage`.
- `tests/test_pipeline_state_persistence.py` — new: `ast`-based invariant guard
  + behavioral persist / resume / atomic-write tests.
- `docs/lab-ui-pipeline-bridge.md` — corrected the write-cadence description.

Test harness:
- `frontend/playwright.config.ts` — Playwright runner config (chromium-headless, baseURL `:3001`)
- `frontend/e2e/lab-smoke.spec.ts` — comprehensive PDF-upload spec (B1-B7) plus shared fixture-run session (A1, D, C, E, E2)
- `frontend/package.json` — `@playwright/test` 1.60.0 devDependency
- `docs/lab-smoke-test-results.md` — this file
