# RLM Lab Go-Live — Design Spec

> **Status:** approved design (brainstormed 2026-05-22).
> **Scope:** make the RLM lab *the* lab — driven by real `--mode rlm` runs, end to end.
> A frontend RLM-only cutover + the backend emission of three SSE events.
> **Canonical inputs:** `docs/design/phase4-6-execution-prompt.md`,
> `docs/design/rlm-pivot-brief.md`, the Phase 4 design spec
> (`docs/superpowers/specs/2026-05-21-rlm-phase4-frontend-design.md`), the
> backend-events handoff (`docs/superpowers/specs/2026-05-21-rlm-phase4-backend-events-handoff.md`),
> `frontend_integration.md`, and **GitHub issue #63** (RLM Pivot Phase 6 — this cutover
> is the *frontend half* of that issue, pulled forward; see §9).
> **Next step:** `superpowers:writing-plans` → the implementation plan.

---

## 1. Goal & done condition

Phase 4 built the RLM lab **fixture-first**: it renders an `rlm` run's event stream,
but it is currently *demonstrated* against a recorded fixture (`?rlmFixture=1`), and a
real run reaches it only via the CLI. This work makes the RLM lab **the lab**: every
run started in the UI is an `rlm` run, the lab always renders `<RlmLab>`, and it is
driven entirely by a real run's **live SSE event stream**. The old 14-stage pipeline
UI is removed from the frontend.

**Done when:**

1. Starting a run in the lab UI starts an `rlm` run — there is no `sdk`/`offline`
   choice anywhere in the UI.
2. The lab always renders `<RlmLab>`; the old 14-stage `WorkflowView` path and its
   components (`progress-strip`, `gate-chips`, `LabCanvas`, …) are deleted from the
   frontend.
3. A real `--mode rlm` run renders live through every region — including a
   **populating exploration tree** — because the backend now emits
   `candidate_proposed` / `candidate_outcome` / `rubric_score`.
4. No static or fabricated display values; the honesty rule (Phase 4 spec §14) holds —
   the UI shows only what a real run produced.
5. `npm run lint`, `npx tsc --noEmit`, `npm test` (vitest), the Playwright e2e, and
   `pytest tests/` are all green; a real `--mode rlm` smoke run renders correctly
   end to end.

---

## 2. Scope

**In scope:**

- **Go-live wiring** — unblock `mode=rlm` through the `/api/demo` proxy and `useRun`;
  remove the 200-event cap that truncates a long `rlm` run's history.
- **The RLM-only frontend cutover** — delete the old 14-stage lab UI; `lab-shell.tsx`
  always renders `<RlmLab>`; the `runMode` union collapses to `rlm`.
- **The 3 backend events** — `candidate_proposed` / `candidate_outcome` /
  `rubric_score`, per the committed handoff doc — **cofounder-coordinated** (§4).
- **Static-value cleanup** — the hardcoded `0.35` threshold and `FIXTURE_RUN_META`.

**Out of scope — stated decisions:**

- **Full Phase 5** — real PaperBench paper reproductions and the I7 score-cap unblock
  (real metric extraction). The lab will *honestly* render whatever a real `rlm` run
  produces today — capped score and all. This plan fixes the *plumbing*, not run quality.
- **The old *backend* 14-stage pipeline** — `PipelineStage`, the stage agents, the
  `sdk`/`offline` orchestration in `pipeline.py` / `orchestrator.py` — is left in place
  but **unreachable from the UI**. Deleting it — together with the Gate 1/2/3
  control-flow, the hardcoded improvement paths, and the old `run_state` / `agent_log`
  SSE event types — is the *rest* of Phase 6 (GitHub issue #63), a separate effort. CLI
  `--mode sdk/offline` therefore still works — nothing is stranded and `main` stays
  runnable.
- **The Phase-6 doc rewrites** — `README`, `system_overview.md`, `CLAUDE.md` (issue
  #63 items 1–3) — are not in this plan; they belong with the backend cleanup above.
- **The tree-framing fix** — the exploration canvas opening panned to dead space — is a
  small Phase-4 polish. It is landed directly on the still-open **PR #70** as a separate
  follow-up commit, not part of this plan.

---

## 3. Architecture & data flow

One path, no fixture in it:

```
UI "start run"
  → POST /api/demo  (mode=rlm)
  → backend spawns the rlm run subprocess → run_pipeline_rlm
  → runs/<id>/dashboard_events.jsonl                 (append-only event log)
  → GET /runs/<id>/events  (SSE dashboard_event frames)
  → /api/demo/events  (Next.js server-side proxy)
  → useRun() → dashboardEvents[]
  → useRlmRun(events) → RlmRunState
  → <RlmLab>
```

`useRun` is reused unchanged as the SSE transport. The `?rlmFixture=1` replay path
stays — but only as a **dev/test affordance** (it backs the Playwright e2e and local
development); it is no longer how the lab is "seen."

The transport and the 8-event model were verified against backend source during
brainstorming (the SSE path `dashboard_events.jsonl → live_runs.py → /api/demo/events
→ useRun` carries RLM events intact; an `rlm` run produces `dashboard_events.jsonl`).

---

## 4. Backend changes — the three events  ·  COFOUNDER-COORDINATED

> **This section requires the cofounder.** It touches Phase 3 files (`run.py`,
> `sse_bridge.py`, `system_prompt.py`, `context.py`, `primitives.py`, `schemas.py`)
> which the kickoff brief assigns to the cofounder ("coordinate, do not silently
> write it").
>
> **The single hardest coordination ask:** `candidate_outcome` is emitted via
> **Option B** — a trivial new `record_candidate_outcome(candidate_id, outcome,
> parent_id)` primitive that the **root model calls** after evaluating a candidate.
> That requires the cofounder to (a) add the primitive to `PRIMITIVE_REGISTRY` and
> (b) **change the root system prompt** to instruct the model to call it. Surface
> this one ask to the cofounder before the backend tasks begin — it is the plan's
> critical-path dependency.

The full, authoritative contract is the committed handoff doc
`docs/superpowers/specs/2026-05-21-rlm-phase4-backend-events-handoff.md`. The changes:

- **`RunContext`** gains `current_iteration` and `propose_round` counters; **`make_emit`
  is plumbed onto it** — today `wrap_primitive` only has `ctx.dashboard.primitive_call`,
  which bypasses the `make_emit` egress lock.
- **`ImprovementHypothesis`** gains a `title: str` field; the `propose_improvements`
  prompt is updated to supply it.
- **`binding.py wrap_primitive`** — after a successful `propose_improvements`, emit one
  `candidate_proposed` per returned hypothesis; after a successful
  `verify_against_rubric`, emit one `rubric_score` (deriving each area's `status` —
  `pass`/`partial`/`fail` — from score thresholds; do **not** emit on a failed
  verification).
- **`candidate_outcome`** — Option B (above): the new `record_candidate_outcome`
  primitive emits it directly from the wrapper; the outcome is the root's
  *authoritative* decision (not backend inference) and `parent_id` comes for free.
- **Event builders** for all three in `sse_bridge.py`, following the existing
  `build_run_complete_event` pattern; route every one through `make_emit`.
- Update `frontend_integration.md`'s SSE event table with the three new rows.

Emitting these is a **pure transport addition** — it does not change the existing 5
events, the reducer, or any frontend file. Once they flow, the Phase-4 UI's tree goes
fully live with no further frontend change.

---

## 5. Frontend changes — the RLM-only cutover

### 5.1 Delete the old lab UI

Remove the 14-stage rendering path and the components used **only** by it:
`progress-strip.tsx`, `gate-chips.tsx`, the `LabCanvas` + `node-card` + the
old-canvas `pan-wrap` usage, `telemetry-strip.tsx`, `agent-timeline-rail`,
`floating-agent-window`, `agent-info-panel`, `script-panel`, `node-config`, and the
`WorkflowView` 14-stage body + `RunOverview` + `RightPanel` inside `lab-shell.tsx`.
The topology machinery (`use-topology`, `topology-context`, `pipeline/layout`,
`pipeline/topology`) is removed **iff** it is used only by the old UI.

`lab-shell.tsx`'s `WorkflowView` collapses to: render `<RlmLab>`. The exact,
grep-verified delete list is produced by the `writing-plans` step — deletion is
reference-checked (grep every symbol), fallout-fixed, and the full suite re-run
**per task**.

### 5.2 The UI starts only `rlm` runs

`frontend/src/app/api/demo/route.ts` `toRunMode()` returns `"rlm"` (the only mode);
`backendQuery()` forwards it; `useRun`'s `startUploadedRun` / `startArxivRun` /
`startFixtureRun` send `mode=rlm`. The `runMode` union in `demo-run-types.ts` and
`pipeline-dashboard.ts` collapses to `"rlm"`. The model picker in the upload view
stays (model choice is orthogonal); **no** mode picker is added.

### 5.3 The event cap

`frontend/src/hooks/use-run.ts` `MAX_DASHBOARD_EVENTS = 200` is a sliding window — a
long `rlm` run emits far more than 200 events and the reducer would lose early
history. Since the UI is now `rlm`-only, raise it substantially or remove it.

### 5.4 Static-value cleanup

The `0.35` degraded threshold in `report-rail.tsx` becomes a named, documented
constant (a small `rlm-config.ts` module is the natural home). `FIXTURE_RUN_META`
stays — the fixture path needs *some* metadata — but is clearly marked the dev
fixture's own data; real runs already receive `sourceLabel` / `sourceNote` /
`projectId` dynamically from the run object.

### 5.5 Verify

A real `--mode rlm` run renders correctly through every region; fix whatever the
real event stream surfaces that the hand-authored fixture did not.

> **Deletion-risk note.** §5.1 is deletion-heavy and is the inverse of Phase 4's
> additive work. Two of the four pre-existing eslint errors live in
> `progress-strip.tsx` / `telemetry-strip.tsx` — deleting those files clears them;
> `library-filters.tsx` is the separate `/library` page and is untouched. Mitigation:
> reference-checked, incremental, suite-green-per-task deletion.

---

## 6. Build sequence

1. **Frontend cutover + go-live wiring** — no coordination needed; start immediately.
   Delete the old lab UI (§5.1), `rlm`-only the run-start path (§5.2), raise the cap
   (§5.3), static cleanup (§5.4). After this step the lab is RLM-only and a real
   `rlm` run renders live — with a trunk-only tree until step 2.
2. **Backend — the 3 events** — cofounder-coordinated; §4. After this a real run's
   exploration tree fully populates with candidate branches.
3. **End-to-end verification** — a real `--mode rlm` smoke run renders through every
   region including the live tree (§5.5).

Frontend-first is deliberate: it has no external dependency and leaves the lab in a
working, RLM-only state at the end of step 1, so step 2 can proceed at the cofounder's
pace without blocking the cutover.

---

## 7. Testing

- **vitest** — the existing reducer/component tests stay green (minus the tests for
  deleted old-UI components); new tests cover the `rlm`-only run-start path and the
  raised cap.
- **Playwright e2e** — `/lab?rlmFixture=1` is kept as the deterministic e2e path; add
  an assertion that the lab is RLM-only (no old-UI elements render).
- **Backend `pytest`** — the 3-event emission, per the handoff doc's test checklist.
- **A real `--mode rlm` smoke run** — the final manual gate (§5.5).

TDD per task: failing test → watch it fail → implement → watch it pass → commit.

---

## 8. Risks & coordination

- **The cofounder ask (§4)** — Option B's prompt + primitive change is the
  critical-path coordination item. If the cofounder cannot change the root prompt,
  fall back to handoff-doc Option A (run-level score-delta inference in `run.py`) —
  an approximation, flagged in code.
- **The `rlm` orchestrator is not Phase-5-proven** — a real run may be partial,
  failed, or score-capped at 0.35 (the I7 cap). The lab renders that **honestly**;
  this plan fixes plumbing, not run quality. Run quality is Phase 5.
- **Deletion fallout (§5.1)** — mitigated by reference-checked, incremental,
  suite-green deletion.

---

## 9. Sequencing & relationship to Phase 6

This work depends on Phase 4 (**PR #70**) being on the base — it deletes the old UI
and makes the RLM lab the only lab, so the RLM lab must already be there. The go-live
work is a **separate branch / PR**, branched off Phase 4. If #70 is not yet merged
when implementation starts, branch off `feat/rlm-phase4-frontend` and rebase onto
`main` once #70 lands. One PR for this work; do not merge without the user's approval.

**Relationship to Phase 6.** The RLM-only frontend cutover (§5.1–5.2) is the
*frontend half* of the brief's Phase 6 (GitHub issue #63), **pulled forward** — ahead
of the brief's Phase-5-then-6 ordering — at the user's explicit direction (2026-05-22:
"RLM is the only option"). Issue #63's remaining work — deleting the backend
`PipelineStage` advancement, the Gate 1/2/3 control-flow, the hardcoded improvement
paths, and the old `run_state` / `agent_log` SSE event types, plus the
`README` / `system_overview.md` / `CLAUDE.md` rewrites — stays a later, separate
effort. This plan does the user-visible cutover now and deliberately leaves backend
dead code and the doc rewrites untouched, so it does not collide with a future full
Phase 6. The `dashboard_event` SSE *frame* type is **kept** — it is the transport the
RLM events ride; only the old `run_state` / `agent_log` frames are Phase-6 deletions.
