# Rubric Climb Panel + Cross-Model Leaderboard — Design Spec

_Date: 2026-05-23 · Status: **design (no user in loop — assumptions made explicit)** · Next: writing-plans._

This spec covers two coupled product surfaces that turn invisible agent progress into something a viewer can feel:

1. **Live rubric climb panel** — a band-2 enrichment of the lab UI that shows the rubric score climbing as `verify_against_rubric` fires, with per-area pass/fail flip highlights, candidate attribution for each jump, and cost/wall-clock alongside.
2. **Cross-model leaderboard** — a new `/leaderboard` page ranking how different RLM root models perform on the same paper. Read-only in this delivery; the larger surface (re-run, per-role picker, dynamic budget) stays deferred to the already-approved cleanup-condensation-leaderboard spec's Phase 4.

This spec is forward-compatible with `docs/superpowers/specs/2026-05-23-cleanup-condensation-leaderboard-design.md` (the larger Phase-4 vision) — it adds a strict subset of that surface without contradicting any locked decision there.

---

## 1. Motivation

A reviewer watching a run today sees a 4-band lab UI: header, rubric strip, workspace (REPL rail | exploration canvas | report rail), and primitive history. The rubric strip shows a current score + bar + bar-sparkline. The report rail shows a per-area breakdown. The two are correct but quiet: a reviewer cannot feel the climb because the big number does not animate, area flips are not highlighted, and there is no attribution from a score jump back to the candidate that produced it. The "spent $0.42, climbed 0.37 points" story is split across two non-adjacent regions.

Separately, the project now has multiple supported RLM root models (gpt-5, claude, qwen3-coder, kimi-k2.5 via Featherless) and no way to compare them. Demo viewers cannot answer the obvious "which model is best on this paper?" question without manually opening multiple run dirs.

A Microsoft VP review is imminent. The bar is demo-ready, honest, and unembarrassing.

## 2. North star

A reviewer opening the lab during a live run watches the big rubric number tick up, sees individual rubric areas flip from fail → pass with a tasteful highlight, and reads "+0.18 from candidate: <title>" inline beside the jump. The same panel renders for a completed run on the same URL — the localStorage auto-resume and SSE replay paths work unchanged.

A reviewer visiting `/leaderboard` sees every completed run ranked by score, with paper / mode / model / cost / wall-clock columns, can sort by any column, and can click through to the per-run detail in the lab.

Measured as:
- The big rubric score animates smoothly between rubric_score events (no jump cuts).
- Per-area flips fail→pass and partial→pass land a 1.2-s subtle background tint that fades.
- The candidate that produced the most recent ≥+0.05 jump is named inline.
- `/leaderboard` shows ≥3 rows on a fresh `docker compose up` with fixture data installed.
- No new SSE event types — the existing `rubric_score` + `candidate_proposed` + `candidate_outcome` + `run_complete` events carry the data.
- No new chart dependency in `frontend/package.json`.

## 3. Locked decisions (assumptions, user not in loop)

| # | Question | Decision | Rationale |
|---|---|---|---|
| 1 | Where does the rubric climb live? | **Enrich band 2 (`RubricStrip`) in place.** | A new band changes layout for every existing consumer test. The strip already owns the score-climb intent; widen its responsibility, not the layout. (Advisor agreed.) |
| 2 | Does Feature 1 need a new SSE event type? | **No.** | All required signal (score, target, per-area scores + status, iteration, candidate id, candidate outcome) is already on `rubric_score`, `candidate_proposed`, `candidate_outcome`. New event = wider sanitizer surface = risk for no gain. |
| 3 | Per-area flip detection — where? | **Inside `useRlmRun`'s `foldRubricScore`.** Track `rubric.previousAreas` (the snapshot prior to the latest event). | Component-level diffing means stale state on re-renders. The reducer is the source of truth; let it surface the diff. |
| 4 | Candidate attribution — how? | **Derive in the reducer.** Maintain a "live candidate" pointer: when `foldCandidateProposed` or `foldCandidateOutcome` runs, update `rubric.attributableCandidate: { id, title, outcome \| null }`. When `foldRubricScore` runs, the current pointer is captured as the new event's `attribution`. | The tree already attaches scores to candidate nodes via `frontierParent`; we just expose the linkage. |
| 5 | Sparkline shape | **SVG line chart** (replaces the bar sparkline). Same `rubric.series` data. | The user spec explicitly says "sparkline / line chart"; a line communicates trajectory better than discrete bars. SVG = no new dep. |
| 6 | Animation system | **CSS transitions + a count-up RAF tween for the big number.** ~400 ms cubic-out for the counter; 1.2-s background-color tint for area flips. | No animation lib (Framer Motion, etc.) — matches "flat surfaces, hairline borders, sentence case" visual language. |
| 7 | Leaderboard data model | **Adopt the cleanup-spec shape: `model_config: {planner, executor, verifier, grader}` + `mode` + `started_at` + `completed_at` in `final_report.json`.** Fill `planner` and `executor` from the run (root model + sub-agent model); leave `verifier` and `grader` `null` until the future picker lands. | Single-string `model` would commit the team to a migration later. Adopting the future shape with nullable unknowns is a strict subset, no contradiction. |
| 8 | Leaderboard persistence | **Compute at request time from filesystem** (`runs/*/final_report.json` + `demo_status.json`). No SQLite projection in this delivery. | <100 runs locally; the cleanup spec's Phase 4 introduces the projection if needed. Avoid premature persistence. |
| 9 | Leaderboard endpoint name | **`GET /leaderboard`** on the backend; **`/api/demo/leaderboard`** Next.js proxy; **`/leaderboard`** page route. | Matches the cleanup spec verbatim. `GET /runs/leaderboard` would collide with the `/runs/{id}` family. |
| 10 | Demo-gate behavior | **`GET /leaderboard` is NOT gated by `REPROLAB_DEMO_SECRET`.** | The gate is for mutating routes that start runs. Reads are public; same policy as `GET /runs`. |
| 11 | Replay support | **None added.** The existing SSE-replay path (server replays `dashboard_events.jsonl` on reconnect) already works for completed runs — the lab page consumes events identically live vs. replayed. New work would be redundant. | The user's spec asks for replay; verify it works as-is, fix any gap if found. Don't add a parallel "replay mode." |
| 12 | Playwright scope in this delivery | **In-session: criteria 3, 5, 6, 7, 8** (driven by `npm run dev` + fixtures + playwright MCP). **Deferred (docker-dependent): criteria 1, 2, 4** — they require `docker compose up --build` + real LLM creds + a real run. Plan documents the deferred criteria; commit boundaries make in-session progress durable. | Honest scoping. The user's spec says "FAIL or SKIP means not done — keep going." Across sessions, yes. In one session: finish the in-session bucket cleanly. |

## 4. Architecture overview

### 4.1 Feature 1 — Rubric climb panel (band 2 enrichment)

Component file: `frontend/src/components/lab/rlm/rubric-strip.tsx` (extended in place).

New rendered sub-regions inside the strip, in order:
1. **Big score** — already there; wrap in a count-up tween for transitions.
2. **Baseline → target bar** — already there; preserve.
3. **Line-chart sparkline** — replaces the bar sparkline. Same `rubric.series` input. SVG path, ≤120 × 28 px, hairline stroke.
4. **Per-area chip row** — compact horizontal list of areas with status glyphs (✓ / ◐ / ✗). Areas that flipped from `fail|partial → pass` (or `fail → partial`) at the latest event get the `.justFlipped` modifier — a subtle background tint that fades over 1.2 s via CSS animation.
5. **Climb annotation** — already there ("baseline X.XX → Y.YY, +Δ vs target Z.ZZ"); preserve, but when `rubric.attributableCandidate` is non-null AND the latest delta is ≥+0.05, append "— from candidate <title>".

Inputs (props):
```ts
interface RubricStripProps {
  rubric: RlmRunState["rubric"];  // extended; see §4.3
}
```

No new props — all new data rides on the existing `rubric` state object. This keeps the consumer signature stable.

### 4.2 Feature 1 — Reducer extensions (`use-rlm-run.ts`)

Two new fields on `RlmRunState.rubric`:

```ts
rubric: {
  current: number | null;
  baseline: number | null;
  target: number | null;
  series: Array<{ iteration: number; score: number }>;
  areas: RubricArea[];
  previousAreas: RubricArea[];          // NEW — snapshot before latest event
  attributableCandidate: {              // NEW — the live candidate pointer
    id: string;
    title: string;
    outcome: TreeNode["outcome"] | null;
  } | null;
}
```

Update rules:
- `foldCandidateProposed`: set `attributableCandidate = { id, title, outcome: null }`.
- `foldCandidateOutcome`: if `attributableCandidate?.id === ev.candidate_id`, update `outcome`. (Don't clear on `promoted` or any other outcome — the next `candidate_proposed` will overwrite.)
- `foldRubricScore`: snapshot the current `areas` into `previousAreas` BEFORE overwriting; keep `attributableCandidate` (caller will read whatever was current at this iteration).

The `previousAreas` snapshot is the same array reference used until the next `rubric_score` event — the component compares `areas[i].status` against `previousAreas[i].status` (matched by `area` name, not index, to survive area reordering across runs).

### 4.3 Feature 1 — Animation specifics

Count-up tween: a custom React hook `useCountUp(target: number, durationMs = 400)` that owns a `requestAnimationFrame` loop, eases with `t => 1 - (1 - t)**3` (cubic-out), and returns the current displayed value. Cancels and restarts on `target` change.

Area flip tint: pure CSS keyframe animation `@keyframes flipTint { 0% { background: <accent>; } 100% { background: transparent; } }`, applied via the `.justFlipped` modifier. No JS.

Big score smoothing: when `rubric.current` is `null`, the tween input is treated as `0` but the rendered string is `"—"` to preserve the honesty-rule (spec §14 in rlm-pivot-brief: no fabricated numbers).

### 4.4 Feature 2 — Leaderboard data flow

```
Browser  →  GET /api/demo/leaderboard  (Next.js proxy, server-side)
           →  GET /leaderboard          (FastAPI backend)
           →  scan runs/*/final_report.json + demo_status.json
           →  build LeaderboardRow[] sorted by overall_score desc
           →  return as JSON
```

Backend route file: `backend/routes/leaderboard.py` (new), mounted in `backend/app.py::create_app`.

```python
@router.get("/leaderboard")
def list_leaderboard_runs(
    paper: str | None = None,
    mode: Literal["rlm", "rdr"] | None = None,
    order_by: Literal["score", "cost", "time", "finished_at"] = "score",
    limit: int = 50,
) -> list[LeaderboardRow]: ...
```

`LeaderboardRow` Pydantic schema (matches the cleanup-spec shape):

```python
class RoleModels(BaseModel):
    planner:  str | None = None
    executor: str | None = None
    verifier: str | None = None
    grader:   str | None = None

class LeaderboardRow(BaseModel):
    project_id: str
    paper_id: str         # derived from final_report.paper["id"] or project_id
    paper_title: str | None
    mode: Literal["rlm", "rdr"] = "rlm"
    models: RoleModels    # field name aligned with final_report.json (see §4.5)
    overall_score: float
    meets_target: bool
    degraded: bool
    cost_usd: float | None
    iterations: int
    wall_clock_s: float | None  # completedAt - startedAt
    sandbox: str | None         # from final_report or demo_status
    started_at: str | None      # ISO-8601
    completed_at: str | None    # ISO-8601
    verdict: str                # "reproduced" | "partial" | "failed"
```

The aggregator gracefully handles legacy runs that pre-date the new fields (returns `null` for missing data; never raises).

### 4.5 Feature 2 — `final_report.json` extensions

`RLMFinalReport` (`backend/agents/rlm/report.py`) gains four optional fields, all backward-compatible:

```python
class RLMFinalReport(BaseModel):
    # existing fields ...
    mode: Literal["rlm", "rdr"] = "rlm"
    model_config: dict[str, str | None] = Field(default_factory=lambda: {
        "planner": None, "executor": None, "verifier": None, "grader": None
    })
    started_at: str | None = None     # ISO-8601 UTC
    completed_at: str | None = None   # ISO-8601 UTC
```

The Pydantic field name `model_config` collides with `BaseModel.model_config` (a Pydantic-2 reserved name). **Mitigation:** rename the field on the wire to `models` and use Pydantic alias:

```python
models: dict[str, str | None] = Field(
    default_factory=lambda: {"planner": None, "executor": None, "verifier": None, "grader": None},
    alias="model_config",
)
# pydantic config: populate_by_name=True
```

Or simpler: name the field `models` everywhere (in JSON too) and align the cleanup-spec to use `models` instead of `model_config` going forward. **Decision:** use `models` on disk and on the wire to avoid the Pydantic clash entirely; document this in the cleanup-spec's followups.

Writer (`backend/agents/rlm/run.py::_finalize`) populates these from:
- `models.planner` ← `llm_model` (the root-model id from `_build_llm_client`)
- `models.executor` ← `ctx.agent_model` (the sub-agent model id from `_resolve_agent_runtime`)
- `models.verifier` ← `None` (no dedicated verifier role today)
- `models.grader` ← `None` (no dedicated grader role today)
- `started_at` ← `RunContext.started_at_utc` (already wired) or `_write_demo_status`'s timestamp
- `completed_at` ← `datetime.now(timezone.utc)` at write time
- `mode` ← `"rlm"` (rdr-mode runs will set their own when that lands)

### 4.6 Feature 2 — Frontend page

Page file: `frontend/src/app/leaderboard/page.tsx` (new).

Server-component shell that fetches `/api/demo/leaderboard` on the server, renders a `<LeaderboardTable>` client component. Columns:

| Column | Source | Notes |
|---|---|---|
| Paper | `paper_title` ?? `paper_id` | links to `/lab?projectId=<project_id>` |
| Mode | `mode` | rlm / rdr badge |
| Planner | `models.planner` | "—" if null |
| Executor | `models.executor` | "—" if null |
| Score | `overall_score.toFixed(2)` | red ≤0.35 (degraded), green ≥target |
| Meets target | `meets_target` | ✓ / ✗ |
| Cost | `cost_usd` | "$X.XX" or "—" |
| Iterations | `iterations` | int |
| Time | `wall_clock_s` | formatted "Xm Ys" |
| Verdict | `verdict` | reproduced / partial / failed |
| Finished | `completed_at` | relative time |

Sortable by any column (client-side after server fetch — small N). Empty state: "No completed runs yet — start one from the lab." with a link to `/lab`.

Filter chips above the table: paper, mode. (Defer multi-select / score-range to Phase 4.)

Visual language: hairline borders, sentence case, monospace numerics, no zebra stripes. Reuses the `frontend/src/components/library/` table primitives where compatible.

### 4.7 Egress / sanitizer impact

Zero. No new SSE events, no widening of `sanitize_iteration`, no new sentinels. The leaderboard endpoint reads file artifacts that are already disk-resident; it does not stream events.

The only new corpus-exposure surface to audit: `paper_title` in `LeaderboardRow`. We use `final_report.paper["title"]` which is *already* extracted text — the same content the lab UI already exposes via `runMeta.paperTitle`. No new threat.

## 5. Component / file inventory

**New backend files:**
- `backend/routes/leaderboard.py` — FastAPI router with `GET /leaderboard`.
- `tests/routes/test_leaderboard.py` — aggregation correctness + ordering + filter tests.
- `tests/rlm/test_final_report_models_field.py` — verifies `models` + `mode` + timestamps land in `final_report.json`.

**Modified backend files:**
- `backend/agents/rlm/report.py` — extend `RLMFinalReport` with the 4 new fields.
- `backend/agents/rlm/run.py::_finalize` — populate the new fields.
- `backend/app.py::create_app` — mount the new router.

**New frontend files:**
- `frontend/src/app/leaderboard/page.tsx` — server page shell.
- `frontend/src/app/leaderboard/leaderboard-table.tsx` — client table component.
- `frontend/src/app/leaderboard/leaderboard-table.test.tsx` — table render + sort + empty-state tests.
- `frontend/src/app/api/demo/leaderboard/route.ts` — Next.js proxy.
- `frontend/src/components/lab/rlm/use-count-up.ts` — count-up tween hook.
- `frontend/src/components/lab/rlm/use-count-up.test.ts` — hook tests.
- `frontend/src/components/lab/rlm/sparkline.tsx` — SVG line-chart sparkline.
- `frontend/src/components/lab/rlm/sparkline.test.tsx` — sparkline tests.
- `frontend/e2e/rubric-climb.spec.ts` — E2E for criteria 3, 5, 6, 7, 8 in-session bucket.

**Modified frontend files:**
- `frontend/src/hooks/use-rlm-run.ts` — extend `RlmRunState.rubric` with `previousAreas` + `attributableCandidate`; update folds.
- `frontend/src/hooks/use-rlm-run.test.ts` — assert new state shape and update rules.
- `frontend/src/components/lab/rlm/rubric-strip.tsx` — enrich with count-up, sparkline component, per-area chip row with flip tint, candidate attribution.
- `frontend/src/components/lab/rlm/rubric-strip.module.css` — new styles for chip row + flip tint + sparkline placement.
- `frontend/src/components/lab/rlm/rubric-strip.test.tsx` — extend tests: flip detection rendering, candidate attribution rendering, sparkline replacement.
- `frontend/src/components/lab/rlm/__fixtures__/rlm-run.fixture.ts` — extend with the flip + attribution scenario.

**Modified docs:**
- `CLAUDE.md` — note the new `/leaderboard` route and the new `final_report.json` fields.
- `system_overview.md` — document the leaderboard endpoint + the extended `final_report.json` shape. No new SSE events to document.
- `system_overview.md` — one-line addition: the leaderboard surface is described.

## 6. Test strategy

### 6.1 Backend (pytest)

- `test_leaderboard_aggregates_existing_runs`: scaffold 3 fixture run dirs under a tmp `runs/`, assert order-by-score returns them ranked correctly.
- `test_leaderboard_handles_legacy_runs_missing_fields`: a run with no `models` field returns `models: {planner: null, ...}` without raising.
- `test_leaderboard_handles_empty_runs_dir`: returns `[]`, status 200.
- `test_leaderboard_filters_by_paper`: 4 fixture rows across 2 papers, assert filter returns only the matching paper.
- `test_leaderboard_filters_by_mode`: same with mode.
- `test_leaderboard_order_by`: each `order_by` produces deterministic ordering.
- `test_final_report_models_field_written`: invoke `_finalize` with a known `llm_model` + `agent_model`, assert disk file carries them.
- `test_final_report_models_field_back_compat`: load an old-shape JSON, assert it parses and defaults are filled.
- `test_demo_gate_does_not_apply_to_leaderboard`: with `REPROLAB_DEMO_SECRET=foo` set, `GET /leaderboard` returns 200 without the header.
- Sanitizer sanity (regression): `test_sanitize_iteration_unchanged_after_leaderboard`: existing sanitizer fixtures still pass — confirms no widening occurred.

### 6.2 Frontend (vitest)

- `use-rlm-run.test.ts` extensions: rubric_score event sets `previousAreas` to the prior areas; candidate_proposed sets `attributableCandidate`; subsequent rubric_score retains the candidate pointer.
- `rubric-strip.test.tsx` extensions: areas that flipped fail→pass render with `.justFlipped`; line-chart sparkline renders with N-1 path segments for N series points; candidate attribution appears only when last delta ≥+0.05.
- `use-count-up.test.ts`: deterministic with fake timers — tween covers (start, end) over duration, eases cubic-out.
- `sparkline.test.tsx`: 0 points renders empty; 1 point renders a dot; ≥2 points renders a polyline; clamps to [0,1] on y axis.
- `leaderboard-table.test.tsx`: empty state renders the placeholder; sort-by-column changes row order; row click navigates to `/lab?projectId=<id>`.

### 6.3 Replay-fidelity test

- Add `frontend/src/components/lab/rlm/replay.test.ts` extension: pass the extended fixture through the reducer; assert at iteration N the rubric panel state matches a known snapshot (current score, baseline, areas[*].status, attributableCandidate, previousAreas). This is the "rubric panel state at each frame" check the task spec requires.

### 6.4 E2E (Playwright)

`frontend/e2e/rubric-climb.spec.ts` — five test cases:

1. **Empty state**: `/lab` with no in-flight run renders the rubric strip with "—" big number, no console errors.
2. **Live climb via fixture URL**: navigate `/lab?fixture=climb`, assert big number is "0.22" at first event, then ticks toward "0.53" — capture intermediate state via `await page.waitForFunction(() => Number(document.querySelector(...)) >= 0.4)`.
3. **Area flip highlight**: same fixture; assert at least one area row has the `.justFlipped` class within 200 ms of the rubric_score event arrival.
4. **Candidate attribution**: same fixture; assert the climb annotation contains "from candidate" after iteration 7.
5. **Leaderboard end-to-end**: seed 3 fixture run dirs via a backend test endpoint, visit `/leaderboard`, assert table renders 3 rows; sort-by-cost works; click row → navigates to `/lab`.

The `?fixture=climb` URL parameter wires the existing `rlmRunFixture` into the page at runtime — same mechanism the existing `?fixture=1` lab query uses. Add `?fixture=climb` to the lab page's allow-list.

Visual polish iteration (criterion 8): playwright MCP screenshots at 1440×900 and 390×844 for: `/lab` mid-climb, `/leaderboard` populated, `/leaderboard` empty. Iterate until visual language matches.

## 7. Failure modes considered

| Risk | Mitigation |
|---|---|
| `previousAreas` snapshot grows the state object on every iteration | The state is replaced by reference, not appended; only one snapshot lives at a time. |
| Count-up tween causes layout thrash on long climbs | Tween targets a single text node; `requestAnimationFrame` is throttled by the browser when offscreen. |
| `models.planner` field overlaps Pydantic v2's `model_config` reserved name | Use the field name `models` on disk and in code; document for the cleanup-spec to align. |
| Leaderboard aggregation is O(N) on every request | Acceptable for ≤100 runs. Cache via FastAPI's per-request scope only. Future: add the SQLite projection from the cleanup spec. |
| New `models` field breaks loading old `final_report.json` files | Pydantic `default_factory` fills missing field; `tests/rlm/test_final_report_models_field_back_compat` asserts this. |
| Reducer's `attributableCandidate` keeps a stale candidate after a long primitive-only stretch | The pointer is intentionally sticky — it represents "the most recent candidate the root was working on." For a flip during baseline build (no candidate yet), `attributableCandidate` is `null` and the strip omits the "from candidate" tail. |
| Sparkline label clobber when many iterations land | Sparkline is decorative (no labels); the score series may grow to ~30 points (max_iterations default). Polyline at 120 × 28 px handles this fine. |
| Animation looks janky on Safari | CSS `transition` and RAF tween are baseline-supported. Cubic-out is plain math, no `cubic-bezier` quirks. |
| Demo gate accidentally applied to the leaderboard | `test_demo_gate_does_not_apply_to_leaderboard` pins this. |
| Visual flip animation collides with screen-reader semantics | Flip tint is pure background-color; the row's `aria-label` is unchanged. Add `aria-live="polite"` to the per-area chip row to announce status changes. |

## 8. Out of scope

- **Re-run from a leaderboard row.** Deferred to cleanup-spec Phase 4.
- **Per-role model picker.** Deferred to cleanup-spec Phase 4.
- **Dynamic budget estimate UI.** Deferred to cleanup-spec Phase 4.
- **SQLite projection for the leaderboard.** Filesystem aggregation is sufficient at current scale.
- **rdr mode integration.** rdr-mode runs will set their own `mode` value; we don't block on rdr landing.
- **New SSE event types.** All Feature 1 derives from existing events.
- **Animation library dependency.** Custom CSS + RAF.
- **Backfilling `models` into the 4 legacy disk runs.** They show `models: {planner: null, ...}` on the leaderboard; documented honestly.
- **Multi-paper comparison views, head-to-head, model league tables.** Single ranked table only.

## 9. Acceptance criteria

A reviewer can confirm this delivery is complete by checking:

1. `pytest tests/ -n auto` returns ≥1083 passed / 0 failed / 1 xfailed (the existing T24 xfail) — confirms no regression.
2. New tests added per §6.1, §6.2, §6.3 — all pass.
3. `npm run lint && npx tsc --noEmit && npm test` green in `frontend/`.
4. A live RLM run renders the big rubric number with smooth count-up (no jump cuts) across at least two `rubric_score` events.
5. An area that flips fail→pass shows the `.justFlipped` tint within 200 ms.
6. The climb annotation surfaces "from candidate <title>" when a ≥+0.05 jump follows a `candidate_proposed`.
7. `/leaderboard` renders ≥3 rows from fixture data and supports sort by any column.
8. `final_report.json` of every new run carries `mode`, `models`, `started_at`, `completed_at`.
9. `CLAUDE.md` + `system_overview.md` reflect the new surface.
10. No new SSE event types; no widening of `sanitize_iteration`.
11. Playwright in-session criteria (3, 5, 6, 7, 8) pass; docker-dependent criteria (1, 2, 4) documented as next-session work.

## 10. Session context

This spec is written without user-in-loop. Assumptions are recorded in §3 — locked decisions a real brainstorming session would have surfaced as questions. If any decision turns out wrong, the spec is the place to fix it before the plan and the code drift.

The spec is forward-compatible with `docs/superpowers/specs/2026-05-23-cleanup-condensation-leaderboard-design.md` Phase 4: every new field, route, and component here is a strict subset of that future surface. The field name `models` (vs. `model_config`) is the only divergence; documented in §4.5 for the cleanup spec to absorb.

The Playwright "8 criteria, all passing on fresh `docker compose up`" bar is genuinely multi-session work (compose boot + LLM creds + a real run takes 2–10 minutes each iteration). The plan splits the bar at the commit boundary so an interrupted session lands durable progress; the docker-dependent criteria are documented as remaining work in the final summary, not silently skipped.

---

## Next step

Hand off to `writing-plans` to produce the executable per-task plan (file paths, test commands, commit boundaries, TDD steps) — saved to `docs/superpowers/plans/2026-05-23-rubric-climb-leaderboard.md`.
