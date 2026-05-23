# Frontend Static-Values Audit — 2026-05-22

**Branch:** `feature/static-values-audit-2026-05-22`
**Worktree:** `/Volumes/CS_Stuff/openresearch-audit/`
**Auditor:** S1 inventory subagent
**Date:** 2026-05-22

---

## Summary

Full read-through of all ~55 files under `frontend/src/` in the audit worktree. Five category-A violations identified, all located in the **lab partition** (`components/lab/`). No violations in library, paperbench, demo, hooks, lib, or app-route files. All category-C fixtures are contained. No backend-data gaps identified in the current UI surface.

| Category | Count |
|---|---|
| A — Violations (fix) | 5 |
| B — Chrome (leave) | Remainder |
| C — Dev fixtures (contained) | 3 clusters |
| Backend gaps | 0 |

---

## Violations — Category A

All violations are in the **lab partition**. No violations in library, paperbench, demo, hooks, lib, or api-route files.

### Lab partition

| # | File | Line(s) | Literal | Why it is A |
|---|---|---|---|---|
| 1 | `components/lab/script-panel.tsx` | 71 | `pdf?.fileName ?? "paper.pdf"` | Rendered in `.pdf-meta` alongside file size and page count — reads as the actual filename of the ingested paper. Honest empty would be `"—"`. |
| 2 | `components/lab/script-panel.tsx` | 88 | `pdf?.codePath ?? \`${run.outputDir}/code/paper.pdf\`` | The synthesized path is rendered as if it is the real generated code root. The fallback path is not meaningful (it mixes the PDF path with a code directory). Honest empty: `"—"` or omit the row. |
| 3 | `components/lab/script-panel.tsx` | 106 | `benchmark?.benchmarkName ?? "PaperBench-style final benchmark"` | Rendered as the title in `.benchmark-title` inside `.benchmark-card`. Reads as the actual benchmark name assigned to this run. Honest empty: `"—"` or `"No benchmark named yet"`. Per spec resolved-edge #1, this is the canonical example. |
| 4 | `components/lab/script-panel.tsx` | 109 | `benchmark?.paperbenchTaskId ?? "pending evaluator output"` | Rendered in `.benchmark-subtitle` — positioned as the subtitle/ID row beneath the benchmark name. Reads as the task-ID string returned by the evaluator. The phrase "pending evaluator output" conveys a specific evaluator state that may not match the actual run state. Honest empty: `"—"` or `"pending"` without the evaluator-output framing. |
| 5 | `components/lab/agent-info-helpers.ts` | 30 | `shortHash()` returns `"pending"` when value is falsy | Called at `script-panel.tsx:74` as `sha256:{shortHash(pdf?.sha256)}`, rendering `sha256:pending` inside a `.pdf-hash mono` div. The div is explicitly a hash value display — `sha256:pending` reads as if "pending" is the hash value. Honest empty: display nothing, `"sha256:—"`, or omit the row when sha256 is absent. |

### Remediation guidance (for S2)

- Items 1, 3: Replace `?? "<literal>"` with `?? "—"` (the house-style honest-empty for string data slots).
- Item 2: Replace the synthesized path fallback with `?? "—"` and add a comment explaining the slot is only meaningful after code generation completes.
- Item 4: Replace with `?? "—"` — do not fabricate evaluator state in a subtitle slot.
- Item 5: In `shortHash()`, return `""` (empty string) or `null` when value is falsy, and update `script-panel.tsx:74` to suppress the hash row entirely when no sha256 is available (`pdf?.sha256 ? <div>sha256:{shortHash(pdf.sha256)}</div> : null`).

---

## Backend Gaps

**None identified in current UI surface.**

The `phase3-5-review-findings.md` file referenced in the audit spec was **absent** from this worktree (it exists only as an untracked file in the main checkout on a different branch). No cost_usd field, no cost rendering component, and no misleading backend-value display were found in the audited files. If a cost field is added in the future, the `telemetry-strip.tsx` component is the likely landing zone — that file is clean and uses honest `"—"` fallbacks throughout.

---

## Category-C Containment Evidence

Three clusters of fixture / test-only data were identified. All are confirmed contained — none render directly in production UI.

### C1 — `DEMO_WORKSPACE` in `lib/demo/node-runner.ts`

**Location:** `/Volumes/CS_Stuff/openresearch-audit/frontend/src/lib/demo/node-runner.ts`, lines 55–77

**Content:** Hardcoded PPO paper text excerpts (`"src_1"`, `"src_2"`, `"src_3"`, CartPole narrative text, `project_id: "prj_e2e_test"`).

**Containment verdict:** CONTAINED. `DEMO_WORKSPACE` is a JavaScript object that is serialized and passed as a CLI argument to a Python subprocess (`spawn(pythonBinary(), [..., JSON.stringify(DEMO_WORKSPACE)])`). The object never flows into a React component or SSE payload. The rendered UI shows what the backend returns after processing — not the raw workspace object. The `prj_e2e_test` project ID is only used for the fixture launch path; it is not persisted to `localStorage` or returned to the UI as a real run identifier.

### C2 — `defaultTopologyFixture` in `lib/pipeline/__fixtures__/default-topology.ts`

**Location:** `/Volumes/CS_Stuff/openresearch-audit/frontend/src/lib/pipeline/__fixtures__/default-topology.ts`

**Content:** A fully elaborated topology fixture with hardcoded node labels, statuses, and edge definitions.

**Containment verdict:** CONTAINED. The file lives in a `__fixtures__/` directory. No import of `defaultTopologyFixture` was found in any production code file during the full audit of `frontend/src/`. The fixture is test-only by location and by import graph. Any future production import would be a containment break and should be treated as a category-A violation at that point.

### C3 — `91.4`, `492.3`, `"Reproduced With Caveats"` literals

**Location:** `components/lab/script-panel.tsx` (comments, lines 22–30) and test files (not in `frontend/src/components/` production code).

**Containment verdict:** CONTAINED. These values appear only in:
1. JSDoc/inline comments in `script-panel.tsx` that document the historical gate-regression bug.
2. Vitest test fixtures (outside the production `frontend/src/components/` tree).

The `pipelineComplete` guard at `script-panel.tsx:31` (`const pipelineComplete = run.payload?.summary.stage === "complete"`) is **intact** and gates the score/metrics display block. The historical regression that caused these values to render prematurely is fixed; the comments are documentation artifacts, not live literals.

---

## Out-of-Scope Notes (Category B Clarifications)

These items were examined closely and decided as category B. They are documented here so S2 does not re-examine them independently.

### `buildFixtureMeta` sourceLabel in `lib/demo/server-fs.ts`

`buildFixtureMeta()` (lines 86–110) hardcodes `sourceLabel: "ReproLab PPO demo paper"` and a `sourceNote` string describing the fixture. These appear in the sidebar Recent section and command palette for fixture-based runs. Classified **B** because the string accurately describes the actual PPO demo paper file that is used (not a fabricated claim), and it executes only on the fixture run path (when no real paper has been uploaded).

### `isoAt()` timestamps in `lib/demo/pipeline-dashboard.ts`

`isoAt()` (line 741–743) synthesizes all dashboard events with 2026-05-09 timestamps. `lastUpdated: "2026-05-09T15:00:00.000Z"` appears in `AgentNode` objects (lines 455, 762). Classified **B** because `payload.events` and `payload.initialSnapshot` are **not consumed** by any current rendered component. `dashboardEvents` in `use-run.ts` is populated exclusively from SSE `dashboard_event` frames (live backend events), not from `payload.events`. These are latent data in the payload object that no renderer currently reads. If a future component adds a `payload.events` consumer, these timestamps would immediately become category A and should be replaced with `new Date().toISOString()` at render time.

### `startFixtureRun` in `hooks/use-run.ts`

The function name suggests fixture-data injection (per spec resolved-edge #3). Examined and confirmed: `startFixtureRun` posts to the real API (`POST /api/demo`) with `mode: "sdk"` — it is a "start-without-file-upload" entry point, not a fixture-data injector. The `?rlmFixture=1` flag and `FIXTURE_RUN_META` referenced in the spec's warning do not exist in the current codebase. **B** — naming is misleading but behavior is clean.

### `"pending"` in `agent-info-helpers.ts:30` — other callsites

`shortHash()` is called from `script-panel.tsx:74` only. No other callsite in the production codebase renders the `"pending"` string from `shortHash()`. The violation is exclusively the script-panel hash-display context (documented as violation #5 above).

### `formatBytes()` returning `"n/a"` and `verdictLabel()` returning `"Pending"`

These are honest empty-state strings in `agent-info-helpers.ts` (lines 13, 21). `"n/a"` for an absent byte count and `"Pending"` for an absent verdict are unambiguous state indicators, not fabricated data values. **B**.

---

## Methodology

**Files read:** All ~55 files under `/Volumes/CS_Stuff/openresearch-audit/frontend/src/` were read in full using the Read tool with absolute paths rooted at the worktree. Files read included:

- All page files: `app/page.tsx`, `app/layout.tsx`, `app/{lab,demo,library,paperbench,unlock}/page.tsx`
- All API routes: `app/api/demo/route.ts`, `events/route.ts`, `final-report/route.ts`, `arxiv/route.ts`, `resume/route.ts`, `source-pdf/route.ts`, `runs/list/route.ts`, `runs/recent/route.ts`, `models/route.ts`, `paperbench/route.ts`, `pipeline/topology/route.ts`, `unlock/route.ts`, `lab/debug-bundle/route.ts`
- All lab components: `script-panel.tsx`, `agent-info-helpers.ts`, `lab-shell.tsx`, `telemetry-strip.tsx`, `progress-strip.tsx`, `command-palette.tsx`, `upload-view.tsx`, `node-config.ts`, `agent-info-panel.tsx`, `agent-timeline-rail.tsx`, `gate-chips.tsx`, `hermes-audit-panel.tsx`, `node-card.tsx`, `floating-agent-window.tsx`, `lab-sidebar.tsx`, `failure-summary.ts`, `shared-helpers.ts`, `status.tsx`, `shortcut-overlay.tsx`, `resizable-split.tsx`, `pan-wrap.tsx`, `lab-canvas.tsx`, `timeline-panel.tsx`
- Library components: `library-table.tsx`, `library-filters.tsx`, `library-shell.tsx`
- Paperbench component: `paperbench-client.tsx`
- Demo component: `demo-overlay.tsx`
- Hooks: `use-run.ts`, `use-topology.ts`, `use-command-palette.ts`, `use-canvas-keyboard-nav.ts`, `use-shortcut-overlay.ts`, `use-pan.ts`
- Lib files: `demo/pipeline-dashboard.ts`, `demo/server-fs.ts`, `demo/node-runner.ts`, `demo/server-run.ts`, `demo/server-payload.ts`, `demo/run-staleness.ts`, `demo/progress.ts`, `pipeline/__fixtures__/default-topology.ts`, `pipeline/topology.ts`, `pipeline/topology-context.tsx`, `pipeline/topology-helpers.ts`, `pipeline/layout.ts`, `runs/server-list.ts`, `models/server-fetch.ts`, `paperbench/runner.ts`, `auth/demo-gate.ts`, `presentation-mode.tsx`, `user-prefs.ts`, `events/contract.ts`

**Classification method:** Each file was checked for (a) `?? "<literal>"` fallbacks in rendered JSX, (b) hardcoded numeric or string data in value slots, (c) placeholder text that reads as real data, and (d) fixture data with import-graph tracing. Borderline cases were resolved by checking the surrounding JSX element class and label context, then applying the spec's discriminator: "does the value read as real data or as a state indicator?"

**No files in `frontend/src/` were modified by S1.** The only file written by S1 was this findings document.

---

## Test evidence (S4)

All four verification layers exercised in the worktree `/Volumes/CS_Stuff/openresearch-audit/` on branch `feature/static-values-audit-2026-05-22`.

| Layer | Command (from `frontend/`) | Result |
|---|---|---|
| Types | `npx tsc --noEmit` | clean (no output) |
| Lint  | `npm run lint` | 4 errors, 9 warnings — all in pre-existing files this PR did not touch (`components/library/library-filters.tsx`, `lib/pipeline/layout.ts`). Verified identical on `main` HEAD. No new errors. |
| Unit  | `npx vitest run` | 14 test files, **75 passed (75)**. Includes 16 new tests added by this audit (10 per-violation + 1 regression + 4 helper + 1 helper file count). |
| E2E   | `npx playwright test e2e/static-values-route-pass.spec.ts` | **5 passed (5)** in 3.7s — one per route (`/lab`, `/demo`, `/library`, `/paperbench`, `/unlock`). Dev server (`PORT=3001 npm run dev`) ready in 209ms via Turbopack; backend not required (server-side fetches gracefully resolve to null and pages render their honest empty state). |

### New tests added in this audit

- `frontend/src/components/lab/script-panel.test.tsx` — 10 tests, two per violation 1–4 (renders-from-source + honest-empty), plus a present/absent pair for the sha256 hash row from violation 5.
- `frontend/src/components/lab/agent-info-helpers.test.ts` — 4 tests pinning `shortHash()`'s new contract (returns `""` for null/undefined/empty, returns first 12 chars for truthy).
- `frontend/src/components/lab/script-panel.regression.test.tsx` — 1 test pinning the historical "91.4 / 492.3 / Reproduced With Caveats" fixture-leak from `script-panel.tsx:22-30`. Constructs an inverse-shaped fixture (complete-looking benchmark + non-complete stage) and asserts none of the leak markers reach the DOM.
- `frontend/e2e/static-values-route-pass.spec.ts` — 5 Playwright tests; load each public route with no data, assert no leak markers in the rendered body, screenshot full-page into `docs/design/audit-2026-05-22-screenshots/`.

### Screenshots

Committed at `docs/design/audit-2026-05-22-screenshots/`:

| File | Size | Route |
|---|---|---|
| `lab.png` | 41 KB | `/lab` — upload view, ReproLab sidebar, "No recent runs." |
| `demo.png` | 43 KB | `/demo` — same upload view (presentation-mode variant) |
| `library.png` | 40 KB | `/library` — empty table state |
| `paperbench.png` | 58 KB | `/paperbench` — start-a-run form, "no bundles vendored" |
| `unlock.png` | 63 KB | `/unlock` — access-code form |

The lab screenshot was visually inspected and confirms an honest empty state: sidebar with "No recent runs.", upload affordance, format-hint arxiv placeholder, model dropdown defaulting to Sonnet — no fabricated data anywhere.

---

## Post-audit follow-ups (borderline items, not fixed in this PR)

Two items surfaced during S2 code-quality review and visual screenshot inspection that sit on the boundary between category A and B. They are NOT fixed in this PR — they would be a small follow-up audit. Flagged here so a reviewer can re-classify:

1. **`script-panel.tsx:79` — `<a download="paper.pdf" ...>`** — the HTML `download` attribute used when the user clicks the "Download" link. If `pdf?.fileName` is missing the file saves as `paper.pdf`. Not rendered text — appears only in the browser's download dialog. Borderline because it IS data the user encounters about their run, but it is conventional default-filename UX. Suggested follow-up: derive from `pdf?.fileName` with a generic fallback (`source-paper.pdf`) or omit the attribute entirely (let the browser default to the response's `Content-Disposition`).

2. **`upload-view.tsx:107` — `placeholder="arxiv.org/abs/2303.04137"`** — the arxiv input's HTML `placeholder` attribute uses a specific paper id (2303.04137 = "Sparks of AGI"). Ghost-text that disappears on type. Conventional format-hint UX. Could be misread as "the app has processed this paper." Suggested follow-up: change to a generic format example like `arxiv.org/abs/XXXX.XXXXX`.

Both are category-B by the strict definition (HTML attributes ≠ rendered data values) but worth a documented decision rather than silent acceptance.

---

## Branch summary

Final commit sequence on `feature/static-values-audit-2026-05-22`:

```
50bf187  chore(frontend): script-panel.test — drop unused screen import + dedupe NOTE
e6aafea  fix(frontend): script-panel — suppress sha256 row when hash unavailable
4ee3766  fix(frontend): script-panel — show honest-empty for benchmark task-id slot
c1fdac7  fix(frontend): script-panel — show honest-empty for benchmark name slot
06f3608  fix(frontend): script-panel — derive code root path from source, not synthesized fallback
b9859f9  fix(frontend): script-panel — derive pdf filename from source, not literal
693cf31  docs: frontend static-values audit — inventory (S1)
c4c5408  docs: implementation plan for frontend static-values audit (2026-05-22)
785fa11  docs: design spec for frontend static-values audit (2026-05-22)
```

Plus the S4 commit (regression test + route-pass spec + screenshots + this doc update) added with this commit.
