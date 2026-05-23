# Frontend static-values audit — design

> Status: draft, awaiting user approval.
> Date: 2026-05-22
> Owner of execution: this PR.
> Output of execution: `docs/design/static-values-audit-2026-05-22.md` (findings),
> code fixes, new tests, screenshots — all in one PR.

## Mission (restated)

Every value the OpenResearch / ReproLab website shows the user must be dynamically
derived from real data — the live run/event stream, the backend API, or
component props/state. No baked-in numbers, scores, counts, titles, statuses,
names, costs, dates, or placeholder/mock values may render in the real product.
Audit the whole frontend (`frontend/src`), find every violation, fix it, prove
it with tests, and produce a findings doc.

## Categorisation rubric

- **(A) Violations — fix.** A literal representing data the user sees about a
  run / experiment / their account that is baked into the code instead of
  coming from the event stream, the API, or props/state. Includes hardcoded
  titles/scores/counts/costs/dates/run-ids, fabricated demo values, mock objects
  rendered outside dev paths, and `value ?? "descriptive default"` fallbacks
  where the default reads as if it were the data.
- **(B) Fine — leave.** Layout/config constants (spacing, sizes, durations,
  soft-cap counts), design tokens, static UI copy (button labels, section
  headings, field names, empty-state text).
- **(C) Dev fixtures — acceptable, must be contained.** Recorded fixtures
  reachable only from tests or an explicit dev flag. If a fixture value can
  leak into a real run's view, it is category-A.

### Resolved edges (decisions, not questions)

1. **Generic descriptive strings used as `value ?? "..."` fallbacks are
   category-A.** Example: `script-panel.tsx:106` renders
   `benchmark?.benchmarkName ?? "PaperBench-style final benchmark"` as the
   benchmark *title*. When no name exists, the default reads as if it were the
   benchmark's name. Replace with `"—"` or contextual honest text (e.g.
   `"Pending — no benchmark named yet"`), mirroring the existing
   `formatPercent → "—"` style in `paperbench-client.tsx`.
2. **Misleading-but-dynamic backend values are out-of-scope for fixes.**
   Documented as Backend Gaps in the findings doc, not fabricated and not fixed
   in this PR. Example: `cost_usd` under-reports per backend review M4
   (`primitive-internal LLM cost is invisible`). The frontend sources the value
   dynamically, so it is not a hardcoded violation. A follow-up PR either fixes
   the backend reporting or relabels the field as a rough estimate.
3. **The mission's named constructs `?rlmFixture=1` and `FIXTURE_RUN_META` do
   not appear in current `frontend/src`.** No URL-flag-gated fixture path
   exists today; `startFixtureRun` in `hooks/use-run.ts` is a misnamed
   "start-without-upload" entry that does not inject fixture data. The
   category-C containment check therefore becomes behavioural: load every
   audited route with no dev affordance, no projectId, and a deliberately
   incomplete `LiveDemoRunState`; assert no fixture-shaped strings appear in
   the rendered DOM. The known leak markers are `91.4`, `492.3`,
   `"Reproduced With Caveats"` — the historical bug from
   `script-panel.tsx:22-30`.

## Audit surface

| Surface | Files | Notes |
|---|---|---|
| Pages | `/lab`, `/demo`, `/library`, `/paperbench`, `/unlock` | `/` is a redirect to `/lab` — no data |
| Components | `lab/*` (24), `library/*` (3), `paperbench/*` (1), `demo/*` (1) | 29 files |
| Hooks/lib | `hooks/use-run.ts`, `lib/demo/*`, `lib/pipeline/*`, `lib/runs/*`, `lib/models/*`, `lib/paperbench/*` | Data layer the components consume |
| API routes | `app/api/**/*.ts` | Light pass; check for hardcoded JSON responses |

Pre-survey high-priority suspects (from grep + targeted reads during
brainstorming): `lab/script-panel.tsx`, `lab/command-palette.tsx`,
`lab/telemetry-strip.tsx`, `lab/upload-view.tsx`, `lab/node-config.ts`,
`lab/agent-info-helpers.ts`, `paperbench-client.tsx` header copy,
`lib/demo/pipeline-dashboard.ts`, `lib/demo/node-runner.ts`,
`lib/demo/server-fs.ts`, `lib/pipeline/topology-helpers.ts`,
`lib/pipeline/__fixtures__/default-topology.ts`.

## Methodology — subagent-driven, fan-out 3

```
S1 (Inventory) ─ sequential, first
   Reads every file in the audit surface.
   Produces docs/design/static-values-audit-2026-05-22.md.
   For each candidate literal:
     • Decide category A | B | C
     • If A: name the dynamic source it should consume
              (prop, event-stream field, API field, hook state)
              and propose the honest empty state
     • If B: note why (chrome, config constant)
     • If C: verify the containment path empirically
   Adds a Backend Gaps section for category-A values whose
     dynamic source does not yet exist on the backend.

(then in parallel — partition has no shared files)

S2 (Fix lab/) ──── components/lab/* + hooks/use-run.ts + lib/demo/*
S3 (Fix non-lab) ─ components/{library,paperbench,demo}/*
                    + lib/{pipeline,runs,models,paperbench}/*
                    + app/api/**

Each fix subagent works strict-TDD: failing test first, then fix.

(then sequential)

S4 (Test pass) ─── vitest + tsc + lint + Playwright
                   Records final test evidence into the audit doc.
```

S2/S3 partition has no shared files, so they merge cleanly without
coordination. Each fix subagent receives only the S1 findings table for its
partition; if a subagent disputes a verdict, it surfaces the dispute to the
integrator rather than acting unilaterally.

## Test plan

| Layer | Command (from `frontend/`) | Pass criteria |
|---|---|---|
| Unit | `npx vitest run` | All green; one new test per category-A fix proving the derived render AND the honest-empty-state render |
| Types | `npx tsc --noEmit` | Clean |
| Lint | `npm run lint` | No new errors |
| E2E | `npx playwright test` | Existing 3 specs green; new specs below pass |

New tests (added under `frontend/e2e/` for Playwright or `frontend/src/` for
component-level vitest, as appropriate):

1. **`static-values-route-pass.spec.ts`** (Playwright) — load `/lab`, `/demo`,
   `/library`, `/paperbench`, `/unlock` with no projectId and no data;
   screenshot each into `docs/design/audit-2026-05-22-screenshots/`; assert no
   known leak markers (`91.4`, `492.3`, `"Reproduced With Caveats"`, and any
   other markers S1 flags) appear in the DOM. This is *also* the category-C
   containment check: with no dev affordance, no fixture-shaped value can be
   on screen.
2. **`script-panel-regression.test.tsx`** (vitest component test) — construct
   a `LiveDemoRunState` where `payload.summary.stage !== "complete"` but
   `benchmark.overallScore = 91.4`; render `<ScriptPanel run={...} />`; assert
   `"Pending"` appears and `"91.4"` / `"492.3"` / `"Reproduced With Caveats"`
   do not. Pins the historical bug at `script-panel.tsx:22-30`.

Screenshots commit to `docs/design/audit-2026-05-22-screenshots/` (directory
already exists from a prior pass; S4 decides retain-or-replace).

## Deliverable — one PR

- `docs/design/static-values-audit-2026-05-22.md` — findings doc:
  violations table (`file:line · old · new · dynamic source`), Backend Gaps
  section, Category-C containment evidence section, Test evidence section.
- Code fixes.
- New tests (vitest + Playwright).
- Screenshots.
- This design doc:
  `docs/superpowers/specs/2026-05-22-frontend-static-values-audit-design.md`.
- Branch: `feature/static-values-audit-2026-05-22` off `main`, pushed to
  `github.com/armaanamatya/openresearch` (never the `replix` fork).
- Commit trailer:
  `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`.

## Conventions

- Honest empty state beats fabricated value. No inventing data to fill backend
  gaps.
- Mirror the existing house style for empty states: `"—"` for
  numbers/percentages (per `formatPercent`); `"Pending — ..."` / `"Run halted"`
  / `"n/a"` for verdicts and slots (per `script-panel.tsx`).
- Stay strictly on the static-DATA hunt. Do not churn category-B chrome or
  restyle anything.
- Each fix gets a test before the fix.
- Stop at the existing uncommitted modifications in the working tree (lab-shell,
  progress-strip, telemetry-strip, library-filters, lib/pipeline/layout) —
  these are pre-existing WIP unrelated to this audit. Carry them forward; do
  not include them in the audit's commits unless the inventory flags one of
  their literals as category-A.

## Open risks

- **Inventory drift** — if S1's findings disagree with S2/S3's read. Mitigation:
  S1 produces a structured per-file table; S2/S3 work only from that table and
  escalate disputes back to the integrator.
- **Backend gaps masquerading as frontend violations** — a misleading value
  that's wrong because the backend never provided it correctly is NOT a
  frontend fix; document and move on. S1 must distinguish.
- **Playwright server-side dependency** — `/lab`, `/demo`, `/library`,
  `/paperbench` need `REPROLAB_BACKEND_URL` reachable for SSR. If unreachable,
  server fetches return null and routes render their honest empty states.
  Tests exercise that path explicitly.
- **The `/demo` route is unlock-gated** (`proxy.ts`). Playwright either
  bypasses the gate via test config or sets `REPROLAB_DEMO_SECRET=""` for the
  test run.

## Non-goals

- Backend changes (cost reporting, score reporting). Out of scope; documented
  as gaps.
- UI restyling, layout changes, copy rewrites that are not honesty-driven.
- Permanent lint/CI rules against future hardcoded values (would be a useful
  follow-up; not in this PR).
