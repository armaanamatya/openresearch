# Frontend Static-Values Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Audit `frontend/src` for hardcoded data the user sees rendered as if it were real, fix every category-A violation, document Backend Gaps, and prove with tests — all on `feature/static-values-audit-2026-05-22`, one PR.

**Architecture:** Subagent fan-out 3 — S1 inventory subagent reads everything and produces a findings doc; S2 and S3 fix partitioned, non-overlapping file sets in parallel using strict-TDD; S4 adds the regression + route-pass tests, runs full verification, screenshots, and finalises the findings doc. The integrator (this session) dispatches, reviews each subagent's structured report, and commits.

**Tech Stack:** Next.js 16 App Router, React 19, TypeScript, Tailwind, vitest, Playwright. Backend is FastAPI but is out of scope.

**Spec:** `docs/superpowers/specs/2026-05-22-frontend-static-values-audit-design.md`.

**Pre-flight invariants** (confirmed before this plan is dispatched):
- On branch `feature/static-values-audit-2026-05-22`, off `main`.
- Spec doc committed at `785fa11` (HEAD~ at plan-creation time).
- Working tree has one travelling backend-test mod (`tests/rlm/test_leaf_scorer.py`) from the user's prior session — NOT to be staged or touched.
- Remote: `origin https://github.com/armaanamatya/openresearch.git`. Never push to a `replix` fork (the remote is not configured here; safe by absence).

---

## Task 1: Inventory subagent (S1) — produce the findings doc

**Files:**
- Create: `docs/design/static-values-audit-2026-05-22.md` (the findings doc)
- Read-only over the entire audit surface:
  - `frontend/src/app/{lab,demo,library,paperbench,unlock}/page.tsx`, `frontend/src/app/page.tsx`, `frontend/src/app/layout.tsx`
  - `frontend/src/components/lab/*.{tsx,ts}` (24 files, excluding `*.test.*`)
  - `frontend/src/components/library/*.{tsx,ts}` (3)
  - `frontend/src/components/paperbench/*.tsx` (1)
  - `frontend/src/components/demo/*.tsx` (1)
  - `frontend/src/hooks/use-run.ts` (and any other hooks/* file)
  - `frontend/src/lib/{demo,pipeline,runs,models,paperbench}/*.{ts,tsx}`
  - `frontend/src/app/api/**/*.ts`

- [ ] **Step 1: Dispatch S1 inventory subagent**

  Use `Agent` tool with `subagent_type: "general-purpose"`. The subagent is read-only for code (it only writes the findings doc). Prompt:

  ```
  You are S1 — the inventory subagent for the frontend static-values audit.
  Read-only on frontend/src; write ONE output file (the findings doc).

  ## Mission
  Survey EVERY file under frontend/src/ for hardcoded VALUES the user sees in the
  rendered UI, per this categorisation rubric:

  - (A) VIOLATIONS — fix. A literal representing data the user sees about a
    run / experiment / their account that is baked into the code instead of
    coming from the event stream, the API, or props/state. INCLUDES
    `value ?? "descriptive default"` fallbacks where the default reads as if
    it were the data.
  - (B) FINE — leave. Layout/config constants, design tokens, static UI copy
    (button labels, section headings, field names, empty-state text).
  - (C) DEV FIXTURES — acceptable, must be contained. Recorded fixtures
    reachable only from tests or an explicit dev flag.

  ## Audit surface (read every file in this list)
  - frontend/src/app/page.tsx
  - frontend/src/app/layout.tsx
  - frontend/src/app/{lab,demo,library,paperbench,unlock}/page.tsx
  - frontend/src/components/lab/*.{tsx,ts}  (exclude *.test.* and *.css)
  - frontend/src/components/library/*.{tsx,ts}  (exclude *.test.* and *.css)
  - frontend/src/components/paperbench/*.tsx
  - frontend/src/components/demo/*.tsx
  - frontend/src/hooks/*.ts (and *.tsx)
  - frontend/src/lib/{demo,pipeline,runs,models,paperbench}/**/*.{ts,tsx}
    (exclude *.test.* and __fixtures__/* — but DO note fixture files in the
    Category-C section)
  - frontend/src/app/api/**/*.ts  (light pass; check for hardcoded JSON
    responses or values, but server-side files generally shouldn't render
    user-facing values)

  ## For EACH candidate literal you find, decide A | B | C.

  Format your output for category-A entries as a row in the violations table:
    file:line | literal (verbatim, max 80 chars) | dynamic source (e.g.
      "run.benchmark?.benchmarkName via DemoBenchmarkSummary in lib/demo/
      demo-run-types.ts:47") | proposed fix (the honest empty state).

  Honest-empty house style (use these for the "proposed fix" column):
    - Numbers/percentages → '—' (mirrors formatPercent in paperbench-client.tsx)
    - Verdict/title slots → 'Pending — ...' / 'n/a' / 'Run halted' (mirrors
      script-panel.tsx's pipelineComplete gating)
    - Lists → render nothing (no rows / empty state with descriptive copy)
    - Names → '—' or contextual "Untitled paper" / "No benchmark named yet"

  ## Backend Gaps
  When a category-A value's dynamic source does not yet exist on the backend
  (or exists but is broken — e.g. the M4 cost_usd under-reporting from
  docs/design/phase3-5-review-findings.md), list it in the Backend Gaps
  section instead of the violations table. Include:
    file:line | literal | what dynamic source is needed | suggested follow-up.
  Backend Gaps are NOT fixed in this PR — the frontend still renders an
  honest empty state, but no backend change is proposed.

  ## Category-C containment evidence
  For every fixture-like construct you find (anything matching `*fixture*`,
  `__fixtures__/`, `FIXTURE_*`, `mock*`, `placeholder*`, sample data
  literals), verify behaviorally: is there any non-test, non-dev-flag code
  path that can render this value? Document the verdict (CONTAINED with
  evidence, or LEAK with the path).

  KNOWN PRIOR-BUG MARKERS to specifically search for:
    - "91.4" and "492.3" and "Reproduced With Caveats" from
      script-panel.tsx:22-30 (a historical real-render of fixture preset
      values when a run failed at Gate 2).

  ## Output file
  Write your findings to docs/design/static-values-audit-2026-05-22.md with
  these sections in this order:

    # Frontend Static-Values Audit — 2026-05-22

    ## Summary
    Counts of A / B-borderline / C. Top 5 violations by impact.

    ## Violations table (category A)
    Markdown table: file:line | literal | dynamic source | proposed fix.
    Group by partition (lab/, library/+paperbench/+demo/, hooks+lib, api).

    ## Backend Gaps
    Markdown table: file:line | literal | dynamic source needed | suggested
    follow-up.

    ## Category-C containment evidence
    Per fixture-like construct: file | usage | verdict (CONTAINED | LEAK).

    ## Out-of-scope notes
    Things that looked suspicious but are category-B chrome, with one-line
    reasons. Helps the reviewer trust the audit.

    ## Methodology
    Brief: what files were read, what regexes/searches drove the hunt.

  ## Constraints
  - DO NOT modify any file in frontend/src.
  - Only write the findings doc (and optionally a scratch file under
    /tmp/, which you should clean up).
  - Be specific. "Looks hardcoded" is not useful; quote file:line and the
    literal verbatim.
  - When in doubt between A and B, lean B but note as "borderline (B)" in
    the out-of-scope section with reasoning — the integrator can re-classify.
  - Background spec: docs/superpowers/specs/2026-05-22-frontend-static-values-
    audit-design.md.

  ## Return value
  Print a structured report to stdout (this comes back to the integrator):
    - Path to the findings doc (should be docs/design/static-values-audit-
      2026-05-22.md).
    - Top-line counts: violations found per partition, Backend Gaps count,
      Category-C verdicts (contained vs leak).
    - The 3 most consequential violations, paraphrased.
    - Any audit-surface files that errored on read (none expected, but
      report if so).
  ```

- [ ] **Step 2: Read S1's output and sanity-check**

  Read `docs/design/static-values-audit-2026-05-22.md`. Verify:
  - Violations table is non-empty (the audit isn't trivially clean).
  - Backend Gaps section exists (even if empty — confirm absence vs presence).
  - Category-C section verdicts.
  - Each violation row has all four fields.

  If S1 disagrees with the spec on a known finding (e.g. the script-panel `'PaperBench-style final benchmark'` fallback) — re-prompt S1 or escalate to the integrator.

- [ ] **Step 3: Commit the findings doc**

  ```bash
  git add docs/design/static-values-audit-2026-05-22.md
  git commit \
    -m "docs: frontend static-values audit — inventory (S1)" \
    -m "Findings doc produced by S1 inventory subagent. Lists every category-A violation in frontend/src with the dynamic source it should consume and the proposed honest empty state. Includes Backend Gaps section and Category-C containment evidence." \
    -m "Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
  ```

  Expected: one commit added on `feature/static-values-audit-2026-05-22`.

---

## Task 2: Fix lab partition (S2)

**Files:** Determined by S1's findings, scoped to:
- `frontend/src/components/lab/*.{tsx,ts}` (everything except `*.test.*` and `*.css`)
- `frontend/src/hooks/*.{ts,tsx}`
- `frontend/src/lib/demo/*.ts`

**Subagent dispatch:**

- [ ] **Step 1: Dispatch S2 fix subagent**

  Use `Agent` tool with `subagent_type: "general-purpose"`. Prompt:

  ```
  You are S2 — the fix subagent for the LAB partition of the frontend
  static-values audit.

  ## Inventory
  Read docs/design/static-values-audit-2026-05-22.md. Your scope is ONLY
  violations whose file path is under:
    - frontend/src/components/lab/
    - frontend/src/hooks/
    - frontend/src/lib/demo/
  Ignore everything else (S3 owns the other partition).

  ## Discipline — strict TDD
  For each violation in your scope, in order:

    1. Write a failing vitest unit test in the matching test file (e.g.
       components/lab/script-panel.tsx → components/lab/script-panel.test.tsx
       — create if missing). The test asserts BOTH:
         (i)  The value renders from the dynamic source when present.
         (ii) The honest empty state renders when the source is absent.

    2. Run only that test:
       cd frontend && npx vitest run path/to/test.test.tsx --reporter=verbose
       Expected: FAIL (preferably on the empty-state assertion, since the
       current code renders the hardcoded value instead of empty state).

    3. Apply the fix per the "proposed fix" column in the inventory.
       Replace the hardcoded literal with the dynamic source + honest
       empty state.

    4. Re-run the test — expect PASS.

    5. Run a wider check:
       cd frontend && npx tsc --noEmit
       cd frontend && npx vitest run src/components/lab src/hooks src/lib/demo
       Expected: all green.

    6. Commit:
       git add <test-file> <source-file>
       git commit \
         -m "fix(frontend): <component> — derive <value> from <source>" \
         -m "<one-sentence rationale tying back to the audit doc>" \
         -m "Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"

  ## Honest-empty conventions (mirror house style — do NOT invent)
  - Numbers / percentages → '—'  (matches formatPercent in paperbench-client.tsx:27-30)
  - Verdict/title slots → 'Pending — ...' / 'n/a' / 'Run halted' (matches script-panel.tsx)
  - Lists → empty (no rows; section can render an honest empty-state row)
  - Names → '—' or contextual "Untitled paper" / "No benchmark named yet"

  ## Do NOT
  - Restyle, rename, or refactor anything not directly required by a violation.
  - Touch category-B chrome.
  - Modify files outside your partition.
  - Touch backend code (Backend Gaps stay documented, not fixed).
  - Touch the working-tree modification at tests/rlm/test_leaf_scorer.py
    (user's pre-existing WIP, NOT in your scope).
  - Silently agree if you think a finding is wrong — mark the violation
    "skipped: <reason>" in your final report and proceed.

  ## Special-case: starting from zero violations
  If the inventory lists zero violations in your partition, write a brief
  report saying so and exit (no fixes, no commits). Verify by re-reading the
  inventory's partition rows.

  ## Return value
  Print a structured report:
    - Violations fixed: count + table of (file:line, before, after, test file).
    - Violations skipped: count + table with reasons.
    - Tests added: paths.
    - Commits made: hashes (output of `git log --oneline -N`).
    - Final verification: outputs of `npx tsc --noEmit` and
      `npx vitest run src/components/lab src/hooks src/lib/demo` (last
      ~20 lines each).
  ```

- [ ] **Step 2: Review S2's report**

  Verify:
  - Each fix has a corresponding test file path that exists.
  - No "skipped" violations unless the reason is sound.
  - All commits are on `feature/static-values-audit-2026-05-22` (run `git log --oneline -N` where N = fixes + 1).
  - Final vitest + tsc outputs in the report are green.

- [ ] **Step 3: Integrator-side partition verification**

  ```bash
  cd frontend && npx vitest run src/components/lab src/hooks src/lib/demo
  cd frontend && npx tsc --noEmit
  ```
  Expected: all green. If red, investigate before dispatching Task 4.

---

## Task 3: Fix non-lab partition (S3) — IN PARALLEL with Task 2

**Files:**
- `frontend/src/components/library/*.{tsx,ts}`
- `frontend/src/components/paperbench/*.tsx`
- `frontend/src/components/demo/*.tsx`
- `frontend/src/lib/pipeline/*.{ts,tsx}` (exclude `__fixtures__/`)
- `frontend/src/lib/runs/*.ts`
- `frontend/src/lib/models/*.ts`
- `frontend/src/lib/paperbench/*.ts`
- `frontend/src/app/api/**/*.ts`

**Subagent dispatch:** Tasks 2 and 3 are dispatched in the SAME response (two `Agent` tool calls in one turn). The partitions have no overlapping files, so commits do not conflict.

- [ ] **Step 1: Dispatch S3 fix subagent**

  Use `Agent` tool with `subagent_type: "general-purpose"`. Prompt:

  ```
  You are S3 — the fix subagent for the NON-LAB partition of the frontend
  static-values audit.

  ## Inventory
  Read docs/design/static-values-audit-2026-05-22.md. Your scope is ONLY
  violations whose file path is under:
    - frontend/src/components/library/
    - frontend/src/components/paperbench/
    - frontend/src/components/demo/
    - frontend/src/lib/pipeline/   (exclude __fixtures__/)
    - frontend/src/lib/runs/
    - frontend/src/lib/models/
    - frontend/src/lib/paperbench/
    - frontend/src/app/api/
  Ignore everything else (S2 owns the lab partition).

  ## Discipline — strict TDD
  For each violation in your scope, in order:

    1. Write a failing vitest unit test in the matching test file. For
       components, prefer a colocated *.test.tsx. For lib/* modules, prefer
       *.test.ts in the same directory. For app/api/* routes, prefer
       *.route.test.ts in the same directory.

       The test asserts BOTH:
         (i)  The value renders from the dynamic source when present.
         (ii) The honest empty state renders when the source is absent.

    2. Run only that test:
       cd frontend && npx vitest run path/to/test --reporter=verbose
       Expected: FAIL.

    3. Apply the fix per the "proposed fix" column in the inventory.

    4. Re-run the test — expect PASS.

    5. Run a wider check:
       cd frontend && npx tsc --noEmit
       cd frontend && npx vitest run src/components/library \
         src/components/paperbench src/components/demo \
         src/lib/pipeline src/lib/runs src/lib/models src/lib/paperbench \
         src/app/api
       Expected: all green.

    6. Commit:
       git add <test-file> <source-file>
       git commit \
         -m "fix(frontend): <component> — derive <value> from <source>" \
         -m "<one-sentence rationale tying back to the audit doc>" \
         -m "Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"

  ## Honest-empty conventions (mirror house style — do NOT invent)
  - Numbers / percentages → '—'  (matches formatPercent in paperbench-client.tsx:27-30)
  - Verdict/title slots → 'Pending — ...' / 'n/a' (matches script-panel.tsx)
  - Lists → empty (no rows; section can render an honest empty-state row)
  - Names → '—' or contextual "Untitled paper" / "No benchmark named yet"

  ## Do NOT
  - Restyle, rename, or refactor anything not directly required by a violation.
  - Touch category-B chrome.
  - Modify files outside your partition (especially nothing under
    frontend/src/components/lab/, frontend/src/hooks/, frontend/src/lib/demo/).
  - Touch backend code (Backend Gaps stay documented, not fixed).
  - Touch the working-tree modification at tests/rlm/test_leaf_scorer.py.
  - Silently agree if you think a finding is wrong — mark the violation
    "skipped: <reason>" in your final report and proceed.

  ## Special-case: starting from zero violations
  If the inventory lists zero violations in your partition, write a brief
  report saying so and exit (no fixes, no commits). Verify by re-reading
  the inventory's partition rows.

  ## Return value
  Print a structured report:
    - Violations fixed: count + table of (file:line, before, after, test file).
    - Violations skipped: count + table with reasons.
    - Tests added: paths.
    - Commits made: hashes.
    - Final verification: outputs of `npx tsc --noEmit` and the partition-
      scoped vitest (last ~20 lines each).
  ```

- [ ] **Step 2: Review S3's report** — same checks as Task 2 Step 2 over the non-lab partition.

- [ ] **Step 3: Integrator-side partition verification**

  ```bash
  cd frontend && npx vitest run src/components/library \
    src/components/paperbench src/components/demo \
    src/lib/pipeline src/lib/runs src/lib/models src/lib/paperbench \
    src/app/api
  cd frontend && npx tsc --noEmit
  ```
  Expected: all green.

---

## Task 4: Regression test + route-pass + screenshots (S4, in-session)

This task is executed in the main session (no subagent) because it's small, the file shapes are known, and the test code is fully specified below.

**Files:**
- Create: `frontend/src/components/lab/script-panel.regression.test.tsx`
- Create: `frontend/e2e/static-values-route-pass.spec.ts`
- Update: `docs/design/static-values-audit-2026-05-22.md` (add Test evidence section)
- Output: `docs/design/audit-2026-05-22-screenshots/{lab,demo,library,paperbench,unlock}.png`

- [ ] **Step 1: Write `script-panel.regression.test.tsx`**

  Create `frontend/src/components/lab/script-panel.regression.test.tsx`:

  ```tsx
  import { describe, expect, it } from "vitest";
  import { render } from "@testing-library/react";

  import { ScriptPanel } from "./script-panel";
  import type { LiveDemoRunState } from "@/lib/demo/demo-run-types";

  // Pins script-panel.tsx:22-30 — a real run that FAILED at Gate 2 once
  // displayed "Reproduced With Caveats / 91.4% / 492.3" because the
  // workspace_fixture template's preset values weren't overwritten. The fix
  // gates the success display on payload.summary.stage === "complete". This
  // test asserts the gate stays in place against an inverse-shaped fixture
  // (complete-looking benchmark + non-complete stage).
  describe("ScriptPanel — fixture-leak regression", () => {
    it("does not render '91.4' / '492.3' / 'Reproduced With Caveats' when stage !== 'complete'", () => {
      const run = {
        projectId: "test-fixture-leak",
        outputDir: "/tmp/test-fixture-leak",
        runMode: "sdk",
        status: "failed",
        sourcePdf: null,
        startedAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        log: "",
        telemetry: [],
        benchmark: {
          benchmarkName: "PaperBench-style",
          paperbenchTaskId: "task-1",
          overallScore: 91.4,
          targetMetric: "accuracy",
          targetValue: 500.0,
          reproducedValue: 492.3,
          deltaValue: -7.7,
          verdict: "verified_with_caveats",
          reportPath: "",
          comparisonPath: "",
          logPath: "",
        },
        payload: {
          summary: { stage: "gate_2_failed" },
        },
      } as unknown as LiveDemoRunState;

      const { container } = render(<ScriptPanel run={run} />);
      const text = container.textContent ?? "";

      // The verdict slot must show the gated state, not the success label.
      expect(text).toContain("Pending");

      // The hard guarantee — none of the historical leak markers may appear.
      expect(text).not.toContain("91.4");
      expect(text).not.toContain("492.3");
      expect(text).not.toContain("Reproduced With Caveats");
    });
  });
  ```

- [ ] **Step 2: Run the regression test**

  ```bash
  cd frontend && npx vitest run src/components/lab/script-panel.regression.test.tsx --reporter=verbose
  ```
  Expected: 1 test passing. If it fails, the leak gate has regressed — investigate before continuing.

- [ ] **Step 3: Write `static-values-route-pass.spec.ts`**

  Create `frontend/e2e/static-values-route-pass.spec.ts`:

  ```typescript
  import path from "node:path";
  import { test, expect } from "@playwright/test";

  // Audit deliverable: load every public page with no projectId and no
  // backend data, assert no fixture-shaped leak markers render, and
  // screenshot each. Doubles as the category-C containment check — with no
  // dev affordance, no fixture-shaped value can reach the DOM.

  const LEAK_MARKERS = ["91.4", "492.3", "Reproduced With Caveats"];
  const ROUTES = ["/lab", "/demo", "/library", "/paperbench", "/unlock"];
  const SCREENSHOT_DIR = path.join(
    __dirname,
    "..",
    "..",
    "docs",
    "design",
    "audit-2026-05-22-screenshots"
  );

  for (const route of ROUTES) {
    test(`route ${route} — honest empty state, no leak markers`, async ({ page }) => {
      await page.goto(route, { waitUntil: "networkidle" });
      const body = (await page.textContent("body")) ?? "";
      for (const marker of LEAK_MARKERS) {
        expect(
          body,
          `route ${route} rendered the leak marker "${marker}"`
        ).not.toContain(marker);
      }
      const slug = route.replace(/^\//, "").replace(/\//g, "-") || "root";
      await page.screenshot({
        path: path.join(SCREENSHOT_DIR, `${slug}.png`),
        fullPage: true,
      });
    });
  }
  ```

- [ ] **Step 4: Run Playwright**

  Playwright requires a running app. Start the dev server in a separate background process:

  ```bash
  # In a background terminal (or background_run via Bash):
  cd frontend && npm run dev
  # Wait for "ready" output (port 3000).
  ```

  Then run the Playwright spec:

  ```bash
  cd frontend && npx playwright test static-values-route-pass.spec.ts --reporter=list
  ```

  Expected: 5 PASS. If `/demo` is unlock-gated (because `REPROLAB_DEMO_SECRET` is set), set `REPROLAB_DEMO_SECRET=""` in the dev-server env for the test run.

  If the dev server can't start (missing `REPROLAB_BACKEND_URL`), the server-side fetches in `app/{lab,library,paperbench}/page.tsx` return null and the pages render their honest empty state — the assertion still holds. Document this in the test evidence.

- [ ] **Step 5: Run the full verification gate**

  ```bash
  cd frontend && npx tsc --noEmit
  cd frontend && npm run lint
  cd frontend && npx vitest run
  cd frontend && npx playwright test
  ```

  Expected: all four green. Capture the last ~10 lines of each into a scratch buffer for the audit doc's Test evidence section.

- [ ] **Step 6: Update the findings doc with Test evidence**

  Append a section to `docs/design/static-values-audit-2026-05-22.md`:

  ```markdown
  ## Test evidence

  | Layer | Command | Result |
  |---|---|---|
  | Types | `cd frontend && npx tsc --noEmit` | clean |
  | Lint  | `cd frontend && npm run lint` | clean (no new errors) |
  | Unit  | `cd frontend && npx vitest run` | N passed, 0 failed |
  | E2E   | `cd frontend && npx playwright test` | M passed (5 from static-values-route-pass) |

  New tests added in this audit:
  - `frontend/src/components/lab/script-panel.regression.test.tsx` — pins the
    historical "91.4 / 492.3 / Reproduced With Caveats" fixture-leak.
  - `frontend/e2e/static-values-route-pass.spec.ts` — loads every public
    route, asserts no leak markers, screenshots each.

  Plus one test per category-A fix (see commit log).

  Screenshots committed at `docs/design/audit-2026-05-22-screenshots/`:
  - lab.png, demo.png, library.png, paperbench.png, unlock.png
  ```

- [ ] **Step 7: Commit tests + audit doc update + screenshots**

  ```bash
  git add frontend/src/components/lab/script-panel.regression.test.tsx \
          frontend/e2e/static-values-route-pass.spec.ts \
          docs/design/static-values-audit-2026-05-22.md \
          docs/design/audit-2026-05-22-screenshots/
  git commit \
    -m "test(frontend): static-values audit — regression + route-pass + screenshots (S4)" \
    -m "Adds script-panel.regression.test.tsx pinning the historical '91.4 / 492.3 / Reproduced With Caveats' fixture-leak from script-panel.tsx:22-30, and static-values-route-pass.spec.ts loading every public route with no data, asserting no leak markers, and screenshotting. Updates audit doc with test evidence." \
    -m "Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
  ```

---

## Task 5: Final verification + PR handoff

- [ ] **Step 1: Re-run full verification from a fresh dependency install**

  ```bash
  cd frontend && npm ci
  cd frontend && npx tsc --noEmit
  cd frontend && npm run lint
  cd frontend && npx vitest run
  cd frontend && npx playwright test
  ```

  Expected: all four green. If any step regresses, halt and investigate.

- [ ] **Step 2: Inspect the branch's commit sequence**

  ```bash
  git log --oneline main..feature/static-values-audit-2026-05-22
  git diff main --stat
  ```

  Expected commit sequence (paraphrased):
  1. `docs: design spec for frontend static-values audit (2026-05-22)`
  2. `docs: frontend static-values audit — inventory (S1)`
  3-N. `fix(frontend): <component> — derive <value> from <source>` × violations fixed
  N+1. `test(frontend): static-values audit — regression + route-pass + screenshots (S4)`

  The diff should touch only `frontend/src/`, `frontend/e2e/`, `docs/design/`, and `docs/superpowers/`.

- [ ] **Step 3: Confirm no stray modifications**

  ```bash
  git status --short
  ```

  Expected: clean except the travelling `tests/rlm/test_leaf_scorer.py` (the user's pre-existing WIP) and any untracked docs that were already present at plan creation.

- [ ] **Step 4: Hand off to the user (do NOT push)**

  Print a summary report to the user:
  - Violations found / fixed (count, by partition).
  - Backend Gaps documented (count).
  - Category-C containment verdicts (count of CONTAINED, count of LEAK; LEAK count should be 0).
  - Tests added.
  - Screenshots in `docs/design/audit-2026-05-22-screenshots/`.
  - Commit hashes.
  - Suggested PR title: "Frontend static-values audit (2026-05-22) — every rendered value is now dynamically derived".
  - Suggested PR body skeleton with the audit doc summary inlined.

  Push only on explicit user instruction:
  ```bash
  git push -u origin feature/static-values-audit-2026-05-22
  ```

---

## Open risks (for the integrator to watch)

1. **Inventory drift.** S1 says X is a violation; S2/S3 disagrees mid-fix. Mitigation: S2/S3 are instructed to mark "skipped" and surface — the integrator decides.
2. **Empty partition.** S1 produces zero violations in one partition (S2 or S3). That subagent reports "no fixes" and exits cleanly; the plan handles this.
3. **Playwright server dependency.** `/lab`, `/demo`, `/library`, `/paperbench` need server-side fetches. If `REPROLAB_BACKEND_URL` is unreachable, those fetches return null and the route renders an honest empty state — the leak-marker assertion still holds; document the dev-server config in test evidence.
4. **`/demo` unlock gate.** If `REPROLAB_DEMO_SECRET` is set, `/demo` returns 401. For the test run set it to empty.
5. **Test fixture type drift.** `LiveDemoRunState` shape may have changed since plan-write time. If `script-panel.regression.test.tsx` fails to compile, adjust the fixture (cast through `unknown as LiveDemoRunState` already handles minor field shifts).
6. **Out-of-scope `tests/rlm/test_leaf_scorer.py` modification.** Pre-existing user WIP that travelled into this branch. Never `git add` it. Final `git status --short` should show it as still modified, untracked or unchanged from session start.
