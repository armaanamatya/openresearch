# RLM Lab Go-Live (Frontend Cutover) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the RLM lab *the* lab — every UI run is an `rlm` run, the lab always renders `<RlmLab>` driven by a real run's live event stream, and the old 14-stage pipeline UI is deleted from the frontend.

**Architecture:** A frontend-only cutover. The `/api/demo` proxy and `useRun` are switched to start `rlm` runs; the `useRun` event cap is raised so a long run's history is not truncated; `lab-shell.tsx`'s `WorkflowView` collapses to render `<RlmLab>`; the now-unreferenced old-UI component/lib/hook files are deleted; one hardcoded threshold is moved to a config constant; the live path is verified end-to-end.

**Tech Stack:** Next.js 16, React, TypeScript, vitest, Playwright. Run all `npm` commands from `frontend/`.

**Scope boundary:**
- **This plan = the frontend cutover.** The **backend** half — emitting `candidate_proposed` / `candidate_outcome` / `rubric_score` — is specified by the committed handoff doc `docs/superpowers/specs/2026-05-21-rlm-phase4-backend-events-handoff.md`; that doc *is* its plan and it is executed in coordination with the cofounder (Phase-3 ownership). It is **not** re-derived here. See the design spec `docs/superpowers/specs/2026-05-22-rlm-lab-go-live-design.md` §4.
- The exploration-canvas **tree-framing fix** lands separately on PR #70 — not in this plan.
- After this plan, a real `--mode rlm` run renders live through every region; the exploration tree shows the trunk and goes fully branch-populated once the backend half lands.

**Pre-existing test baseline:** `npm run lint` currently reports 4 errors (`progress-strip.tsx`, `telemetry-strip.tsx`, `library-filters.tsx`) — the first two are deleted by this plan (Task 4), so lint ends cleaner. `npm test` is 122 passing on the branch base.

---

### Task 1: Switch the `/api/demo` proxy to `rlm`-only

**Files:**
- Modify: `frontend/src/app/api/demo/route.ts`

`route.ts` is a Next.js route handler (an integration boundary, not unit-tested in this repo) — it is verified by `npx tsc --noEmit` and by the Task 7 end-to-end check. Today `toRunMode()` accepts only `"offline"`/`"sdk"` and silently drops `"rlm"`; the `POST` body defaults `mode` to `"offline"`. The lab is now RLM-only, so the proxy always uses `rlm`.

- [ ] **Step 1: Make `toRunMode` return `"rlm"`**

Replace the `toRunMode` function (lines 30-33) with:

```ts
// The lab is RLM-only — every run the UI starts is an `rlm` run. The `mode`
// query param is no longer read; sdk/offline remain reachable via the CLI only.
function toRunMode(): "rlm" {
  return "rlm";
}
```

- [ ] **Step 2: Update the two `toRunMode(request)` call sites**

`toRunMode` no longer takes `request`. In `backendQuery` (line 75) change `const mode = toRunMode(request);` to `const mode = toRunMode();`. In the `POST` JSON body (line 171) change `mode: toRunMode(request) ?? "offline",` to `mode: toRunMode(),`.

- [ ] **Step 3: Verify types**

Run: `npx tsc --noEmit`
Expected: clean (no output). `mode` is now the literal `"rlm"`; `DemoRunMode` still includes `"rlm"` so the assignment is valid.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/app/api/demo/route.ts
git commit -m "feat(rlm-ui): /api/demo proxy starts rlm runs only

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Switch `useRun` to `rlm` runs, raise the event cap, relocate `DashboardLiveEvent`

**Files:**
- Modify: `frontend/src/hooks/use-run.ts`
- Modify: `frontend/src/components/lab/lab-shell.test.tsx` (the run-start URL contract)
- Modify: any other importer of `DashboardLiveEvent` that Step 1's grep surfaces (likely `frontend/src/lib/events/rlm-events.ts`)

`useRun`'s three start functions hardcode `mode: "sdk"`. `MAX_DASHBOARD_EVENTS = 200` is a sliding window that truncates a long `rlm` run's event history (the `useRlmRun` reducer needs the whole stream). `lab-shell.test.tsx` asserts the produced run-start URL (the `mode=sdk` contract — see the comment at `use-run.ts:321-322`). Separately: `use-run.ts:7` imports the `DashboardLiveEvent` *type* from `@/components/lab/agent-timeline-rail` — a file Task 4 deletes — so the type must move to a kept module first.

- [ ] **Step 1: Relocate the `DashboardLiveEvent` type into `use-run.ts`**

`use-run.ts:7` is `import type { DashboardLiveEvent } from "@/components/lab/agent-timeline-rail";` — `agent-timeline-rail.tsx` is deleted in Task 4. Open `agent-timeline-rail.tsx`, copy the `DashboardLiveEvent` type definition **verbatim**, and paste it into `use-run.ts` as an exported type just below the imports:

```ts
// The generic shape of a dashboard SSE event. Relocated here from the retired
// agent-timeline-rail.tsx — useRun owns the dashboardEvents stream, so the type
// lives with it. (Copy the exact definition from agent-timeline-rail.tsx.)
export type DashboardLiveEvent = /* exact definition from agent-timeline-rail.tsx */;
```

Delete the `import type { DashboardLiveEvent } …` line (line 7). Then run `git grep -n "DashboardLiveEvent" -- frontend/src` — for every importer other than `use-run.ts` itself and the doomed `agent-timeline-rail.tsx` (notably `frontend/src/lib/events/rlm-events.ts` if `isRlmEvent` references the type), repoint its import to `import type { DashboardLiveEvent } from "@/hooks/use-run";`. Run: `npx tsc --noEmit` — Expected: clean.

- [ ] **Step 2: Update the failing test first**

In `frontend/src/components/lab/lab-shell.test.tsx`, find the assertion(s) on the run-start request URL/body that expect `mode=sdk` (the fixture-run / uploaded-run start path). Change the expected `mode` from `sdk` to `rlm`. Run: `npx vitest run src/components/lab/lab-shell.test.tsx` — Expected: the run-start test now FAILS (code still sends `sdk`).

- [ ] **Step 3: Send `mode: "rlm"` from the three start functions**

In `use-run.ts`: `startFixtureRun` — change `mode: "sdk"` to `mode: "rlm"` in the `URLSearchParams` (line 324). `startUploadedRun` — change `formData.set("mode", "sdk")` to `formData.set("mode", "rlm")` (line 355). `startArxivRun` — change `mode: "sdk"` to `mode: "rlm"` in the JSON body (line 397).

- [ ] **Step 4: Raise the event cap**

In `use-run.ts` line 11, replace:

```ts
const MAX_DASHBOARD_EVENTS = 200;
```

with:

```ts
// An rlm run emits one repl_iteration + several primitive/candidate/rubric
// events per root-loop iteration — well over 200 across a full run. The
// useRlmRun reducer folds the whole stream, so the in-memory history must
// not be truncated. 20000 covers a long run with headroom; the backend's
// dashboard_events.jsonl is the durable record on reconnect either way.
const MAX_DASHBOARD_EVENTS = 20000;
```

- [ ] **Step 5: Run the test + checks to verify they pass**

Run: `npx vitest run src/components/lab/lab-shell.test.tsx` — Expected: PASS.
Run: `npx vitest run` — Expected: full suite green.
Run: `npx tsc --noEmit` — Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/hooks/use-run.ts frontend/src/components/lab/lab-shell.test.tsx frontend/src/lib/events/rlm-events.ts
git commit -m "feat(rlm-ui): useRun starts rlm runs; raise the event cap; relocate DashboardLiveEvent

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Collapse `lab-shell.tsx` to RLM-only

**Files:**
- Modify: `frontend/src/components/lab/lab-shell.tsx`
- Modify: `frontend/src/components/lab/lab-shell.test.tsx`

`WorkflowView` currently branches: `runMode === "rlm"` → `<RlmLab>`, else the 14-stage canvas. The 14-stage path and every helper/sub-component that serves only it are deleted; `WorkflowView` collapses to render `<RlmLab>`. A run whose `runMode` is not `rlm` (an old pre-pivot run opened from the sidebar) gets an honest placeholder, not a broken empty lab.

- [ ] **Step 1: Update the test first**

In `lab-shell.test.tsx`: (a) remove the `stateMapForRun` import from `./lab-shell` (the re-export is deleted in Step 2) and any test that exercises it / the old 14-stage rendering (the "still renders the 14-stage workflow for sdk/offline" test — delete it). (b) Keep the "renders RlmLab when runMode is rlm" test. (c) Add a test: a run with `runMode: "sdk"` renders the honest non-rlm placeholder (assert text `/retired pipeline|not.*rlm/i`) and does NOT render `<RlmLab>` (`queryByTestId("rlm-lab")` is null).
Run: `npx vitest run src/components/lab/lab-shell.test.tsx` — Expected: FAIL (placeholder not implemented; `stateMapForRun` import removed but old behavior unchanged).

- [ ] **Step 2: Replace `lab-shell.tsx` with the collapsed version**

Replace the entire file with:

```tsx
"use client";

import { useState, Suspense, type ReactNode } from "react";

import type { DemoModelChoice, LiveDemoRunState } from "@/lib/demo/demo-run-types";
import type { RecentRunSummary } from "@/lib/runs/server-list";
import type { ModelChoice } from "@/lib/models/server-fetch";
import { useSearchParams } from "next/navigation";
import { UploadView } from "./upload-view";
import { LabSidebar } from "./lab-sidebar";
import { CommandPalette } from "./command-palette";
import { ShortcutOverlay } from "./shortcut-overlay";
import { useRun } from "@/hooks/use-run";
import { useCommandPalette } from "@/hooks/use-command-palette";
import { useShortcutOverlay } from "@/hooks/use-shortcut-overlay";
import { PresentationModeProvider, type PresentationMode } from "@/lib/presentation-mode";
import { readUserPrefs, writeUserPref } from "@/lib/user-prefs";
import { RlmLab } from "./rlm/rlm-lab";
import { isRlmEvent } from "@/lib/events/rlm-events";
import { replayFixture } from "./rlm/replay";

import "./lab-shell.css";

type LabShellProps = {
  initialRun?: LiveDemoRunState | null;
  initialRecents?: RecentRunSummary[];
  initialModels?: ModelChoice[];
  presentationMode?: PresentationMode;
};

// Dev/test-only fixture metadata for the ?rlmFixture=1 replay path. A real run
// gets its title/meta/projectId from the run object (sourceLabel / sourceNote /
// projectId); this constant is the fixture's own stand-in for that.
const FIXTURE_RUN_META = {
  projectId: "prj_fixture",
  paperTitle: "Attention is all you need",
  paperMeta: "Vaswani et al. · fixture replay",
};

// The lab is RLM-only. A live run renders <RlmLab>; an older run that predates
// the RLM orchestrator (runMode !== "rlm", e.g. opened from Recent) cannot be
// rendered by the RLM lab — show an honest notice rather than an empty shell.
function RunView({
  run,
  dashboardEvents,
}: {
  run: LiveDemoRunState;
  dashboardEvents: ReturnType<typeof useRun>["dashboardEvents"];
}) {
  if (run.runMode !== "rlm") {
    return (
      <div className="card" style={{ padding: 24 }} data-testid="non-rlm-notice">
        <div className="eyebrow">Run</div>
        <p style={{ marginTop: 8 }}>
          This run was produced by the retired 14-stage pipeline. The RLM lab
          renders <code>rlm</code>-mode runs only — start a new run to watch it live.
        </p>
      </div>
    );
  }
  const rlmEvents = dashboardEvents.filter(isRlmEvent);
  return (
    <RlmLab
      events={rlmEvents}
      runMeta={{
        projectId: run.projectId,
        paperTitle: run.sourceLabel ?? "Untitled paper",
        paperMeta: run.sourceNote ?? "",
      }}
    />
  );
}

// Dev/test-only: ?rlmFixture=1 renders the fixture-driven RlmLab regardless of
// any live run. useSearchParams requires a Suspense boundary (App Router rule).
function RlmFixtureContent({ children }: { children: ReactNode }) {
  const searchParams = useSearchParams();
  if (searchParams?.get("rlmFixture") === "1") {
    return <RlmLab events={replayFixture("instant")} runMeta={FIXTURE_RUN_META} />;
  }
  return <>{children}</>;
}

export function LabShell({
  initialRun = null,
  initialRecents = [],
  initialModels = [],
  presentationMode = "internal",
}: LabShellProps) {
  const [arxiv, setArxiv] = useState("");
  const [over, setOver] = useState(false);
  const [model, setModel] = useState<DemoModelChoice>(() => readUserPrefs().model ?? "sonnet");
  const {
    run,
    busy,
    error,
    dashboardEvents,
    startFixtureRun,
    startUploadedRun,
    startArxivRun,
    resetToUpload: resetRun,
  } = useRun(initialRun);

  const resetToUpload = () => {
    setArxiv("");
    setOver(false);
    resetRun();
  };

  const palette = useCommandPalette();
  const shortcuts = useShortcutOverlay();

  const main = (
    <main className="content">
      <Suspense fallback={null}>
        <RlmFixtureContent>
          {run ? (
            <RunView run={run} dashboardEvents={dashboardEvents} />
          ) : (
            <UploadView
              arxiv={arxiv}
              busy={busy}
              error={error}
              model={model}
              models={initialModels}
              onArxivChange={setArxiv}
              onArxivSubmit={() =>
                arxiv.trim().length > 0
                  ? void startArxivRun(arxiv, model)
                  : void startFixtureRun(model)
              }
              onFileSelected={(file) => void startUploadedRun(file, model)}
              onModelChange={(value) => {
                setModel(value);
                writeUserPref("model", value);
              }}
              over={over}
              setOver={setOver}
            />
          )}
        </RlmFixtureContent>
      </Suspense>
    </main>
  );

  return (
    <div className="reproLab">
      <PresentationModeProvider mode={presentationMode}>
        <div className="layout">
          <LabSidebar active="lab" onBrandClick={resetToUpload} recents={initialRecents} />
          {main}
        </div>
        <CommandPalette
          open={palette.open}
          setOpen={palette.setOpen}
          recents={initialRecents}
          currentRun={run}
        />
        <ShortcutOverlay open={shortcuts.open} setOpen={shortcuts.setOpen} />
      </PresentationModeProvider>
    </div>
  );
}
```

Note what is removed: the `WorkflowView` 14-stage body, `RunOverview`, `Stat`, `RightPanel`, `FailurePanel`, `parseLogEntries`, `agentStartIndex`, `humanizeStage`, `telemetryForSelectedNode`, `failedNodeIdForRun`, `formatTime`, `LOG_LINE_RE`, `COMPLETED_LINE_RE`, the `stateMapForRun` re-export, the `TopologyProvider`/`useTopology` wiring, the `onClear`/`onResume`/`resumeRun`/`clearRun` plumbing, and the `initialTopology` prop. If `clearRun`/`resumeRun` removal breaks `useRun`'s exported surface usage elsewhere, leave those functions in `useRun` (unused is fine) — do not modify `use-run.ts` in this task.

- [ ] **Step 3: Check the `/lab` page passes no removed prop**

Run: `npx tsc --noEmit`. If `frontend/src/app/lab/page.tsx` (or wherever `<LabShell>` is mounted) passes `initialTopology`, remove that prop from the call site. Re-run `npx tsc --noEmit` — Expected: clean.

- [ ] **Step 4: Run the tests**

Run: `npx vitest run src/components/lab/lab-shell.test.tsx` — Expected: PASS.
Run: `npx vitest run` — Expected: green except failures that are purely deleted-component tests (those files are removed in Task 4).
Run: `npm run lint` — Expected: no NEW errors in `lab-shell.tsx` (old imports are gone).

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/lab/lab-shell.tsx frontend/src/components/lab/lab-shell.test.tsx
# include frontend/src/app/lab/page.tsx only if Step 3 changed it
git commit -m "feat(rlm-ui): collapse the lab to RLM-only — WorkflowView renders RlmLab

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Delete the old lab UI component files

**Files — DELETE:**
```
frontend/src/components/lab/progress-strip.tsx
frontend/src/components/lab/gate-chips.tsx
frontend/src/components/lab/node-card.tsx
frontend/src/components/lab/lab-canvas.tsx
frontend/src/components/lab/pan-wrap.tsx
frontend/src/components/lab/telemetry-strip.tsx
frontend/src/components/lab/agent-timeline-rail.tsx
frontend/src/components/lab/floating-agent-window.tsx
frontend/src/components/lab/agent-info-panel.tsx
frontend/src/components/lab/hermes-audit-panel.tsx
frontend/src/components/lab/script-panel.tsx
frontend/src/components/lab/resizable-split.tsx
frontend/src/components/lab/node-config.ts
frontend/src/components/lab/agent-info-helpers.ts
frontend/src/components/lab/failure-summary.ts
frontend/src/components/lab/failure-summary.test.ts
frontend/src/components/demo/demo-overlay.tsx
```
plus any `*.module.css` whose only importer is a file in that list, and any `*.test.tsx` whose only subject is a deleted component.

These are all rendered only by the old 14-stage path, which Task 3 removed. **`use-pan.ts` is KEPT** — `pan-wrap.tsx` is deleted but the `usePan` hook it used lives in `frontend/src/hooks/use-pan.ts` and the RLM `exploration-canvas.tsx` imports it. `status.tsx`, `icons.tsx`, `shared-helpers.ts` are KEPT (used by kept components / `paperbench-client.tsx` / `use-run.ts`).

- [ ] **Step 1: Confirm the files are unreferenced**

For each file above, run e.g. `git grep -n "node-card\|NodeCard" -- frontend/src ':!frontend/src/components/lab/node-card.tsx'` — confirm no remaining importer outside the delete-set. Resolve any straggler import before deleting.

- [ ] **Step 2: Delete the files**

`git rm` each file in the list (and the CSS-modules / test files identified above).

- [ ] **Step 3: Verify the suite, types, and lint are green**

Run: `npx tsc --noEmit` — Expected: clean (no dangling imports).
Run: `npx vitest run` — Expected: full suite green (the deleted components' tests are gone).
Run: `npm run lint` — Expected: the `progress-strip.tsx` / `telemetry-strip.tsx` pre-existing errors are gone; no new errors.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor(rlm-ui): delete the old 14-stage lab UI components

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Delete the old pipeline topology lib and hooks

**Files — DELETE:**
```
frontend/src/lib/pipeline/layout.ts
frontend/src/lib/pipeline/layout.test.ts
frontend/src/lib/pipeline/topology.ts
frontend/src/lib/pipeline/topology-helpers.ts
frontend/src/lib/pipeline/topology-context.tsx
frontend/src/lib/pipeline/server-fetch.ts
frontend/src/hooks/use-topology.ts
frontend/src/hooks/use-canvas-keyboard-nav.ts
```
plus any `frontend/src/lib/pipeline/__fixtures__/*` and `frontend/src/lib/pipeline/*.test.ts` whose only consumers are in this list.

After Task 3 + Task 4, the only consumers of the pipeline-topology layer (`LabCanvas`, `GateChips`, `NodeCard`, `useTopology`, the old `WorkflowView`) are gone.

- [ ] **Step 1: Confirm unreferenced**

`git grep -n "lib/pipeline\|use-topology\|use-canvas-keyboard-nav\|useTopology\|topology-context\|PipelineTopology\|LaidOutNode" -- frontend/src ':!frontend/src/lib/pipeline' ':!frontend/src/hooks/use-topology.ts' ':!frontend/src/hooks/use-canvas-keyboard-nav.ts'` — Expected: no matches. Resolve any straggler (e.g. an unused `import type` left in `lab-shell.tsx` or `route.ts`) before deleting.

- [ ] **Step 2: Check `/lab` (and `/demo`) route data-loading**

If `frontend/src/app/lab/page.tsx` (server component) calls the deleted `server-fetch.ts` to SSR-warm the topology, remove that call and the `initialTopology` prop it fed. Run `npx tsc --noEmit` to surface every such reference.

- [ ] **Step 3: Delete the files**

`git rm` each file in the list (and the verified `__fixtures__`/test files).

- [ ] **Step 4: Verify green**

Run: `npx tsc --noEmit` — clean.
Run: `npx vitest run` — full suite green.
Run: `npm run lint` — no new errors.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor(rlm-ui): delete the old pipeline-topology lib and hooks

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Move the `0.35` degraded threshold to a named config constant

**Files:**
- Create: `frontend/src/components/lab/rlm/rlm-config.ts`
- Create: `frontend/src/components/lab/rlm/rlm-config.test.ts`
- Modify: `frontend/src/components/lab/rlm/report-rail.tsx`

`report-rail.tsx` hardcodes `0.35` (the degraded-run cap, twice) — a policy value, not a layout constant. Promote it to a named, documented constant.

- [ ] **Step 1: Write the failing test**

`rlm-config.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { DEGRADED_SCORE_CAP } from "./rlm-config";

describe("rlm-config", () => {
  it("exposes the degraded-score cap as a documented constant", () => {
    expect(DEGRADED_SCORE_CAP).toBe(0.35);
  });
});
```

Run: `npx vitest run src/components/lab/rlm/rlm-config.test.ts` — Expected: FAIL (module missing).

- [ ] **Step 2: Create the config module**

`rlm-config.ts`:

```ts
// Tunable RLM-lab policy constants — values the UI uses to derive displayed
// state, kept out of component bodies so they are named and reviewable.

/**
 * Rubric scores at or below this are flagged as a possibly-degraded run (a run
 * that produced no measured metrics is capped here by the backend's rubric
 * verifier — the I7 cap). The UI shows a hedged "may be a degraded run" note.
 */
export const DEGRADED_SCORE_CAP = 0.35;
```

- [ ] **Step 3: Use the constant in `report-rail.tsx`**

In `report-rail.tsx`, add `import { DEGRADED_SCORE_CAP } from "./rlm-config";` and replace both `0.35` literals (the `isDegraded` comparison and any copy/aria reference) with `DEGRADED_SCORE_CAP`. Keep the user-facing note text ("≤ 0.35 …") as-is — it is display copy, not the policy value.

- [ ] **Step 4: Verify**

Run: `npx vitest run src/components/lab/rlm/rlm-config.test.ts src/components/lab/rlm/report-rail.test.tsx` — Expected: PASS.
Run: `npx tsc --noEmit` — clean. `npm run lint` — no new errors.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/lab/rlm/rlm-config.ts frontend/src/components/lab/rlm/rlm-config.test.ts frontend/src/components/lab/rlm/report-rail.tsx
git commit -m "refactor(rlm-ui): move the degraded-score cap to rlm-config

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Verify the live path end-to-end

**Files:**
- Modify: `frontend/e2e/rlm-lab.spec.ts`

Confirm the cutover holds: the lab is RLM-only, and a real `--mode rlm` run renders. The Playwright e2e keeps `/lab?rlmFixture=1` as its deterministic path; a real run is the manual gate.

- [ ] **Step 1: Extend the e2e — the lab is RLM-only**

In `frontend/e2e/rlm-lab.spec.ts`, add a test that loads `/lab?rlmFixture=1` and asserts the old UI is gone: `await expect(page.locator(".progress-strip, [data-testid='gate-chips']")).toHaveCount(0)` (or an equivalent selector for an old-UI element), and `await expect(page.getByTestId("rlm-lab")).toBeVisible()`.

- [ ] **Step 2: Run the e2e**

Read `frontend/playwright.config.ts`; start a dev server on its port if there is no `webServer` block; run `npx playwright test e2e/rlm-lab.spec.ts` — Expected: PASS.

- [ ] **Step 3: Full suite + checks**

Run: `npx vitest run` — green. `npx tsc --noEmit` — clean. `npm run lint` — Expected: the only remaining errors, if any, are in `library-filters.tsx` (the separate `/library` page, untouched by this plan).

- [ ] **Step 4: Manual real-run smoke (gate)**

Start a real run: `python -m backend.cli reproduce 2512.24601 --mode rlm` (or start one from the lab UI), open `/lab?projectId=<id>` (no `rlmFixture`), and confirm `<RlmLab>` renders it live — header, rubric strip, REPL-state rail, exploration-tree trunk, report rail, primitive history all populate from the real event stream. (Candidate branches appear once the backend half — the handoff doc — lands; their absence here is expected, not a bug.) Record the observed behavior in the PR description.

- [ ] **Step 5: Commit**

```bash
git add frontend/e2e/rlm-lab.spec.ts
git commit -m "test(rlm-ui): e2e asserts the lab is RLM-only

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Backend half — separate, cofounder-coordinated

The three SSE events (`candidate_proposed` / `candidate_outcome` / `rubric_score`) that make a real run's exploration tree grow candidate branches are **not** in this plan. They are specified, task-by-task with field mappings and a checklist, in the committed handoff doc:

`docs/superpowers/specs/2026-05-21-rlm-phase4-backend-events-handoff.md`

That doc is the backend plan. It touches `run.py` / `sse_bridge.py` / `system_prompt.py` / `context.py` / `binding.py` / `primitives.py` / `schemas.py` — the cofounder's Phase-3 territory. **The critical-path coordination ask:** `candidate_outcome` uses Option B — a new `record_candidate_outcome` primitive plus a **root-system-prompt change**. Raise that with the cofounder before the backend work starts. Once those events flow, the frontend built by this plan renders the full tree with **no further frontend change**.

## Self-review

- **Spec coverage** — design spec §5 (frontend cutover): §5.1 delete old UI → Tasks 3-5; §5.2 `rlm`-only run-start → Tasks 1-2; §5.3 the cap → Task 2; §5.4 static cleanup → Task 6 (the `0.35` threshold; `FIXTURE_RUN_META` is retained-and-documented in the Task 3 `lab-shell.tsx`); §5.5 verify → Task 7. §4 (backend) → the handoff doc, noted above. The framing fix is explicitly out (lands on #70).
- **No placeholders** — every code step shows complete code or an exact command; deletion tasks list exact paths and a grep-confirm step.
- **Type consistency** — `DemoRunMode` is deliberately *not* narrowed (the backend keeps `sdk`/`offline` for the CLI, and an old run-state object can still carry them — Task 3's `RunView` handles `runMode !== "rlm"`); `toRunMode()` returns the literal `"rlm"`, assignable to `DemoRunMode`; `useRun`'s exported surface is unchanged (Task 3 stops *calling* `clearRun`/`resumeRun` but does not remove them).
