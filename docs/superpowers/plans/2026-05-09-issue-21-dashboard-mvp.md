# Issue 21 Dashboard MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a safe, isolated MVP dashboard scaffold that can render topology, activity, citations, and delegation detail views from PRD-shaped payloads even though the upstream app and backend contracts are not yet present on `main`.

**Architecture:** Create a minimal Next.js + TypeScript frontend in this worktree, model the dashboard around typed event/task contracts extracted from the PRD, and derive all visible panels from a single normalized dashboard snapshot. Keep live transport integration out of scope for now by isolating fixtures, normalization logic, and view shells behind small interfaces that can be swapped for real SSE/WebSocket data later.

**Tech Stack:** Next.js 15, React 19, TypeScript 5, Vitest, PRD-derived fixtures

---

### Task 1: Scaffold The Frontend Workspace

**Files:**
- Create: `package.json`
- Create: `tsconfig.json`
- Create: `next.config.ts`
- Create: `next-env.d.ts`
- Create: `app/layout.tsx`
- Create: `app/page.tsx`
- Create: `app/globals.css`

- [ ] **Step 1: Write the failing build/test setup**

```json
{
  "scripts": {
    "build": "next build",
    "test": "vitest run"
  }
}
```

- [ ] **Step 2: Run verification to confirm the app does not exist yet**

Run: `npm test`
Expected: FAIL because `package.json` and test config are missing

- [ ] **Step 3: Add the minimal Next.js app shell**

```tsx
export default function Page() {
  return <main>Dashboard scaffold</main>;
}
```

- [ ] **Step 4: Run build to verify the scaffold compiles**

Run: `npm run build`
Expected: PASS with a rendered app shell

### Task 2: Define PRD-Derived Dashboard Contracts And Fixtures

**Files:**
- Create: `lib/dashboard/contracts.ts`
- Create: `lib/dashboard/fixtures.ts`
- Create: `lib/dashboard/normalize.ts`
- Test: `tests/dashboard/normalize.test.ts`

- [ ] **Step 1: Write failing normalization tests against the required payload shapes**

```ts
it("derives parent/child topology and citations from dashboard events", () => {
  expect(model.tasksById["task_042"].parentTaskId).toBe("task_017");
  expect(model.feed[0].citations.length).toBeGreaterThan(0);
});
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `npm test -- tests/dashboard/normalize.test.ts`
Expected: FAIL because contracts and normalizer are missing

- [ ] **Step 3: Implement typed contracts, sample PRD fixtures, and a normalizer**

```ts
export type DashboardEvent =
  | AgentLifecycleEvent
  | AgentReasoningStepEvent
  | SharedStateUpdatedEvent
  | ContextEnrichmentEvent;
```

- [ ] **Step 4: Run tests to verify the data model passes**

Run: `npm test -- tests/dashboard/normalize.test.ts`
Expected: PASS

### Task 3: Build The Dashboard View Shells

**Files:**
- Create: `components/dashboard/dashboard-shell.tsx`
- Create: `components/dashboard/topology-view.tsx`
- Create: `components/dashboard/activity-feed.tsx`
- Create: `components/dashboard/citation-panel.tsx`
- Create: `components/dashboard/task-detail-panel.tsx`
- Create: `components/dashboard/pipeline-overview.tsx`
- Modify: `app/page.tsx`
- Modify: `app/globals.css`

- [ ] **Step 1: Add a failing UI smoke test or compile target via the page composition**

```tsx
expect(screen.getByText("Agent topology")).toBeInTheDocument();
```

- [ ] **Step 2: Run test/build to verify the shell is not implemented**

Run: `npm run build`
Expected: FAIL or missing symbols for dashboard components

- [ ] **Step 3: Implement the shell components with normalized fixture data**

```tsx
<DashboardShell snapshot={sampleDashboardSnapshot} />
```

- [ ] **Step 4: Run build and tests to verify all panels compile together**

Run: `npm test && npm run build`
Expected: PASS

### Task 4: Document Dependency Gaps And Integration Boundaries

**Files:**
- Create: `docs/dashboard-contract-notes.md`
- Modify: `README.md`

- [ ] **Step 1: Write down the missing dependencies and safe integration seam**

```md
- `main` does not yet contain the backend event stream or upstream UI shell work.
- The dashboard consumes typed `DashboardSnapshot` fixtures today and can later accept SSE/WebSocket payloads through the same normalizer.
```

- [ ] **Step 2: Run a final verification sweep**

Run: `npm test && npm run build`
Expected: PASS

- [ ] **Step 3: Summarize changed files and remaining blockers for handoff**

Run: `git status --short`
Expected: shows only the new dashboard scaffold files for this issue
