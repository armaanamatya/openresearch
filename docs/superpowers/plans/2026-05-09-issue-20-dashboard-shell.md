# Issue #20 Dashboard Shell Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scaffold a standalone Next.js frontend with a stable Agent Lab dashboard shell and a mock-safe event adapter so UI work can continue before live backend events exist.

**Architecture:** Build a `frontend/` Next.js App Router app that owns a local TypeScript event contract mirroring the PRD event surface, then layer a mock adapter and dashboard state reducer on top of it. Keep the event boundary explicit so a future shared package from issue #8 can replace the local contract without changing the shell components.

**Tech Stack:** Next.js App Router, React, TypeScript, ESLint, Vitest, Testing Library

---

### Task 1: Scaffold the frontend workspace and test harness

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/tsconfig.json`
- Create: `frontend/next.config.ts`
- Create: `frontend/eslint.config.mjs`
- Create: `frontend/vitest.config.ts`
- Create: `frontend/next-env.d.ts`
- Create: `frontend/src/test/setup.ts`

- [ ] **Step 1: Write the failing shell sanity test**

```tsx
import { describe, expect, it } from "vitest";

describe("frontend workspace", () => {
  it("loads the dashboard shell test suite", () => {
    expect(true).toBe(true);
  });
});
```

- [ ] **Step 2: Run test to verify the workspace is not wired yet**

Run: `npm run test`
Expected: FAIL because the frontend package and test runner do not exist yet.

- [ ] **Step 3: Write the minimal frontend/tooling scaffold**

Create a minimal `frontend/package.json` with:
- `next`, `react`, `react-dom`
- `typescript`, `@types/node`, `@types/react`, `@types/react-dom`
- `eslint`, `eslint-config-next`
- `vitest`, `jsdom`, `@testing-library/react`, `@testing-library/jest-dom`

Scripts:
- `dev`, `build`, `start`
- `lint`
- `test`

Use App Router-compatible TypeScript and ESLint config, plus a Vitest setup file that loads `@testing-library/jest-dom`.

- [ ] **Step 4: Run tests to verify the harness works**

Run: `npm run test`
Expected: PASS for the placeholder test under the new frontend workspace.

### Task 2: Define the event contract and mock adapter with TDD

**Files:**
- Create: `frontend/src/lib/events/contract.ts`
- Create: `frontend/src/lib/events/mock-events.ts`
- Create: `frontend/src/lib/events/mock-event-adapter.ts`
- Create: `frontend/src/lib/events/mock-event-adapter.test.ts`

- [ ] **Step 1: Write the failing adapter tests**

```tsx
it("returns an initial snapshot and replays ordered events to subscribers", async () => {
  const adapter = createMockEventAdapter();
  const snapshot = adapter.getSnapshot();

  expect(snapshot.agents.length).toBeGreaterThan(0);

  const received: string[] = [];
  const unsubscribe = adapter.subscribe((event) => {
    received.push(event.event);
  });

  await adapter.flush();

  expect(received).toContain("agent_started");
  expect(received).toContain("agent_reasoning_step");
  unsubscribe();
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm run test -- src/lib/events/mock-event-adapter.test.ts`
Expected: FAIL because the contract and adapter do not exist.

- [ ] **Step 3: Write the minimal event contract and adapter**

Model the PRD event surface with:
- discriminated union event types including `agent_started`, `agent_completed`, `agent_failed`, `agent_reasoning_step`, `rlm_query_executed`, `semantic_search_executed`, `shared_state_updated`, `verification_gate_result`, `approval_requested`, `approval_resolved`, and `context_enrichment`
- shared metadata types for agents, citations, progress stages, feed entries, and dashboard snapshot state
- `DashboardEventAdapter` interface with `getSnapshot()`, `subscribe(listener)`, and mock-only `flush()` support
- deterministic mock fixtures that let UI code replay a believable timeline

Keep the contract isolated in one module with a comment explaining it is the temporary frontend-local mirror until issue #8 lands.

- [ ] **Step 4: Run adapter tests to verify they pass**

Run: `npm run test -- src/lib/events/mock-event-adapter.test.ts`
Expected: PASS

### Task 3: Build the dashboard state model and shell against the adapter

**Files:**
- Create: `frontend/src/features/dashboard/dashboard-state.ts`
- Create: `frontend/src/features/dashboard/use-dashboard-state.ts`
- Create: `frontend/src/features/dashboard/dashboard-shell.tsx`
- Create: `frontend/src/features/dashboard/dashboard-shell.test.tsx`
- Create: `frontend/src/app/layout.tsx`
- Create: `frontend/src/app/page.tsx`
- Create: `frontend/src/app/globals.css`

- [ ] **Step 1: Write the failing shell test**

```tsx
it("renders the agent lab shell sections from mock state", async () => {
  render(<DashboardShell />);

  expect(screen.getByText(/Agent Lab/i)).toBeInTheDocument();
  expect(screen.getByText(/Topology/i)).toBeInTheDocument();
  expect(screen.getByText(/Reasoning Stream/i)).toBeInTheDocument();
  expect(screen.getByText(/Messages/i)).toBeInTheDocument();
  expect(screen.getByText(/Citations/i)).toBeInTheDocument();
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm run test -- src/features/dashboard/dashboard-shell.test.tsx`
Expected: FAIL because the state model and shell do not exist.

- [ ] **Step 3: Implement the minimal shell**

Build:
- a pure dashboard state reducer that derives visible cards, feed rows, approvals, and progress from the event stream
- a client hook that seeds from `getSnapshot()` and subscribes to adapter events
- an App Router page that renders an intentional dashboard layout with sections for topology, reasoning stream, inter-agent messages, citations, progress, and data panels
- styling in `globals.css` using CSS variables and a clear visual direction that feels like an operations lab, not boilerplate admin chrome

Keep components simple and local to the dashboard feature unless a boundary is already obvious.

- [ ] **Step 4: Run targeted shell tests**

Run: `npm run test -- src/features/dashboard/dashboard-shell.test.tsx`
Expected: PASS

### Task 4: Verify the scaffold end-to-end and document the current boundary

**Files:**
- Modify: `README.md`
- Modify: `docs/setup-guide.md`
- Modify: `docs/superpowers/plans/2026-05-09-issue-20-dashboard-shell.md`

- [ ] **Step 1: Update docs for the new frontend workspace**

Document:
- how to install frontend deps
- how to run the dashboard locally
- that mock events back the shell until a backend/shared contract replaces the local adapter boundary

- [ ] **Step 2: Run full verification**

Run: `npm run test`
Expected: PASS

Run: `npm run lint`
Expected: PASS

Run: `npm run build`
Expected: PASS

- [ ] **Step 3: Mark completed plan steps**

Update this plan file so each completed checkbox reflects the final execution state.
