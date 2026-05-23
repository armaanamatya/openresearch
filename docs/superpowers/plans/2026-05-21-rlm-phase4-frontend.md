# RLM Phase 4 Frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the RLM lab UI — a live, branching exploration-tree canvas wrapped in a lab-notebook shell — driven by a recorded events fixture, coexisting with the existing 14-stage pipeline UI.

**Architecture:** A pure reducer (`fold` / `useRlmRun`) folds the RLM SSE event stream into one `RlmRunState`. `<RlmLab>` renders it as the v3 layout — a dominant `<ExplorationCanvas>` tree flanked by a REPL-state rail and a report rail, under a header + rubric strip. `WorkflowView` branches on `run.runMode === "rlm"`. Fixture-first: built and tested against a hand-authored events fixture; **no backend changes**.

**Tech Stack:** Next.js 16, React 19, TypeScript, CSS Modules, vitest + React Testing Library, Playwright. Design tokens from `frontend/src/styles/tokens.css`.

---

## Before you start (read once)

- **Design spec — read it first:** `docs/superpowers/specs/2026-05-21-rlm-phase4-frontend-design.md`. Every task cites spec sections (`spec §N`); the spec is the source of truth for schemas, the data model, component contracts, and visual rules. This plan is the *task breakdown* — it does not repeat the spec.
- **Working directory:** all commands run from `frontend/` unless stated. Paths in this plan are repo-root-relative.
- **Per-task green gate:** every task ends with `npm run lint`, `npx tsc --noEmit`, and `npm test` all green. A task is not done until they pass.
- **TDD:** failing test → watch it fail → minimal implementation → watch it pass → commit. Never write implementation before its test.
- **Commits:** one per task (some tasks commit twice — test, then implementation). Conventional-commit subject; every commit message ends with the trailer:
  `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`
- **Coexistence rule:** do not modify the 14-stage pipeline UI, `useRun`'s existing behavior, the `/api/demo/*` proxy, or anything under `backend/`. The only existing file this plan modifies is `lab-shell.tsx` (one `runMode` branch, Task 15).
- **Honesty rule (spec §14):** the UI shows only what events provide. No fabricated values.

## File structure (what gets created)

All new frontend code lives under these paths:

- `frontend/src/lib/events/rlm-events.ts` — the 8 RLM event TypeScript types + union + guard.
- `frontend/src/hooks/use-rlm-run.ts` — `RlmRunState`, the pure `fold`, the `useRlmRun` hook.
- `frontend/src/components/lab/rlm/layout-tree.ts` — pure tree-layout function.
- `frontend/src/components/lab/rlm/__fixtures__/rlm-run.fixture.ts` — the hand-authored events fixture.
- `frontend/src/components/lab/rlm/replay.ts` — the fixture replay harness.
- `frontend/src/components/lab/rlm/rlm-lab.tsx` (+ `.module.css`) — the shell.
- `frontend/src/components/lab/rlm/rlm-header.tsx` (+ `.module.css`)
- `frontend/src/components/lab/rlm/rubric-strip.tsx` (+ `.module.css`)
- `frontend/src/components/lab/rlm/repl-state-rail.tsx` (+ `.module.css`)
- `frontend/src/components/lab/rlm/report-rail.tsx` (+ `.module.css`)
- `frontend/src/components/lab/rlm/primitive-history-bar.tsx` (+ `.module.css`)
- `frontend/src/components/lab/rlm/tree-node.tsx` (+ `.module.css`)
- `frontend/src/components/lab/rlm/tree-edges.tsx`
- `frontend/src/components/lab/rlm/node-detail-popup.tsx` (+ `.module.css`)
- `frontend/src/components/lab/rlm/exploration-canvas.tsx` (+ `.module.css`)
- `*.test.ts` / `*.test.tsx` colocated with each module above.
- `frontend/e2e/rlm-lab.spec.ts` — the Playwright e2e.

Modified: `frontend/src/components/lab/lab-shell.tsx` (Task 15 — one `runMode` branch).
New doc: `docs/superpowers/specs/2026-05-21-rlm-phase4-backend-events-handoff.md` (Task 18).

---

## Task 1: RLM event types

**Context:** spec §4 (the 8 event schemas). Pure types — the foundation every other task imports.

**Files:**
- Create: `frontend/src/lib/events/rlm-events.ts`
- Test: `frontend/src/lib/events/rlm-events.test.ts`

- [ ] **Step 1: Write the failing test**

```ts
// frontend/src/lib/events/rlm-events.test.ts
import { describe, it, expect } from "vitest";
import { isRlmEvent, RLM_EVENT_TYPES } from "./rlm-events";
import type { RlmDashboardEvent } from "./rlm-events";

describe("isRlmEvent", () => {
  it("accepts every RLM event type", () => {
    for (const event of RLM_EVENT_TYPES) {
      expect(isRlmEvent({ event, timestamp: "2026-05-21T00:00:00Z" })).toBe(true);
    }
  });
  it("rejects a non-RLM dashboard event", () => {
    expect(isRlmEvent({ event: "agent_started", timestamp: "x" })).toBe(false);
    expect(isRlmEvent({})).toBe(false);
    expect(isRlmEvent(null)).toBe(false);
  });
  it("narrows the union", () => {
    const e: unknown = { event: "run_complete", timestamp: "x", status: "completed",
      iterations: 3, rubric_score: 0.5, cost_usd: 1, final_report_path: null };
    if (isRlmEvent(e) && e.event === "run_complete") {
      expect(e.iterations).toBe(3); // type-checks only if the union is correct
    }
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run (from `frontend/`): `npx vitest run src/lib/events/rlm-events.test.ts`
Expected: FAIL — `rlm-events.ts` does not exist.

- [ ] **Step 3: Write the implementation**

Create `rlm-events.ts` with the 8 interfaces **exactly as spec §4.1 and §4.2 define them**: `ReplIterationEvent`, `PrimitiveCallEvent`, `SubRlmSpawnedEvent`, `SubRlmCompleteEvent`, `RunCompleteEvent`, `CandidateProposedEvent`, `CandidateOutcomeEvent`, `RubricScoreEvent`. Then:

```ts
export const RLM_EVENT_TYPES = [
  "repl_iteration", "primitive_call", "sub_rlm_spawned", "sub_rlm_complete",
  "run_complete", "candidate_proposed", "candidate_outcome", "rubric_score",
] as const;

export type RlmDashboardEvent =
  | ReplIterationEvent | PrimitiveCallEvent | SubRlmSpawnedEvent
  | SubRlmCompleteEvent | RunCompleteEvent | CandidateProposedEvent
  | CandidateOutcomeEvent | RubricScoreEvent;

export function isRlmEvent(value: unknown): value is RlmDashboardEvent {
  if (typeof value !== "object" || value === null) return false;
  const ev = (value as { event?: unknown }).event;
  return typeof ev === "string" && (RLM_EVENT_TYPES as readonly string[]).includes(ev);
}
```

Each interface's fields, types, and optionality must match spec §4 verbatim (e.g. `CandidateProposedEvent.parent_id` is optional `string`; `outcome` is the 6-value union).

- [ ] **Step 4: Run test + type-check to verify they pass**

Run: `npx vitest run src/lib/events/rlm-events.test.ts` → PASS
Run: `npx tsc --noEmit` → no errors
Run: `npm run lint` → clean

- [ ] **Step 5: Commit**

```bash
git add frontend/src/lib/events/rlm-events.ts frontend/src/lib/events/rlm-events.test.ts
git commit -m "feat(rlm-ui): RLM dashboard event types

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: The events fixture

**Context:** spec §10 (fixture scenarios). A hand-authored, schema-conformant `RlmDashboardEvent[]` representing one realistic ~14-iteration run. The fixture is the contract for every downstream test, so its completeness is itself tested.

**Files:**
- Create: `frontend/src/components/lab/rlm/__fixtures__/rlm-run.fixture.ts`
- Test: `frontend/src/components/lab/rlm/__fixtures__/rlm-run.fixture.test.ts`

- [ ] **Step 1: Write the failing test** — the scenario-coverage guard

```ts
import { describe, it, expect } from "vitest";
import { rlmRunFixture } from "./rlm-run.fixture";
import { isRlmEvent } from "../../../../lib/events/rlm-events";

describe("rlmRunFixture", () => {
  it("is a non-trivial event stream", () => {
    expect(rlmRunFixture.length).toBeGreaterThanOrEqual(30);
    expect(rlmRunFixture.every(isRlmEvent)).toBe(true);
  });
  it("covers every required scenario (spec §10)", () => {
    const byType = (t: string) => rlmRunFixture.filter((e) => e.event === t);
    // >= 2 propose_improvements rounds
    const rounds = new Set(byType("candidate_proposed").map((e: any) => e.round));
    expect(rounds.size).toBeGreaterThanOrEqual(2);
    // a round that promotes nothing
    const outcomes = byType("candidate_outcome") as any[];
    const proposed = byType("candidate_proposed") as any[];
    const roundOf = (id: string) => proposed.find((p) => p.candidate.id === id)?.round;
    const promotedRounds = new Set(
      outcomes.filter((o) => o.outcome === "promoted").map((o) => roundOf(o.candidate_id)));
    expect([...rounds].some((r) => !promotedRounds.has(r))).toBe(true);
    // an out-of-order proposed/outcome pair (an outcome before its proposed)
    const firstOutcomeIdx = rlmRunFixture.findIndex((e) => e.event === "candidate_outcome");
    const hasOutOfOrder = outcomes.some((o) => {
      const pIdx = rlmRunFixture.findIndex(
        (e) => e.event === "candidate_proposed" && (e as any).candidate.id === o.candidate_id);
      const oIdx = rlmRunFixture.indexOf(o);
      return oIdx < pIdx;
    });
    expect(hasOutOfOrder).toBe(true);
    // >= 2 declined in one round
    const declinedByRound: Record<number, number> = {};
    for (const o of outcomes.filter((o) => o.outcome === "declined")) {
      const r = roundOf(o.candidate_id); if (r != null) declinedByRound[r] = (declinedByRound[r] ?? 0) + 1;
    }
    expect(Object.values(declinedByRound).some((n) => n >= 2)).toBe(true);
    // sub-RLM pair
    expect(byType("sub_rlm_spawned").length).toBeGreaterThanOrEqual(1);
    expect(byType("sub_rlm_complete").length).toBeGreaterThanOrEqual(1);
    // a rubric_score with a failing area
    expect((byType("rubric_score") as any[]).some(
      (e) => e.areas.some((a: any) => a.status === "fail"))).toBe(true);
    // a terminal run_complete, exactly one, last
    expect(byType("run_complete").length).toBe(1);
    expect(rlmRunFixture[rlmRunFixture.length - 1].event).toBe("run_complete");
    void firstOutcomeIdx;
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/components/lab/rlm/__fixtures__/rlm-run.fixture.test.ts`
Expected: FAIL — `rlm-run.fixture.ts` does not exist.

- [ ] **Step 3: Author the fixture**

Create `rlm-run.fixture.ts` exporting `export const rlmRunFixture: RlmDashboardEvent[] = [ ... ]`. Author a realistic run, in stream order. Use the "Attention is all you need" scenario from the v3 mockup for content. It must, in order:
1. `repl_iteration` i1–i3 calling `understand_section` / `extract_hyperparameters` (comprehension), with `primitive_call` start/ok pairs and growing `code_blocks[].vars`.
2. `repl_iteration` i4–i6 calling `detect_environment` / `build_environment` (environment; one failed build attempt then ok).
3. `repl_iteration` i7–i9 calling `implement_baseline` / `run_experiment`, then a `rubric_score` (overall ~0.22, with areas, ≥1 `fail`) — the baseline.
4. Round 1: `repl_iteration` calling `propose_improvements`; then `candidate_proposed` ×5 with `round: 1` (titles: "learning-rate warmup", "attention dropout", "pre-norm placement", "label-smoothing sweep", + one more). Then `candidate_outcome` for each: 2 `promoted`, 1 `marginal`, 1 `failed`, 1 `declined` — **plus** a 2nd `declined` so round 1 has ≥2 declined. Interleave `rubric_score` events as candidates resolve (climbing).
5. **One out-of-order pair:** emit one candidate's `candidate_outcome` (e.g. `outcome: "running"`) *before* its `candidate_proposed` event.
6. Round 2 (`round: 2`): branches off a promoted round-1 candidate; `candidate_proposed` ×3, with a `sub_rlm_spawned` + `sub_rlm_complete` pair, and outcomes — make this round promote **nothing** (all `running` / `declined` / `marginal`, none `promoted`).
7. A final `rubric_score` (~0.53) and a terminal `run_complete` (`status: "completed"`, `rubric_score: 0.53`, counts).

Every event must conform to the Task 1 types. Keep `parent_id` **absent** on `candidate_proposed` (exercises the reducer's inference, Task 4).

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run src/components/lab/rlm/__fixtures__/rlm-run.fixture.test.ts` → PASS
Run: `npx tsc --noEmit` → no errors

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/lab/rlm/__fixtures__/
git commit -m "feat(rlm-ui): hand-authored RLM run events fixture

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: RlmRunState + the linear fold (iterations, variables, primitives, rubric, report)

**Context:** spec §5.1 (`RlmRunState`), §5.2 (the fold table). This task builds the state shape and the non-tree event folding. Task 4 adds the tree.

**Files:**
- Create: `frontend/src/hooks/use-rlm-run.ts`
- Test: `frontend/src/hooks/use-rlm-run.test.ts`

- [ ] **Step 1: Write the failing test**

```ts
import { describe, it, expect } from "vitest";
import { fold, INITIAL_RLM_STATE } from "./use-rlm-run";
import { rlmRunFixture } from "../components/lab/rlm/__fixtures__/rlm-run.fixture";

const reduce = (events = rlmRunFixture) => events.reduce(fold, INITIAL_RLM_STATE);

describe("fold — linear state", () => {
  it("starts empty", () => {
    expect(INITIAL_RLM_STATE.iterationCount).toBe(0);
    expect(INITIAL_RLM_STATE.status).toBe("queued");
    expect(INITIAL_RLM_STATE.report).toBeNull();
  });
  it("counts iterations from repl_iteration", () => {
    const s = reduce();
    expect(s.iterationCount).toBeGreaterThanOrEqual(13);
    expect(s.iterations.length).toBe(s.iterationCount);
    expect(s.currentIteration).toEqual(s.iterations[s.iterations.length - 1]);
  });
  it("builds the REPL variable manifest from code_blocks[].vars", () => {
    const s = reduce();
    expect(s.variables.paper_text).toMatchObject({ type: "str" });
    expect(typeof s.variables.paper_text.firstSeenIteration).toBe("number");
  });
  it("accumulates primitive calls", () => {
    const s = reduce();
    expect(s.primitiveCalls.length).toBeGreaterThan(0);
    expect(s.primitiveCalls[0]).toHaveProperty("primitive");
    expect(s.primitiveCalls[0]).toHaveProperty("status");
  });
  it("tracks the rubric series, baseline, current, target, areas", () => {
    const s = reduce();
    expect(s.rubric.baseline).toBeCloseTo(0.22, 1);
    expect(s.rubric.current).toBeCloseTo(0.53, 1);
    expect(s.rubric.target).toBeGreaterThan(0);
    expect(s.rubric.series.length).toBeGreaterThanOrEqual(2);
    expect(s.rubric.areas.length).toBeGreaterThan(0);
  });
  it("sets status + report on run_complete", () => {
    const s = reduce();
    expect(s.status).toBe("completed");
    expect(s.report).not.toBeNull();
    expect(s.report!.counts.iterations).toBe(s.iterationCount);
  });
  it("is pure — same input, same output", () => {
    expect(reduce()).toEqual(reduce());
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/hooks/use-rlm-run.test.ts`
Expected: FAIL — `use-rlm-run.ts` does not exist.

- [ ] **Step 3: Write the implementation**

Create `use-rlm-run.ts`. Define `RlmRunState` and its member types (`IterationView`, `RubricArea`, `PrimitiveCallView`, `SubRlmView`, `TreeNode`) **exactly per spec §5.1 and §6** — `TreeNode` is used now (declared) and populated in Task 4. `IterationView` mirrors the `repl_iteration` payload verbatim — `{ iteration, response, code_blocks, sub_calls, timing }`, snake_case preserved — so no field remapping is needed. Export `INITIAL_RLM_STATE` (status `"queued"`, empty collections, `report: null`, `tree: []`). Export a pure `fold(state: RlmRunState, event: RlmDashboardEvent): RlmRunState` that returns a new state (never mutates). Implement the spec §5.2 table for these events only: `repl_iteration` (bump `iterationCount`, push `IterationView` with `response` as root reasoning, merge `code_blocks[].vars` into `variables` recording `firstSeenIteration`, set `currentIteration`); `primitive_call` (append `PrimitiveCallView`); `rubric_score` (push `{iteration, score}` to `rubric.series`, set `rubric.current`/`areas`, set `rubric.baseline` if unset, set `rubric.target`); `run_complete` (set `status`, build `report` with `counts`). Leave `candidate_proposed` / `candidate_outcome` / `sub_rlm_*` as no-ops for now (Task 4). Unknown events: return state unchanged.

- [ ] **Step 4: Run test + checks to verify they pass**

Run: `npx vitest run src/hooks/use-rlm-run.test.ts` → PASS
Run: `npx tsc --noEmit` → clean · `npm run lint` → clean

- [ ] **Step 5: Commit**

```bash
git add frontend/src/hooks/use-rlm-run.ts frontend/src/hooks/use-rlm-run.test.ts
git commit -m "feat(rlm-ui): RlmRunState + linear event fold

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: The tree fold — candidate nodes, the frontier rule, sub-RLM, the useRlmRun hook

**Context:** spec §5.2, §5.3 (the frontier rule — pin it exactly), §6 (the node & tree model). Extends `fold` in the same file; adds the React hook.

**Files:**
- Modify: `frontend/src/hooks/use-rlm-run.ts`
- Modify: `frontend/src/hooks/use-rlm-run.test.ts` (add the tree tests)

- [ ] **Step 1: Write the failing tests**

```ts
describe("fold — the tree", () => {
  const reduce = () => rlmRunFixture.reduce(fold, INITIAL_RLM_STATE);
  it("has a paper root and a baseline node", () => {
    const s = reduce();
    expect(s.tree.find((n) => n.kind === "paper")).toBeDefined();
    expect(s.tree.find((n) => n.kind === "baseline")).toBeDefined();
  });
  it("creates a node per proposed candidate", () => {
    const s = reduce();
    const proposed = rlmRunFixture.filter((e) => e.event === "candidate_proposed");
    const candidateNodes = s.tree.filter((n) => n.kind === "candidate");
    expect(candidateNodes.length).toBe(proposed.length);
  });
  it("parents round-1 candidates on the baseline node (frontier default)", () => {
    const s = reduce();
    const baseline = s.tree.find((n) => n.kind === "baseline")!;
    const round1 = s.tree.filter((n) => n.kind === "candidate" && n.round === 1);
    expect(round1.every((n) => n.parentId === baseline.id)).toBe(true);
  });
  it("parents round-2 candidates on a promoted round-1 node", () => {
    const s = reduce();
    const round2 = s.tree.filter((n) => n.kind === "candidate" && n.round === 2);
    const promotedR1 = s.tree.filter(
      (n) => n.kind === "candidate" && n.round === 1 && n.outcome === "promoted");
    expect(round2.length).toBeGreaterThan(0);
    expect(round2.every((n) => promotedR1.some((p) => p.id === n.parentId))).toBe(true);
  });
  it("applies candidate_outcome regardless of event order", () => {
    const s = reduce();
    expect(s.tree.filter((n) => n.kind === "candidate").every((n) => n.outcome != null)).toBe(true);
  });
  it("collapses a round's declined candidates into one declined-group node", () => {
    const s = reduce();
    const groups = s.tree.filter((n) => n.kind === "declined-group");
    expect(groups.length).toBeGreaterThanOrEqual(1);
    expect(groups[0].declinedCount).toBeGreaterThanOrEqual(2);
  });
  it("adds a sub-RLM node", () => {
    const s = reduce();
    expect(s.tree.find((n) => n.kind === "subrlm")).toBeDefined();
    expect(s.subRlms.length).toBeGreaterThanOrEqual(1);
  });
  it("never produces an orphan (every non-paper node resolves to a parent)", () => {
    const s = reduce();
    const ids = new Set(s.tree.map((n) => n.id));
    for (const n of s.tree) {
      if (n.kind === "paper") continue;
      expect(n.parentId != null && ids.has(n.parentId)).toBe(true);
    }
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx vitest run src/hooks/use-rlm-run.test.ts`
Expected: FAIL — tree is empty / nodes undefined.

- [ ] **Step 3: Extend the fold + add the hook**

In `fold`: build trunk nodes (`paper`, `work`/phase nodes via the spec §6 phase derivation from `primitive_call` sequences, `baseline` at the first `rubric_score`); on `candidate_proposed` add a `candidate` `TreeNode` (`parentId` from `event.parent_id` if present, else the **frontier rule §5.3** — most-recent `promoted` node, else the previous fan's parent, else `baseline`; group a round's declined candidates into one synthesized `declined-group` node); on `candidate_outcome` set the node's `outcome` + `rubricDelta` (tolerate arriving before the matching `candidate_proposed` — buffer pending outcomes and apply on proposal); on `sub_rlm_spawned`/`_complete` add/update a `SubRlmView` + a `subrlm` `TreeNode`. Implement the frontier exactly as §5.3 states. Then add the hook:

```ts
import { useMemo } from "react";
export function useRlmRun(events: RlmDashboardEvent[]): RlmRunState {
  return useMemo(() => events.reduce(fold, INITIAL_RLM_STATE), [events]);
}
```

- [ ] **Step 4: Run tests + checks to verify they pass**

Run: `npx vitest run src/hooks/use-rlm-run.test.ts` → PASS (all, incl. Task 3's)
Run: `npx tsc --noEmit` → clean · `npm run lint` → clean

- [ ] **Step 5: Commit**

```bash
git add frontend/src/hooks/use-rlm-run.ts frontend/src/hooks/use-rlm-run.test.ts
git commit -m "feat(rlm-ui): tree fold, frontier rule, and the useRlmRun hook

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: layoutTree — the pure tree-layout function

**Context:** spec §8 (the layout pass). Pure: `TreeNode[]` → positioned nodes + edges. No React. The hardest pure function — keep it isolated and well-tested.

**Files:**
- Create: `frontend/src/components/lab/rlm/layout-tree.ts`
- Test: `frontend/src/components/lab/rlm/layout-tree.test.ts`

- [ ] **Step 1: Write the failing test**

```ts
import { describe, it, expect } from "vitest";
import { layoutTree } from "./layout-tree";
import type { TreeNode } from "../../../hooks/use-rlm-run";

const node = (id: string, parentId: string | null, kind: TreeNode["kind"] = "candidate"): TreeNode =>
  ({ id, parentId, kind, title: id } as TreeNode);

describe("layoutTree", () => {
  it("places the root at depth 0 and children to its right", () => {
    const { positioned } = layoutTree([node("p", null, "paper"), node("a", "p"), node("b", "p")]);
    const p = positioned.find((n) => n.id === "p")!;
    const a = positioned.find((n) => n.id === "a")!;
    expect(a.x).toBeGreaterThan(p.x);
  });
  it("stacks siblings vertically without overlap", () => {
    const { positioned } = layoutTree([node("p", null, "paper"), node("a", "p"), node("b", "p")]);
    const a = positioned.find((n) => n.id === "a")!;
    const b = positioned.find((n) => n.id === "b")!;
    expect(a.y).not.toBe(b.y);
  });
  it("emits one edge per parent->child relation", () => {
    const { edges } = layoutTree([node("p", null, "paper"), node("a", "p"), node("b", "a")]);
    expect(edges).toHaveLength(2);
    expect(edges.some((e) => e.from === "p" && e.to === "a")).toBe(true);
  });
  it("deepens along the longest branch", () => {
    const { positioned } = layoutTree([
      node("p", null, "paper"), node("a", "p"), node("a2", "a"), node("a3", "a2")]);
    const xs = ["p", "a", "a2", "a3"].map((id) => positioned.find((n) => n.id === id)!.x);
    expect(xs).toEqual([...xs].sort((m, n) => m - n));
  });
  it("handles an empty tree", () => {
    expect(layoutTree([])).toEqual({ positioned: [], edges: [] });
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/components/lab/rlm/layout-tree.test.ts` → FAIL (no module).

- [ ] **Step 3: Write the implementation**

Implement `layoutTree(nodes)`. Export `PositionedNode` (`TreeNode & { x: number; y: number }`) and `Edge` (`{ from: string; to: string; outcome?: string }`). Algorithm (spec §8): assign `x` by depth (depth × column-width); assign `y` by a post-order sibling-stacking pass (each leaf takes the next row; each parent centers on its children's `y` span). Emit one `Edge` per node with a `parentId`, carrying the child's `outcome` for edge coloring. Constants (`COLUMN_WIDTH`, `ROW_HEIGHT`) are module-level. Pure — no DOM, no React.

- [ ] **Step 4: Run test + checks**

Run: `npx vitest run src/components/lab/rlm/layout-tree.test.ts` → PASS · `npx tsc --noEmit` clean · `npm run lint` clean

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/lab/rlm/layout-tree.ts frontend/src/components/lab/rlm/layout-tree.test.ts
git commit -m "feat(rlm-ui): pure tree-layout function

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Tasks 6–10: presentational components (shared structure)

Tasks 6–10 build the five presentational components. They share this structure — for each: create `<name>.tsx` + `<name>.module.css` + `<name>.test.tsx`; the test renders the component with a fixture-derived `RlmRunState` (build it via `rlmRunFixture.reduce(fold, INITIAL_RLM_STATE)`) and asserts the listed elements; the component is a pure function of its props (no data fetching); it reads design tokens from `frontend/src/styles/tokens.css` and follows the spec §9 visual rules (flat, hairline, sentence case, outcome palette, `prefers-reduced-motion`); CSS lives in the `.module.css`. Each task: write test → fail → implement → pass → `tsc`+`lint`+`test` green → commit.

## Task 6: RlmHeader

**Context:** spec §7.2 (RlmHeader row), v3 mockup header band.

**Files:** Create `frontend/src/components/lab/rlm/rlm-header.tsx`, `.module.css`, `rlm-header.test.tsx`.

- [ ] **Step 1: Failing test**

```tsx
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { RlmHeader } from "./rlm-header";

describe("RlmHeader", () => {
  const props = {
    paperTitle: "Attention is all you need",
    paperMeta: "Vaswani et al. · NeurIPS 2017 · arXiv:1706.03762",
    projectId: "prj_8b78ac6368bad043",
    status: "running" as const,
    iterationCount: 13,
    costUsd: 18.4,
  };
  it("renders the paper title, project id, status, and iteration count", () => {
    render(<RlmHeader {...props} />);
    expect(screen.getByText("Attention is all you need")).toBeInTheDocument();
    expect(screen.getByText(/prj_8b78ac6368bad043/)).toBeInTheDocument();
    expect(screen.getByText(/running/i)).toBeInTheDocument();
    expect(screen.getByText(/13/)).toBeInTheDocument();
  });
  it("omits cost when costUsd is null", () => {
    render(<RlmHeader {...props} costUsd={null} />);
    expect(screen.queryByText(/\$/)).not.toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run → FAIL** (`npx vitest run src/components/lab/rlm/rlm-header.test.tsx`)
- [ ] **Step 3: Implement** `RlmHeader` — props `{ paperTitle, paperMeta, projectId, status, iterationCount, costUsd }`; renders the header band per the v3 design (title + meta line, project-id chip, a status pill — reuse `../status` if its API fits, else a local pill, iteration + cost). Honesty: when `costUsd` is null, render no cost element.
- [ ] **Step 4: Run test + `tsc` + `lint` → all green**
- [ ] **Step 5: Commit** — `git add` the three files; commit subject `feat(rlm-ui): RlmHeader component` + the standard trailer.

---

## Task 7: RubricStrip

**Context:** spec §7.2 (RubricStrip), §9; v3 rubric band — score, baseline→target bar, climb sparkline.

**Files:** Create `frontend/src/components/lab/rlm/rubric-strip.tsx`, `.module.css`, `rubric-strip.test.tsx`.

- [ ] **Step 1: Failing test**

```tsx
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { RubricStrip } from "./rubric-strip";

describe("RubricStrip", () => {
  const rubric = { current: 0.53, baseline: 0.22, target: 0.4,
    series: [{ iteration: 9, score: 0.22 }, { iteration: 13, score: 0.53 }], areas: [] };
  it("shows the current score and the target", () => {
    render(<RubricStrip rubric={rubric} />);
    expect(screen.getByText("0.53")).toBeInTheDocument();
    expect(screen.getByText(/0\.40/)).toBeInTheDocument();
  });
  it("shows the climb from baseline", () => {
    render(<RubricStrip rubric={rubric} />);
    expect(screen.getByText(/0\.22/)).toBeInTheDocument(); // baseline marker
  });
  it("renders one sparkline bar per series point", () => {
    const { container } = render(<RubricStrip rubric={rubric} />);
    expect(container.querySelectorAll("[data-spark-bar]")).toHaveLength(2);
  });
  it("handles a not-yet-scored run", () => {
    render(<RubricStrip rubric={{ current: null, baseline: null, target: 0.4, series: [], areas: [] }} />);
    expect(screen.getByText(/—|not scored/i)).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run → FAIL**
- [ ] **Step 3: Implement** `RubricStrip` — prop `{ rubric: RlmRunState["rubric"] }`; renders the score number, a baseline→fill→target progress bar, and a sparkline (`<span data-spark-bar>` per `series` point). `current === null` → render an em-dash / "not scored" placeholder (honesty §14). Outcome/forest-green accent from tokens.
- [ ] **Step 4: Run test + `tsc` + `lint` → green**
- [ ] **Step 5: Commit** — `feat(rlm-ui): RubricStrip component` + trailer.

---

## Task 8: ReplStateRail

**Context:** spec §7.2 (ReplStateRail), v3 left rail — variables with type/size, primitives list, collapsible.

**Files:** Create `frontend/src/components/lab/rlm/repl-state-rail.tsx`, `.module.css`, `repl-state-rail.test.tsx`.

- [ ] **Step 1: Failing test**

```tsx
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { fireEvent } from "@testing-library/react";
import { ReplStateRail } from "./repl-state-rail";

describe("ReplStateRail", () => {
  const props = {
    variables: {
      paper_text: { type: "str", size: 88000, firstSeenIteration: 1 },
      final_report: { type: "NoneType", size: 0, firstSeenIteration: 0 },
    },
    primitives: ["understand_section", "run_experiment"],
    collapsed: false,
    onToggle: () => {},
  };
  it("lists variables with their type", () => {
    render(<ReplStateRail {...props} />);
    expect(screen.getByText("paper_text")).toBeInTheDocument();
    expect(screen.getByText(/str/)).toBeInTheDocument();
  });
  it("lists the available primitives", () => {
    render(<ReplStateRail {...props} />);
    expect(screen.getByText(/understand_section/)).toBeInTheDocument();
  });
  it("calls onToggle when the collapse control is clicked", () => {
    let toggled = false;
    render(<ReplStateRail {...props} onToggle={() => { toggled = true; }} />);
    fireEvent.click(screen.getByRole("button", { name: /collapse|expand/i }));
    expect(toggled).toBe(true);
  });
});
```

- [ ] **Step 2: Run → FAIL**
- [ ] **Step 3: Implement** `ReplStateRail` — props `{ variables, primitives, collapsed, onToggle }`; renders the variable rows (name · type · human size; a variable with `firstSeenIteration === 0` or type `NoneType` renders dimmed = "not set"), an "N/M set" count, the primitives list, and a collapse `<button>`. When `collapsed`, render only an icon strip.
- [ ] **Step 4: Run test + `tsc` + `lint` → green**
- [ ] **Step 5: Commit** — `feat(rlm-ui): ReplStateRail component` + trailer.

---

## Task 9: ReportRail

**Context:** spec §7.2 (ReportRail), §14 (honesty — the 0.35 cap, cost). v3 right rail — verdict, stat grid, live rubric breakdown.

**Files:** Create `frontend/src/components/lab/rlm/report-rail.tsx`, `.module.css`, `report-rail.test.tsx`.

- [ ] **Step 1: Failing test**

```tsx
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { ReportRail } from "./report-rail";

const areas = [
  { area: "Architecture matches paper", score: 0.9, weight: 0.2, status: "pass" as const },
  { area: "Attention-mask leak-free", score: 0.2, weight: 0.1, status: "fail" as const },
];
describe("ReportRail", () => {
  it("shows the verdict and stat grid for a completed run", () => {
    render(<ReportRail status="completed" elapsedMs={4320000}
      report={{ finalReportPath: "x", costUsd: 18.4,
        counts: { iterations: 13, primitiveCalls: 21, proposed: 7, promoted: 2 } }}
      rubric={{ current: 0.53, baseline: 0.22, target: 0.4, series: [], areas }} />);
    expect(screen.getByText(/0\.53/)).toBeInTheDocument();
    expect(screen.getByText("13")).toBeInTheDocument();      // iterations tile
    expect(screen.getByText("7")).toBeInTheDocument();        // proposed tile
  });
  it("renders one breakdown row per rubric area with its status", () => {
    render(<ReportRail status="running" elapsedMs={0} report={null}
      rubric={{ current: 0.3, baseline: 0.22, target: 0.4, series: [], areas }} />);
    expect(screen.getByText("Architecture matches paper")).toBeInTheDocument();
    expect(screen.getByText("Attention-mask leak-free")).toBeInTheDocument();
  });
  it("shows a degraded-score note when the score is at the 0.35 cap", () => {
    render(<ReportRail status="completed" elapsedMs={1000}
      report={{ finalReportPath: "x", costUsd: null,
        counts: { iterations: 4, primitiveCalls: 3, proposed: 0, promoted: 0 } }}
      rubric={{ current: 0.35, baseline: 0.35, target: 0.4, series: [], areas: [] }} />);
    expect(screen.getByText(/degraded|no real metrics/i)).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run → FAIL**
- [ ] **Step 3: Implement** `ReportRail` — props `{ status, elapsedMs, report, rubric }`. Renders a verdict pill (from `status`; "in progress" while `running`), a 6-tile stat grid (iterations / primitive calls / proposed / promoted from `report.counts`; cost from `report.costUsd`; elapsed from `elapsedMs`), and a rubric breakdown (one row per `rubric.areas` — ✓ pass / ◐ partial / ✗ fail + area name). Honesty (§14): cost tile only when `costUsd != null`; when `rubric.current` is `<= 0.35`, render a visible "degraded — no real metrics" note; `report === null` → stat tiles show what's known, the rest dimmed.
- [ ] **Step 4: Run test + `tsc` + `lint` → green**
- [ ] **Step 5: Commit** — `git add` the three files; commit subject `feat(rlm-ui): ReportRail component` + the standard trailer.

---

## Task 10: PrimitiveHistoryBar

**Context:** spec §7.2 (PrimitiveHistoryBar). v3 bottom strip — collapsible, reverse-chronological primitive-call list.

**Files:** Create `frontend/src/components/lab/rlm/primitive-history-bar.tsx`, `.module.css`, `primitive-history-bar.test.tsx`.

- [ ] **Step 1: Failing test**

```tsx
import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { PrimitiveHistoryBar } from "./primitive-history-bar";

const calls = [
  { primitive: "understand_section", status: "ok" as const, iteration: 1 },
  { primitive: "run_experiment", status: "error" as const, iteration: 8 },
];
describe("PrimitiveHistoryBar", () => {
  it("is collapsed by default, showing only a count summary", () => {
    render(<PrimitiveHistoryBar calls={calls} />);
    expect(screen.getByText(/2 calls/)).toBeInTheDocument();
    expect(screen.queryByText("understand_section")).not.toBeInTheDocument();
  });
  it("expands to a reverse-chronological list on click", () => {
    render(<PrimitiveHistoryBar calls={calls} />);
    fireEvent.click(screen.getByRole("button"));
    const rows = screen.getAllByTestId("primitive-call-row");
    expect(rows[0]).toHaveTextContent("run_experiment"); // newest first
  });
});
```

- [ ] **Step 2: Run → FAIL**
- [ ] **Step 3: Implement** `PrimitiveHistoryBar` — prop `{ calls: PrimitiveCallView[] }`; owns local `collapsed` state (default true); collapsed shows `"▸ primitive call history — N calls"`; expanded shows reverse-chronological rows (`data-testid="primitive-call-row"`), each with primitive name + status (error rows in coral). The toggle is a `<button>`.
- [ ] **Step 4: Run test + `tsc` + `lint` → green**
- [ ] **Step 5: Commit** — subject `feat(rlm-ui): PrimitiveHistoryBar component` + trailer.

---

## Task 11: TreeNode

**Context:** spec §6 (node kinds), §8 (node visual), §9 (outcome palette). One node card in the exploration tree.

**Files:** Create `frontend/src/components/lab/rlm/tree-node.tsx`, `.module.css`, `tree-node.test.tsx`.

- [ ] **Step 1: Failing test**

```tsx
import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { TreeNode } from "./tree-node";
import type { TreeNode as TreeNodeData } from "../../../hooks/use-rlm-run";

const mk = (over: Partial<TreeNodeData>): TreeNodeData =>
  ({ id: "n1", parentId: "p", kind: "candidate", title: "learning-rate warmup",
     category: "optimizer", outcome: "promoted", rubricDelta: 0.14, round: 1 } as TreeNodeData);

describe("TreeNode", () => {
  it("renders a candidate's title and outcome", () => {
    render(<TreeNode node={mk({})} selected={false} onSelect={() => {}} />);
    expect(screen.getByText("learning-rate warmup")).toBeInTheDocument();
    expect(screen.getByText(/promoted/)).toBeInTheDocument();
  });
  it("applies an outcome-specific class (not color alone — has a label)", () => {
    const { container } = render(<TreeNode node={mk({})} selected={false} onSelect={() => {}} />);
    expect(container.firstChild).toHaveAttribute("data-outcome", "promoted");
  });
  it("renders a declined-group node with its count", () => {
    render(<TreeNode node={mk({ kind: "declined-group", declinedCount: 3, title: "3 declined" })}
      selected={false} onSelect={() => {}} />);
    expect(screen.getByText(/3 declined/)).toBeInTheDocument();
  });
  it("fires onSelect with the node id when clicked or activated by keyboard", () => {
    let picked = "";
    render(<TreeNode node={mk({})} selected={false} onSelect={(id) => { picked = id; }} />);
    fireEvent.click(screen.getByRole("button"));
    expect(picked).toBe("n1");
  });
});
```

- [ ] **Step 2: Run → FAIL**
- [ ] **Step 3: Implement** `TreeNode` — props `{ node: TreeNodeData; selected: boolean; onSelect: (id: string) => void }`. Kind-specific rendering (`paper` / `work` / `baseline` / `candidate` / `subrlm` / `declined-group` per spec §6); a `candidate` shows title + `category · rubricDelta` subtitle + an outcome badge; `data-outcome` attribute carries the outcome; outcome color from the §9 palette; `running` pulses (gated on `prefers-reduced-motion`). The node is a `<button>` (keyboard-focusable, ARIA label = title + outcome); `onSelect(node.id)` on activate; `selected` adds a highlight.
- [ ] **Step 4: Run test + `tsc` + `lint` → green**
- [ ] **Step 5: Commit** — subject `feat(rlm-ui): TreeNode component` + trailer.

---

## Task 12: TreeEdges

**Context:** spec §8 (SVG edge layer, colored by child outcome, dashed for failed/declined).

**Files:** Create `frontend/src/components/lab/rlm/tree-edges.tsx`, `tree-edges.test.tsx`.

- [ ] **Step 1: Failing test**

```tsx
import { describe, it, expect } from "vitest";
import { render } from "@testing-library/react";
import { TreeEdges } from "./tree-edges";

describe("TreeEdges", () => {
  const positioned = [
    { id: "p", x: 0, y: 0 }, { id: "a", x: 100, y: 0 }, { id: "b", x: 100, y: 60 },
  ];
  it("renders one SVG path per edge", () => {
    const { container } = render(<TreeEdges positioned={positioned as any}
      edges={[{ from: "p", to: "a", outcome: "promoted" },
              { from: "p", to: "b", outcome: "failed" }]} />);
    expect(container.querySelectorAll("path")).toHaveLength(2);
  });
  it("dashes a failed/declined edge", () => {
    const { container } = render(<TreeEdges positioned={positioned as any}
      edges={[{ from: "p", to: "b", outcome: "failed" }]} />);
    const path = container.querySelector("path")!;
    expect(path.getAttribute("stroke-dasharray")).toBeTruthy();
  });
});
```

- [ ] **Step 2: Run → FAIL**
- [ ] **Step 3: Implement** `TreeEdges` — props `{ positioned: PositionedNode[]; edges: Edge[] }` (types from `layout-tree.ts`). Renders one `<svg>` layer with one `<path>` per edge, routed from the parent's position to the child's; stroke color from the child `outcome` (§9 palette), `failed`/`declined` get `stroke-dasharray`. Purely positional — no interactivity. `aria-hidden` (decorative).
- [ ] **Step 4: Run test + `tsc` + `lint` → green**
- [ ] **Step 5: Commit** — subject `feat(rlm-ui): TreeEdges SVG layer` + trailer.

---

## Task 13: NodeDetailPopup

**Context:** spec §7.2 (NodeDetailPopup), §8 (selection → popup), §9 (accessibility — dismissable, focus return).

**Files:** Create `frontend/src/components/lab/rlm/node-detail-popup.tsx`, `.module.css`, `node-detail-popup.test.tsx`.

- [ ] **Step 1: Failing test**

```tsx
import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { NodeDetailPopup } from "./node-detail-popup";

const node = { id: "n1", parentId: "p", kind: "candidate" as const, title: "longer warmup",
  category: "optimizer", outcome: "running" as const, round: 2 };
const iteration = { iteration: 11, response: "Extending warmup to 8k steps.",
  code_blocks: [{ code: "res = run_experiment(code, env)",
    stdout_meta: { length: 1400, prefix: "ok", has_traceback: false } }] };

describe("NodeDetailPopup", () => {
  it("shows the node title, iteration code, and reasoning", () => {
    render(<NodeDetailPopup node={node as any} iteration={iteration as any} onClose={() => {}} />);
    expect(screen.getByText(/longer warmup/)).toBeInTheDocument();
    expect(screen.getByText(/run_experiment/)).toBeInTheDocument();
    expect(screen.getByText(/Extending warmup/)).toBeInTheDocument();
  });
  it("calls onClose on the dismiss control and on Escape", () => {
    let closed = 0;
    render(<NodeDetailPopup node={node as any} iteration={iteration as any}
      onClose={() => { closed += 1; }} />);
    fireEvent.click(screen.getByRole("button", { name: /close|dismiss/i }));
    fireEvent.keyDown(document, { key: "Escape" });
    expect(closed).toBe(2);
  });
});
```

- [ ] **Step 2: Run → FAIL**
- [ ] **Step 3: Implement** `NodeDetailPopup` — props `{ node: TreeNode; iteration: IterationView | null; onClose: () => void }`. A floating card showing the node title + identity, the iteration's `code` blocks, `stdout_meta` summary, root `response` reasoning, and (for a candidate) category + rubric delta. Dismissable via a close `<button>` and the `Escape` key; on unmount returns focus to the triggering node (the caller passes focus back — document the contract). Corpus-safe: render only the §4 metadata fields.
- [ ] **Step 4: Run test + `tsc` + `lint` → green**
- [ ] **Step 5: Commit** — subject `feat(rlm-ui): NodeDetailPopup component` + trailer.

---

## Task 14: ExplorationCanvas

**Context:** spec §8 (the whole tree canvas — the largest component). Composes `layoutTree` + `TreeNode` + `TreeEdges` + `NodeDetailPopup`, adds pan/zoom, soft-cap collapse, declined-collapse, selection.

**Files:** Create `frontend/src/components/lab/rlm/exploration-canvas.tsx`, `.module.css`, `exploration-canvas.test.tsx`.

- [ ] **Step 1: Failing test**

```tsx
import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ExplorationCanvas } from "./exploration-canvas";
import { fold, INITIAL_RLM_STATE } from "../../../hooks/use-rlm-run";
import { rlmRunFixture } from "./__fixtures__/rlm-run.fixture";

const state = rlmRunFixture.reduce(fold, INITIAL_RLM_STATE);

describe("ExplorationCanvas", () => {
  it("renders a node for every tree node", () => {
    render(<ExplorationCanvas tree={state.tree} iterations={state.iterations} />);
    // paper + work + baseline + candidates + declined-group + subrlm
    expect(screen.getAllByRole("button").length).toBeGreaterThanOrEqual(state.tree.length);
  });
  it("opens the detail popup when a node is clicked", () => {
    render(<ExplorationCanvas tree={state.tree} iterations={state.iterations} />);
    fireEvent.click(screen.getAllByRole("button")[1]);
    expect(screen.getByTestId("node-detail-popup")).toBeInTheDocument();
  });
  it("collapses a fan beyond the 8-node soft cap into a +N node", () => {
    const big = [{ id: "b", parentId: null, kind: "baseline", title: "baseline" },
      ...Array.from({ length: 12 }, (_, i) =>
        ({ id: `c${i}`, parentId: "b", kind: "candidate", title: `c${i}`, round: 1 }))]
      as typeof state.tree;
    render(<ExplorationCanvas tree={big} iterations={[]} />);
    expect(screen.getByText(/\+\d+ more/)).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run → FAIL**
- [ ] **Step 3: Implement** `ExplorationCanvas` — props `{ tree: TreeNode[]; iterations: IterationView[] }`. Runs `layoutTree(tree)`; renders `TreeEdges` (behind) + a `TreeNode` per positioned node (absolutely positioned); wraps the scene in pan/zoom (reuse `../pan-wrap` / `usePan`); owns `selectedNodeId` state — selecting a node renders `NodeDetailPopup` (`data-testid="node-detail-popup"`) for it, resolving its `IterationView` from `iterations`; the live frontier node is auto-selected until the user clicks. Soft cap: a fan with >8 sibling candidates renders the first 8 + a `"+N more"` node that expands on click. `declined-group` nodes already arrive collapsed from the reducer. New nodes fade-in 200ms; respect `prefers-reduced-motion`.
- [ ] **Step 4: Run test + `tsc` + `lint` → green**
- [ ] **Step 5: Commit** — subject `feat(rlm-ui): ExplorationCanvas — the live tree canvas` + trailer.

---

## Task 15: RlmLab shell + the WorkflowView branch

**Context:** spec §3 (coexistence), §7 (the 4-band layout). Composes everything; wires the one `runMode` branch into the existing lab.

**Files:**
- Create: `frontend/src/components/lab/rlm/rlm-lab.tsx`, `.module.css`, `rlm-lab.test.tsx`
- Modify: `frontend/src/components/lab/lab-shell.tsx` (the `WorkflowView` `runMode` branch)
- Test: `frontend/src/components/lab/lab-shell.test.tsx` (add a runMode-branch case)

- [ ] **Step 1: Write the failing tests**

```tsx
// rlm-lab.test.tsx
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { RlmLab } from "./rlm-lab";
import { rlmRunFixture } from "./__fixtures__/rlm-run.fixture";

describe("RlmLab", () => {
  it("renders all 6 regions from an event stream", () => {
    render(<RlmLab events={rlmRunFixture} runMeta={{ projectId: "prj_x",
      paperTitle: "Attention is all you need", paperMeta: "Vaswani et al." }} />);
    expect(screen.getByText("Attention is all you need")).toBeInTheDocument(); // header
    expect(screen.getByText(/0\.53/)).toBeInTheDocument();                      // rubric strip
    expect(screen.getByText(/primitive call history/i)).toBeInTheDocument();    // history bar
    expect(screen.getAllByRole("button").length).toBeGreaterThan(5);            // tree nodes
  });
});
```

For `lab-shell.test.tsx`, add a case: a `run` with `runMode: "rlm"` causes `WorkflowView` to render `<RlmLab>` (assert an RlmLab-only testid), and a `run` with `runMode: "offline"` still renders the existing `LabCanvas` (assert it is unchanged).

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx vitest run src/components/lab/rlm/rlm-lab.test.tsx src/components/lab/lab-shell.test.tsx`
Expected: FAIL — `RlmLab` missing; `lab-shell` has no `rlm` branch.

- [ ] **Step 3: Implement**

Create `RlmLab` — props `{ events: RlmDashboardEvent[]; runMeta: { projectId; paperTitle; paperMeta } }`. Calls `useRlmRun(events)`; computes `elapsedMs` from the first/last event `timestamp`; lays out the 4 bands (spec §7.1) — `RlmHeader`, `RubricStrip`, a 3-zone workspace row (`ReplStateRail` | `ExplorationCanvas` | `ReportRail`), `PrimitiveHistoryBar`; owns rail-collapse state. Then in `lab-shell.tsx`, find `WorkflowView` and add, as the first branch: `if (run.runMode === "rlm") return <RlmLab events={dashboardEvents} runMeta={…} />;` — derive `runMeta` from the existing `run` fields. Touch nothing else in `lab-shell.tsx`; the `offline`/`sdk` path is unchanged.

- [ ] **Step 4: Run tests + checks**

Run: `npx vitest run` (the whole suite — confirm no regression in `lab-shell` or elsewhere)
Run: `npx tsc --noEmit` clean · `npm run lint` clean

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/lab/rlm/rlm-lab.tsx frontend/src/components/lab/rlm/rlm-lab.module.css frontend/src/components/lab/rlm/rlm-lab.test.tsx frontend/src/components/lab/lab-shell.tsx frontend/src/components/lab/lab-shell.test.tsx
git commit -m "feat(rlm-ui): RlmLab shell + WorkflowView runMode branch

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 16: Replay harness + the ?rlmFixture=1 dev path

**Context:** spec §10 (replay harness). Lets the lab render the fixture with no backend — used by manual dev and the Playwright e2e.

**Files:**
- Create: `frontend/src/components/lab/rlm/replay.ts`, `replay.test.ts`
- Modify: `frontend/src/components/lab/lab-shell.tsx` (read the `?rlmFixture=1` query param)

- [ ] **Step 1: Failing test**

```ts
import { describe, it, expect, vi } from "vitest";
import { replayFixture } from "./replay";
import { rlmRunFixture } from "./__fixtures__/rlm-run.fixture";

describe("replayFixture", () => {
  it("instant mode yields the whole fixture at once", () => {
    expect(replayFixture("instant")).toEqual(rlmRunFixture);
  });
  it("timed mode emits events progressively", async () => {
    vi.useFakeTimers();
    const seen: number[] = [];
    const stop = replayFixture("timed", (events) => seen.push(events.length));
    await vi.runAllTimersAsync();
    stop();
    expect(seen[seen.length - 1]).toBe(rlmRunFixture.length);
    expect(seen.length).toBeGreaterThan(1);
    vi.useRealTimers();
  });
});
```

- [ ] **Step 2: Run → FAIL**
- [ ] **Step 3: Implement** `replay.ts` — `replayFixture(mode: "instant", )` returns the full array; `replayFixture("timed", onUpdate)` emits a growing prefix on a timer (~150ms/event) and returns a `stop()`. Then in `lab-shell.tsx`: when the page query has `rlmFixture=1`, `WorkflowView` mounts `<RlmLab events={replayed} runMeta={fixtureMeta} />` regardless of any live run — a dev/test-only path; guard it so it never affects a real run.
- [ ] **Step 4: Run test + full `npx vitest run` + `tsc` + `lint` → green**
- [ ] **Step 5: Commit** — subject `feat(rlm-ui): fixture replay harness + ?rlmFixture dev path` + trailer.

---

## Task 17: Playwright e2e

**Context:** spec §11. Drives the real lab at `/lab?rlmFixture=1`.

**Files:** Create `frontend/e2e/rlm-lab.spec.ts`

- [ ] **Step 1: Write the test**

```ts
import { test, expect } from "@playwright/test";

test("RLM lab renders the exploration tree from the fixture", async ({ page }) => {
  await page.goto("/lab?rlmFixture=1");
  await expect(page.getByText("Attention is all you need")).toBeVisible();
  await expect(page.getByText(/0\.53/)).toBeVisible();                 // rubric strip
  await expect(page.getByText("learning-rate warmup")).toBeVisible();  // a candidate node
  await page.getByText("learning-rate warmup").click();
  await expect(page.getByTestId("node-detail-popup")).toBeVisible();   // click → popup
});
```

- [ ] **Step 2: Run to verify it fails** (or is pending)

Run: `npx playwright test e2e/rlm-lab.spec.ts` — FAIL until the dev path serves the fixture.

- [ ] **Step 3: Make it pass** — with Tasks 1–16 done, the `?rlmFixture=1` path already serves it. Fix any selector mismatches surfaced here (adjust `data-testid`s / text in the components, re-running their unit tests).
- [ ] **Step 4: Run `npx playwright test e2e/rlm-lab.spec.ts` → PASS**; full `npx vitest run`, `tsc`, `lint` still green.
- [ ] **Step 5: Commit** — subject `test(rlm-ui): Playwright e2e for the RLM lab` + trailer.

---

## Task 18: Backend-emission handoff doc

**Context:** spec §13. Documents how a later backend task makes the 3 fixture-only events go live. Not code — a doc.

**Files:** Create `docs/superpowers/specs/2026-05-21-rlm-phase4-backend-events-handoff.md`

- [ ] **Step 1: Write the doc** — per spec §13: for each of `candidate_proposed`, `candidate_outcome`, `rubric_score` — the exact schema (copy from spec §4.2), where in `backend/agents/rlm/binding.py` / `run.py` it should be emitted, corpus-safety, and the `candidate_outcome` run-level-correlation challenge. State clearly that implementing it is **not** Phase 4 scope.
- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-05-21-rlm-phase4-backend-events-handoff.md
git commit -m "docs(rlm): Phase 4 — backend-emission handoff for the 3 RLM events

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Done condition (verify before declaring Phase 4 complete)

- All 18 tasks committed on `feat/rlm-phase4-frontend`.
- `npm run lint`, `npx tsc --noEmit`, `npx vitest run`, `npx playwright test` — all green.
- `/lab?rlmFixture=1` renders the full v3 lab; the exploration tree populates; clicking a node opens the detail popup; the old 14-stage UI (`offline`/`sdk` runs) is unchanged.
- Maps to spec §1 done condition and issue #61's done condition.

## Plan self-review

- **Spec coverage:** §3 integration → T15; §4 events → T1; §5 state/reducer → T3+T4; §6 tree model → T4; §7 layout/components → T6–T15; §8 tree canvas → T5,T11–T14; §9 visual rules → applied across T6–T14; §10 fixture/replay → T2,T16; §11 testing → tests in every task + T17; §13 handoff → T18; §14 honesty → T6 (cost), T9 (0.35 cap). All sections covered.
- **Type consistency:** `RlmDashboardEvent`, `RlmRunState`, `TreeNode`, `IterationView`, `PrimitiveCallView`, `RubricArea`, `SubRlmView` (from `use-rlm-run.ts`); `PositionedNode`, `Edge` (from `layout-tree.ts`); `fold` / `INITIAL_RLM_STATE` / `useRlmRun` names used identically in every task that imports them.
- **Ordering:** T1→T2→T3→T4 are strictly sequential (each imports the prior). T5 needs T4's `TreeNode`. T6–T13 are mutually independent (parallelizable). T14 needs T5+T11+T12+T13. T15 needs T6–T10+T14. T16 needs T15. T17 needs T16. T18 is independent.

