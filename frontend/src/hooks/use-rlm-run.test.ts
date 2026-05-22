import { describe, it, expect } from "vitest";
import { fold, INITIAL_RLM_STATE } from "./use-rlm-run";
import type { RlmDashboardEvent } from "../lib/events/rlm-events";
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
  it("never mutates its inputs", () => {
    const fixtureClone = structuredClone(rlmRunFixture);
    const initialClone = structuredClone(INITIAL_RLM_STATE);
    rlmRunFixture.reduce(fold, INITIAL_RLM_STATE);
    expect(rlmRunFixture).toEqual(fixtureClone);
    expect(INITIAL_RLM_STATE).toEqual(initialClone);
  });
  it("keeps iterations a dense 1..N array (no holes, stream order)", () => {
    const s = reduce();
    expect(s.iterations.map((v) => v.iteration)).toEqual(
      Array.from({ length: s.iterationCount }, (_, i) => i + 1)
    );
  });
  it("does not hold a shared reference into the input code_blocks", () => {
    const s = reduce();
    const firstIter = rlmRunFixture.find((e) => e.event === "repl_iteration");
    if (firstIter && firstIter.event === "repl_iteration") {
      expect(s.iterations[0].code_blocks).not.toBe(firstIter.code_blocks);
      expect(s.iterations[0].code_blocks).toEqual(firstIter.code_blocks);
    }
  });
});

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

  /**
   * §5.3 branch (b) — the previous fan's parent.
   *
   * Setup: emit a work node → baseline → round-1 fan whose candidates explicitly
   * parent on the work node (not baseline) via `parent_id`. Round-1 promotes
   * nothing (all marginal). A round-2 candidate arrives with no `parent_id`.
   *
   * Expected: `frontierParent` skips branch (a) (no promoted node) and fires
   * branch (b), returning the previous fan's parent — the work node — NOT the
   * baseline node that branch (c) would return. The work-node id is therefore
   * a value-discriminating witness between (b) and (c).
   */
  it("§5.3 branch (b): round-2 candidate parents on the previous fan's parent when round-1 promotes nothing", () => {
    const T = "2026-01-01T00:00:00.000Z";

    // Build a minimal work node by emitting a primitive_call with phase.
    const events: RlmDashboardEvent[] = [
      {
        event: "primitive_call",
        timestamp: T,
        primitive: "implement_baseline",
        status: "start",
        args_summary: {},
        result_summary: null,
        iteration: 1,
        rubric_delta: null,
      },
      // rubric_score → creates the baseline node (parent = work node above).
      {
        event: "rubric_score",
        timestamp: T,
        iteration: 1,
        score: 0.2,
        target: 0.7,
        areas: [{ area: "x", score: 0.2, weight: 1.0, status: "fail" }],
      },
      // Round-1 fan: two candidates explicitly parented on the work node (not
      // baseline). This makes branch (b) return the work node id, while branch
      // (c) would return "baseline" — a clear value distinction.
      {
        event: "candidate_proposed",
        timestamp: T,
        iteration: 2,
        round: 1,
        parent_id: "work-baseline-build-1",
        candidate: {
          id: "a1",
          title: "candidate a1",
          category: "test",
          description: "desc",
          reasoning: "reason",
        },
      },
      {
        event: "candidate_proposed",
        timestamp: T,
        iteration: 2,
        round: 1,
        parent_id: "work-baseline-build-1",
        candidate: {
          id: "a2",
          title: "candidate a2",
          category: "test",
          description: "desc",
          reasoning: "reason",
        },
      },
      // Both round-1 candidates resolve as marginal — no promotion.
      {
        event: "candidate_outcome",
        timestamp: T,
        iteration: 3,
        candidate_id: "a1",
        outcome: "marginal",
        rubric_delta: 0.01,
      },
      {
        event: "candidate_outcome",
        timestamp: T,
        iteration: 3,
        candidate_id: "a2",
        outcome: "marginal",
        rubric_delta: 0.01,
      },
      // Round-2 candidate with NO explicit parent_id — reducer must infer.
      {
        event: "candidate_proposed",
        timestamp: T,
        iteration: 4,
        round: 2,
        candidate: {
          id: "b1",
          title: "candidate b1",
          category: "test",
          description: "desc",
          reasoning: "reason",
        },
      },
    ];

    const s = events.reduce(fold, INITIAL_RLM_STATE);

    const b1 = s.tree.find((n) => n.kind === "candidate" && n.candidate?.id === "b1");
    expect(b1).toBeDefined();
    // Branch (b): the previous fan's parentId is "work-baseline-build-1".
    // Branch (c) would have returned "baseline" — different value.
    expect(b1!.parentId).toBe("work-baseline-build-1");
    // Confirm neither a promoted-node id nor orphaned.
    expect(b1!.parentId).not.toBe("baseline");
  });
});
