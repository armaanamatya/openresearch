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
  it("transitions status queued→running on first primitive_call", () => {
    const ev: RlmDashboardEvent = {
      event: "primitive_call",
      timestamp: "2026-05-23T00:00:00.000Z",
      primitive: "understand_section",
      status: "start",
      args_summary: {},
      result_summary: null,
      iteration: 1,
      rubric_delta: null,
    };
    expect(fold(INITIAL_RLM_STATE, ev).status).toBe("running");
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
  it("every tree node carries a non-empty title", () => {
    const s = reduce();
    for (const n of s.tree) {
      expect(n.title).toBeTruthy();
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

// ──────────────────────────────────────────────────────────────────────────────
// Phase-4-forward-compat extensions
// Spec: docs/superpowers/specs/2026-05-23-rubric-climb-leaderboard.md §4.2
// ──────────────────────────────────────────────────────────────────────────────

import type { RubricScoreEvent } from "../lib/events/rlm-events";

function _rubricEvent(
  iteration: number,
  score: number,
  areas: Array<{ area: string; status: "pass" | "partial" | "fail"; score: number; weight: number }>,
): RubricScoreEvent {
  return {
    event: "rubric_score",
    timestamp: new Date(2026, 4, 23, 4, 10 + iteration).toISOString(),
    iteration,
    score,
    target: 0.8,
    areas,
  };
}

describe("rubric.previousAreas (spec §4.2)", () => {
  it("is empty on initial state", () => {
    expect(INITIAL_RLM_STATE.rubric.previousAreas).toEqual([]);
  });

  it("captures the prior areas snapshot when a new rubric_score arrives", () => {
    const e1 = _rubricEvent(1, 0.30, [
      { area: "Setup", status: "fail", score: 0.1, weight: 1 },
      { area: "Train", status: "fail", score: 0.05, weight: 1 },
    ]);
    const e2 = _rubricEvent(3, 0.55, [
      { area: "Setup", status: "pass", score: 0.85, weight: 1 },
      { area: "Train", status: "partial", score: 0.45, weight: 1 },
    ]);

    let s = fold(INITIAL_RLM_STATE, e1);
    // After e1, previousAreas is still empty (no prior snapshot).
    expect(s.rubric.previousAreas).toEqual([]);
    expect(s.rubric.areas.length).toBe(2);

    s = fold(s, e2);
    // After e2, previousAreas is the e1 areas snapshot.
    expect(s.rubric.previousAreas.map((a) => a.area)).toEqual(["Setup", "Train"]);
    expect(s.rubric.previousAreas[0].status).toBe("fail");
    expect(s.rubric.areas[0].status).toBe("pass");
  });
});

describe("rubric.attributableCandidate (spec §4.2)", () => {
  it("starts as null", () => {
    expect(INITIAL_RLM_STATE.rubric.attributableCandidate).toBeNull();
  });

  it("is set when candidate_proposed lands", () => {
    const ev = {
      event: "candidate_proposed" as const,
      timestamp: "2026-05-23T04:10:00.000Z",
      iteration: 5,
      round: 1,
      candidate: {
        id: "c1",
        title: "Bigger batch size",
        category: "training",
        description: "...",
        reasoning: "...",
      },
    };
    const s = fold(INITIAL_RLM_STATE, ev);
    expect(s.rubric.attributableCandidate).toEqual({
      id: "c1",
      title: "Bigger batch size",
      outcome: null,
    });
  });

  it("updates outcome when matching candidate_outcome lands", () => {
    let s = fold(INITIAL_RLM_STATE, {
      event: "candidate_proposed",
      timestamp: "2026-05-23T04:10:00.000Z",
      iteration: 5,
      round: 1,
      candidate: { id: "c1", title: "Bigger batch", category: "training", description: "", reasoning: "" },
    });
    s = fold(s, {
      event: "candidate_outcome",
      timestamp: "2026-05-23T04:11:00.000Z",
      iteration: 6,
      candidate_id: "c1",
      outcome: "promoted",
      rubric_delta: 0.12,
    });
    expect(s.rubric.attributableCandidate).toEqual({
      id: "c1",
      title: "Bigger batch",
      outcome: "promoted",
    });
  });

  it("overwrites the pointer when a newer candidate is proposed", () => {
    let s = fold(INITIAL_RLM_STATE, {
      event: "candidate_proposed",
      timestamp: "2026-05-23T04:10:00.000Z",
      iteration: 5,
      round: 1,
      candidate: { id: "c1", title: "First", category: "x", description: "", reasoning: "" },
    });
    s = fold(s, {
      event: "candidate_proposed",
      timestamp: "2026-05-23T04:11:00.000Z",
      iteration: 6,
      round: 1,
      candidate: { id: "c2", title: "Second", category: "x", description: "", reasoning: "" },
    });
    expect(s.rubric.attributableCandidate?.id).toBe("c2");
    expect(s.rubric.attributableCandidate?.title).toBe("Second");
  });

  it("reflects buffered outcome when candidate_outcome arrives before candidate_proposed", () => {
    // §4.2 + §5.3 out-of-order path: outcome arrives first.
    let s = fold(INITIAL_RLM_STATE, {
      event: "candidate_outcome",
      timestamp: "2026-05-23T04:10:00.000Z",
      iteration: 5,
      candidate_id: "c1",
      outcome: "promoted",
      rubric_delta: 0.15,
    });
    // Before the proposal arrives, attributableCandidate is null (the buffered
    // outcome lives only in _pendingOutcomes).
    expect(s.rubric.attributableCandidate).toBeNull();

    s = fold(s, {
      event: "candidate_proposed",
      timestamp: "2026-05-23T04:11:00.000Z",
      iteration: 6,
      round: 1,
      candidate: { id: "c1", title: "Bigger batch", category: "x", description: "", reasoning: "" },
    });
    // Once the proposal lands, the pointer is set with the buffered outcome.
    expect(s.rubric.attributableCandidate).toEqual({
      id: "c1",
      title: "Bigger batch",
      outcome: "promoted",
    });
  });
});

// ──────────────────────────────────────────────────────────────────────────────
// Constellation node kinds — primitive / llm_primitive
// Spec: live constellation feature 2026-05-23
// ──────────────────────────────────────────────────────────────────────────────

const T_CONST = "2026-05-23T10:00:00.000Z";

describe("foldPrimitiveCall — constellation nodes", () => {
  it("adds an llm_primitive node for understand_section with status=ok", () => {
    const ev: RlmDashboardEvent = {
      event: "primitive_call",
      timestamp: T_CONST,
      primitive: "understand_section",
      status: "ok",
      args_summary: {},
      result_summary: "MethodSpec[...]",
      iteration: 1,
      rubric_delta: null,
    };
    const s = fold(INITIAL_RLM_STATE, ev);
    const llmNodes = s.tree.filter((n) => n.kind === "llm_primitive");
    expect(llmNodes).toHaveLength(1);
    expect(llmNodes[0].primitiveName).toBe("understand_section");
    expect(llmNodes[0].primitivePhase).toBe("understand");
  });

  it("adds a plain primitive node for build_environment with status=ok", () => {
    const ev: RlmDashboardEvent = {
      event: "primitive_call",
      timestamp: T_CONST,
      primitive: "build_environment",
      status: "ok",
      args_summary: {},
      result_summary: "env_id[...]",
      iteration: 2,
      rubric_delta: null,
    };
    const s = fold(INITIAL_RLM_STATE, ev);
    const primNodes = s.tree.filter((n) => n.kind === "primitive");
    expect(primNodes).toHaveLength(1);
    expect(primNodes[0].primitiveName).toBe("build_environment");
    expect(primNodes[0].primitivePhase).toBe("environment");
  });

  it("does NOT add a node for heartbeat (non-visualized primitive)", () => {
    // check_user_messages is in NON_VISUALIZED_PRIMITIVES
    const ev: RlmDashboardEvent = {
      event: "primitive_call",
      timestamp: T_CONST,
      primitive: "check_user_messages",
      status: "ok",
      args_summary: {},
      result_summary: null,
      iteration: 1,
      rubric_delta: null,
    };
    const s = fold(INITIAL_RLM_STATE, ev);
    const primNodes = s.tree.filter(
      (n) => n.kind === "primitive" || n.kind === "llm_primitive"
    );
    expect(primNodes).toHaveLength(0);
  });

  it("does NOT add a node for a start-status primitive_call (only ok)", () => {
    const ev: RlmDashboardEvent = {
      event: "primitive_call",
      timestamp: T_CONST,
      primitive: "understand_section",
      status: "start",
      args_summary: {},
      result_summary: null,
      iteration: 1,
      rubric_delta: null,
    };
    const s = fold(INITIAL_RLM_STATE, ev);
    const primNodes = s.tree.filter(
      (n) => n.kind === "primitive" || n.kind === "llm_primitive"
    );
    expect(primNodes).toHaveLength(0);
  });

  it("does NOT add a node for record_candidate_outcome (non-visualized)", () => {
    const ev: RlmDashboardEvent = {
      event: "primitive_call",
      timestamp: T_CONST,
      primitive: "record_candidate_outcome",
      status: "ok",
      args_summary: {},
      result_summary: null,
      iteration: 3,
      rubric_delta: null,
    };
    const s = fold(INITIAL_RLM_STATE, ev);
    const primNodes = s.tree.filter(
      (n) => n.kind === "primitive" || n.kind === "llm_primitive"
    );
    expect(primNodes).toHaveLength(0);
  });
});

// ──────────────────────────────────────────────────────────────────────────────
// Lane γ — perModelMetrics (per-model multi-model UI, 2026-05-23)
// ──────────────────────────────────────────────────────────────────────────────

const T_EXP = "2026-05-23T12:00:00.000Z";

describe("foldExperimentCompleted — perModelMetrics", () => {
  it("starts as null in INITIAL_RLM_STATE", () => {
    expect(INITIAL_RLM_STATE.perModelMetrics).toBeNull();
  });

  it("sets perModelMetrics from a successful experiment_completed with per_model", () => {
    const ev: RlmDashboardEvent = {
      event: "experiment_completed",
      timestamp: T_EXP,
      success: true,
      metrics: { wall_time_seconds: 697 },
      per_model: {
        qwen3_1_7b: { alfworld_success_rate: 0.34, searchqa_accuracy: 0.42 },
        qwen2_5_3b: { alfworld_success_rate: 0.51, searchqa_accuracy: 0.55 },
      },
    };
    const s = fold(INITIAL_RLM_STATE, ev);
    expect(s.perModelMetrics).not.toBeNull();
    expect(Object.keys(s.perModelMetrics!).sort()).toEqual(["qwen2_5_3b", "qwen3_1_7b"]);
    expect(s.perModelMetrics!["qwen3_1_7b"].alfworld_success_rate).toBeCloseTo(0.34);
    expect(s.perModelMetrics!["qwen2_5_3b"].searchqa_accuracy).toBeCloseTo(0.55);
  });

  it("does NOT update perModelMetrics on a failed experiment", () => {
    // First set some per-model metrics.
    const successEv: RlmDashboardEvent = {
      event: "experiment_completed",
      timestamp: T_EXP,
      success: true,
      metrics: {},
      per_model: { model_a: { acc: 0.8 }, model_b: { acc: 0.7 } },
    };
    let s = fold(INITIAL_RLM_STATE, successEv);
    expect(s.perModelMetrics).not.toBeNull();

    // Now a failed experiment — must NOT overwrite.
    const failEv: RlmDashboardEvent = {
      event: "experiment_completed",
      timestamp: T_EXP,
      success: false,
      metrics: { error: "oom" },
      per_model: { model_a: { acc: 0.0 }, model_b: { acc: 0.0 } },
    };
    s = fold(s, failEv);
    // Still the successful data.
    expect(s.perModelMetrics!["model_a"].acc).toBeCloseTo(0.8);
  });

  it("leaves perModelMetrics null when per_model is absent (single-model paper)", () => {
    const ev: RlmDashboardEvent = {
      event: "experiment_completed",
      timestamp: T_EXP,
      success: true,
      metrics: { bleu: 28.4 },
    };
    const s = fold(INITIAL_RLM_STATE, ev);
    expect(s.perModelMetrics).toBeNull();
  });

  it("leaves perModelMetrics null when per_model is an empty dict", () => {
    const ev: RlmDashboardEvent = {
      event: "experiment_completed",
      timestamp: T_EXP,
      success: true,
      metrics: {},
      per_model: {},
    };
    const s = fold(INITIAL_RLM_STATE, ev);
    expect(s.perModelMetrics).toBeNull();
  });

  it("is pure — repeated fold of same event produces same result", () => {
    const ev: RlmDashboardEvent = {
      event: "experiment_completed",
      timestamp: T_EXP,
      success: true,
      metrics: {},
      per_model: { a: { x: 1 }, b: { x: 2 } },
    };
    expect(fold(INITIAL_RLM_STATE, ev)).toEqual(fold(INITIAL_RLM_STATE, ev));
  });
});
