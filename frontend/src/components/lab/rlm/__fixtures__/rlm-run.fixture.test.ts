/* eslint-disable @typescript-eslint/no-explicit-any */
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
    // all 6 candidate_outcome variants present
    const outcomesSet = new Set(outcomes.map((o) => o.outcome));
    expect(outcomesSet).toEqual(
      new Set(["promoted", "marginal", "failed", "running", "skipped", "declined"]));
    // an out-of-order proposed/outcome pair (an outcome before its proposed)
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
  });
});
