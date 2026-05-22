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
