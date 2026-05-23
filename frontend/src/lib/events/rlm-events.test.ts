import { describe, it, expect } from "vitest";
import { isRlmEvent, RLM_EVENT_TYPES } from "./rlm-events";

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
