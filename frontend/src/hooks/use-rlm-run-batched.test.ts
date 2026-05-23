/**
 * Tests for D1 (rAF batching) and the batched hook in use-rlm-run.ts
 * TDD spec: D1 — 10 synchronous addEvent calls produce ONE re-fold.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useRlmRunBatched } from "./use-rlm-run";
import { fold, INITIAL_RLM_STATE } from "./use-rlm-run";
import type { RlmDashboardEvent } from "../lib/events/rlm-events";

// ── Fixtures ─────────────────────────────────────────────────────────────────

const T = "2026-05-23T10:00:00.000Z";

function makePrimitiveEvent(i: number): RlmDashboardEvent {
  return {
    event: "primitive_call",
    timestamp: T,
    primitive: "understand_section",
    status: "ok",
    args_summary: { slice: `section-${i}` },
    result_summary: `result-${i}`,
    iteration: i,
    rubric_delta: null,
  };
}

// ── D1: rAF batching ─────────────────────────────────────────────────────────

describe("useRlmRunBatched — rAF batching (D1)", () => {
  let rafCallbacks: FrameRequestCallback[];

  beforeEach(() => {
    rafCallbacks = [];
    // Intercept requestAnimationFrame so we control when frames fire.
    vi.spyOn(window, "requestAnimationFrame").mockImplementation((cb) => {
      rafCallbacks.push(cb);
      return rafCallbacks.length;
    });
    vi.spyOn(window, "cancelAnimationFrame").mockImplementation(() => {});
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("10 synchronous addEvent calls produce ONE state update (one re-fold)", () => {
    // Spy on the fold function is tricky since it's imported; instead we count
    // render cycles by watching state reference changes.
    const { result } = renderHook(() => useRlmRunBatched());
    const { addEvent } = result.current;

    // Capture initial state reference.
    const statesBefore = [result.current.state];

    // Fire 10 addEvent calls synchronously — no rAF has flushed yet.
    act(() => {
      for (let i = 1; i <= 10; i++) {
        addEvent(makePrimitiveEvent(i));
      }
    });

    // State should NOT have changed yet (rAF not flushed).
    // The pending buffer holds events but doesn't setState.
    expect(result.current.state).toBe(statesBefore[0]);

    // Now flush the rAF.
    act(() => {
      rafCallbacks.forEach((cb) => cb(0));
      rafCallbacks.length = 0;
    });

    // After flush, state updated once (10 events folded in one pass).
    expect(result.current.state).not.toBe(statesBefore[0]);
    // All 10 primitive call events have been folded.
    expect(result.current.state.primitiveCalls.length).toBe(10);
    // Only 1 requestAnimationFrame was scheduled (not 10).
    expect(window.requestAnimationFrame).toHaveBeenCalledTimes(1);
  });

  it("preserves event ordering after batch flush", () => {
    const { result } = renderHook(() => useRlmRunBatched());
    const { addEvent } = result.current;

    act(() => {
      addEvent(makePrimitiveEvent(1));
      addEvent(makePrimitiveEvent(2));
      addEvent(makePrimitiveEvent(3));
    });

    act(() => {
      rafCallbacks.forEach((cb) => cb(0));
      rafCallbacks.length = 0;
    });

    const calls = result.current.state.primitiveCalls;
    expect(calls.length).toBe(3);
    expect(calls[0].args_summary).toEqual({ slice: "section-1" });
    expect(calls[1].args_summary).toEqual({ slice: "section-2" });
    expect(calls[2].args_summary).toEqual({ slice: "section-3" });
  });

  it("a second batch after the first flushes correctly", () => {
    const { result } = renderHook(() => useRlmRunBatched());
    const { addEvent } = result.current;

    // First batch
    act(() => {
      addEvent(makePrimitiveEvent(1));
      addEvent(makePrimitiveEvent(2));
    });
    act(() => {
      rafCallbacks.forEach((cb) => cb(0));
      rafCallbacks.length = 0;
    });
    expect(result.current.state.primitiveCalls.length).toBe(2);

    // Second batch
    act(() => {
      addEvent(makePrimitiveEvent(3));
      addEvent(makePrimitiveEvent(4));
    });
    act(() => {
      rafCallbacks.forEach((cb) => cb(0));
      rafCallbacks.length = 0;
    });
    expect(result.current.state.primitiveCalls.length).toBe(4);
    // Only 2 total rAF frames — one per batch.
    expect(window.requestAnimationFrame).toHaveBeenCalledTimes(2);
  });

  it("initial state is INITIAL_RLM_STATE", () => {
    const { result } = renderHook(() => useRlmRunBatched());
    expect(result.current.state).toEqual(INITIAL_RLM_STATE);
  });
});

// ── Existing fold purity unchanged ────────────────────────────────────────────

describe("fold purity preserved after D1 changes", () => {
  it("fold is still a pure function", () => {
    const ev = makePrimitiveEvent(1);
    const s1 = fold(INITIAL_RLM_STATE, ev);
    const s2 = fold(INITIAL_RLM_STATE, ev);
    expect(s1).toEqual(s2);
  });
});
