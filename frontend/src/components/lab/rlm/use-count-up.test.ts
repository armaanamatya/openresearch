import { act, renderHook } from "@testing-library/react";
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { useCountUp } from "./use-count-up";

describe("useCountUp", () => {
  let rafCallbacks: Map<number, FrameRequestCallback>;
  let rafId: number;
  let nowMs: number;

  beforeEach(() => {
    rafCallbacks = new Map();
    rafId = 0;
    nowMs = 0;
    vi.stubGlobal("requestAnimationFrame", (cb: FrameRequestCallback) => {
      rafId += 1;
      rafCallbacks.set(rafId, cb);
      return rafId;
    });
    vi.stubGlobal("cancelAnimationFrame", (id: number) => {
      rafCallbacks.delete(id);
    });
    vi.stubGlobal("performance", { now: () => nowMs });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  function tick(deltaMs: number) {
    nowMs += deltaMs;
    const cbs = Array.from(rafCallbacks.values());
    rafCallbacks.clear();
    cbs.forEach((cb) => cb(nowMs));
  }

  it("returns initial target on first render", () => {
    const { result } = renderHook(() => useCountUp(0.5, 400));
    expect(result.current).toBe(0.5);
  });

  it("eases from start to target over the duration on a retarget", () => {
    const { result, rerender } = renderHook(({ target }) => useCountUp(target, 400), {
      initialProps: { target: 0 },
    });
    rerender({ target: 1.0 });

    // After ~half the duration, we should be > 0 but < 1.
    act(() => tick(200));
    expect(result.current).toBeGreaterThan(0);
    expect(result.current).toBeLessThan(1);

    // After full duration, we should land at 1.
    act(() => tick(400));
    expect(result.current).toBeCloseTo(1.0, 5);
  });

  it("cancels and restarts on target change mid-tween", () => {
    const { result, rerender } = renderHook(({ target }) => useCountUp(target, 400), {
      initialProps: { target: 0 },
    });

    rerender({ target: 1.0 });
    act(() => tick(100));
    const midValue = result.current;
    expect(midValue).toBeGreaterThan(0);

    // Retarget to 0.5 — restart from midValue.
    rerender({ target: 0.5 });
    act(() => tick(400));
    expect(result.current).toBeCloseTo(0.5, 5);
  });
});
