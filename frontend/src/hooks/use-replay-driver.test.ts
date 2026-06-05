import { renderHook, act } from "@testing-library/react";
import { useReplayDriver } from "./use-replay-driver";
import type { RlmDashboardEvent } from "@/lib/events/rlm-events";

function mkEvents(n: number): RlmDashboardEvent[] {
  return Array.from({ length: n }, (_, i) => ({
    event: "primitive_call",
    timestamp: new Date(Date.UTC(2026, 0, 1, 0, 0, i)).toISOString(), // 1s apart
    primitive: "understand_section",
    status: "complete",
    iteration: 1,
  })) as unknown as RlmDashboardEvent[];
}

describe("useReplayDriver", () => {
  it("opens at the end (full run shown), paused", () => {
    const { result } = renderHook(() => useReplayDriver(mkEvents(5)));
    expect(result.current.state.index).toBe(5);
    expect(result.current.state.total).toBe(5);
    expect(result.current.state.playing).toBe(false);
    expect(result.current.state.atEnd).toBe(true);
    expect(result.current.events).toHaveLength(5);
  });

  it("seek reveals a prefix slice and pauses", () => {
    const { result } = renderHook(() => useReplayDriver(mkEvents(5)));
    act(() => result.current.seek(2));
    expect(result.current.state.index).toBe(2);
    expect(result.current.events).toHaveLength(2);
    expect(result.current.state.playing).toBe(false);
  });

  it("step clamps to [0, total]", () => {
    const { result } = renderHook(() => useReplayDriver(mkEvents(3)));
    act(() => result.current.seek(0));
    act(() => result.current.step(-1));
    expect(result.current.state.index).toBe(0);
    act(() => result.current.step(99));
    expect(result.current.state.index).toBe(3);
  });

  it("play from the end restarts at 0", () => {
    const { result } = renderHook(() => useReplayDriver(mkEvents(4)));
    expect(result.current.state.atEnd).toBe(true);
    act(() => result.current.play());
    expect(result.current.state.index).toBe(0);
    expect(result.current.state.playing).toBe(true);
  });

  it("play advances the cursor over paced timers", () => {
    vi.useFakeTimers();
    try {
      const { result } = renderHook(() => useReplayDriver(mkEvents(4)));
      act(() => result.current.seek(0));
      act(() => result.current.play());
      expect(result.current.state.index).toBe(0);
      act(() => {
        vi.advanceTimersByTime(6000);
      });
      expect(result.current.state.index).toBeGreaterThanOrEqual(1);
    } finally {
      vi.useRealTimers();
    }
  });

  it("setSpeed updates speed", () => {
    const { result } = renderHook(() => useReplayDriver(mkEvents(3)));
    act(() => result.current.setSpeed(4));
    expect(result.current.state.speed).toBe(4);
  });
});
