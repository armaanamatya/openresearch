import { describe, it, expect, vi, afterEach } from "vitest";
import { replayFixture } from "./replay";
import { rlmRunFixture } from "./__fixtures__/rlm-run.fixture";

describe("replayFixture", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

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
  });

  it("timed mode: stop() cancels a live interval", async () => {
    vi.useFakeTimers();
    const onUpdate = vi.fn();
    const stop = replayFixture("timed", onUpdate);
    // Advance two intervals (2 × 150 ms = 300 ms) so the interval has fired
    // at least twice — confirming it is genuinely live.
    await vi.advanceTimersByTimeAsync(300);
    const callsBefore = onUpdate.mock.calls.length;
    expect(callsBefore).toBeGreaterThanOrEqual(2);
    stop();
    // Advance well past the full fixture duration to verify no further calls.
    await vi.advanceTimersByTimeAsync(30_000);
    expect(onUpdate.mock.calls.length).toBe(callsBefore);
  });
});
