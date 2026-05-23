import { renderHook, act } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { useRdrArtifacts } from "./use-rdr-artifacts";
import type { DemoClusterStatus, DemoLeafScore, DemoRepairPass } from "@/lib/demo/demo-run-types";

const CLUSTER: DemoClusterStatus = {
  index: 0,
  cluster_id: "clus-1",
  title: "Introduction",
  leaf_ids: ["leaf-a", "leaf-b"],
  failed: false,
  file_count: 3,
  repair_history: [],
};

const LEAF: DemoLeafScore = {
  id: "leaf-a",
  score: 0.75,
  justification: "Correct implementation",
};

const REPAIR_PASS: DemoRepairPass = {
  pass: 1,
  cluster_count: 2,
  failed_count: 1,
};

function makeFetch(
  clustersData: unknown = null,
  leafData: unknown = null,
  repairData: unknown = null
) {
  return vi.fn((url: string | URL | Request) => {
    const path = typeof url === "string" ? url : url.toString();
    if (path.includes("/clusters")) {
      return Promise.resolve({
        ok: clustersData !== null,
        status: clustersData !== null ? 200 : 404,
        json: () => Promise.resolve(clustersData),
      });
    }
    if (path.includes("/leaf-scores")) {
      return Promise.resolve({
        ok: leafData !== null,
        status: leafData !== null ? 200 : 404,
        json: () => Promise.resolve(leafData),
      });
    }
    if (path.includes("/repair-iterations")) {
      return Promise.resolve({
        ok: repairData !== null,
        status: repairData !== null ? 200 : 404,
        json: () => Promise.resolve(repairData),
      });
    }
    return Promise.resolve({ ok: false, status: 404, json: () => Promise.resolve(null) });
  });
}

describe("useRdrArtifacts", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.useRealTimers();
  });

  it("returns empty arrays when projectId is null", () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    const { result } = renderHook(() => useRdrArtifacts(null, false));
    expect(result.current.clusters).toEqual([]);
    expect(result.current.leafScores).toEqual([]);
    expect(result.current.repairPasses).toEqual([]);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("fetches all three endpoints immediately on mount", async () => {
    const fetchMock = makeFetch(
      { clusters: [CLUSTER] },
      { leaf_scores: [LEAF] },
      { passes: [REPAIR_PASS] }
    );
    vi.stubGlobal("fetch", fetchMock);

    const { result } = renderHook(() => useRdrArtifacts("prj_test", false));

    // Allow microtasks to settle
    await act(async () => {});

    expect(result.current.clusters).toEqual([CLUSTER]);
    expect(result.current.leafScores).toEqual([LEAF]);
    expect(result.current.repairPasses).toEqual([REPAIR_PASS]);
    expect(fetchMock).toHaveBeenCalledTimes(3);
  });

  it("does not set state when a fetch returns non-ok (404)", async () => {
    const fetchMock = vi.fn(() =>
      Promise.resolve({ ok: false, status: 404, json: () => Promise.resolve(null) })
    );
    vi.stubGlobal("fetch", fetchMock);

    const { result } = renderHook(() => useRdrArtifacts("prj_404", false));
    await act(async () => {});

    expect(result.current.clusters).toEqual([]);
    expect(result.current.leafScores).toEqual([]);
    expect(result.current.repairPasses).toEqual([]);
  });

  it("polls at the specified interval when isActive is true", async () => {
    vi.useFakeTimers();

    const fetchMock = makeFetch(
      { clusters: [CLUSTER] },
      { leaf_scores: [LEAF] },
      { passes: [] }
    );
    vi.stubGlobal("fetch", fetchMock);

    renderHook(() => useRdrArtifacts("prj_active", true, 1000));

    // Initial fetch = 3 calls
    await act(async () => { await Promise.resolve(); });
    expect(fetchMock).toHaveBeenCalledTimes(3);

    // Advance one interval
    await act(async () => { vi.advanceTimersByTime(1000); await Promise.resolve(); });
    expect(fetchMock).toHaveBeenCalledTimes(6);
  });

  it("does not poll when isActive is false", async () => {
    vi.useFakeTimers();

    const fetchMock = makeFetch({ clusters: [CLUSTER] }, null, null);
    vi.stubGlobal("fetch", fetchMock);

    renderHook(() => useRdrArtifacts("prj_inactive", false, 1000));

    await act(async () => { await Promise.resolve(); });
    const callsAfterMount = fetchMock.mock.calls.length;

    await act(async () => { vi.advanceTimersByTime(5000); await Promise.resolve(); });
    // No additional calls beyond the initial fetch
    expect(fetchMock.mock.calls.length).toBe(callsAfterMount);
  });

  it("resets to empty and stops polling when projectId becomes null", async () => {
    vi.useFakeTimers();
    const fetchMock = makeFetch({ clusters: [CLUSTER] }, null, null);
    vi.stubGlobal("fetch", fetchMock);

    const { result, rerender } = renderHook(
      ({ pid, active }: { pid: string | null; active: boolean }) =>
        useRdrArtifacts(pid, active, 1000),
      { initialProps: { pid: "prj_switch" as string | null, active: true } }
    );

    await act(async () => { await Promise.resolve(); });
    expect(result.current.clusters.length).toBeGreaterThan(0);

    rerender({ pid: null, active: false });
    expect(result.current.clusters).toEqual([]);
    expect(result.current.leafScores).toEqual([]);
    expect(result.current.repairPasses).toEqual([]);
  });

  it("test_hook_stops_polling_after_3_404_cycles: stops after 3 all-404 cycles and exposes noRdrArtifacts", async () => {
    vi.useFakeTimers();

    // All 3 endpoints return 404 on every call.
    const fetchMock = vi.fn(() =>
      Promise.resolve({ ok: false, status: 404, json: () => Promise.resolve(null) })
    );
    vi.stubGlobal("fetch", fetchMock);

    const { result } = renderHook(() => useRdrArtifacts("prj_all404", true, 10));

    // Cycle 1: initial fetch (3 calls)
    await act(async () => { await Promise.resolve(); });
    // Cycle 2
    await act(async () => { vi.advanceTimersByTime(10); await Promise.resolve(); });
    // Cycle 3
    await act(async () => { vi.advanceTimersByTime(10); await Promise.resolve(); });

    // After 3 cycles, noRdrArtifacts should be true and fetch count exactly 9.
    expect(fetchMock).toHaveBeenCalledTimes(9);
    expect(result.current.noRdrArtifacts).toBe(true);

    // Advance further — no more fetches should fire.
    await act(async () => { vi.advanceTimersByTime(50); await Promise.resolve(); });
    expect(fetchMock).toHaveBeenCalledTimes(9);
  });

  it("test_hook_keeps_polling_when_data_arrives: resets counter and continues when data appears", async () => {
    vi.useFakeTimers();

    let callCount = 0;
    const fetchMock = vi.fn((url: string | URL | Request) => {
      const path = typeof url === "string" ? url : url.toString();
      // First 6 calls (2 full cycles × 3 endpoints) return 404, then return data.
      callCount += 1;
      if (callCount <= 6) {
        return Promise.resolve({ ok: false, status: 404, json: () => Promise.resolve(null) });
      }
      // Cycle 3 onwards: return real data per endpoint.
      if (path.includes("/clusters")) {
        return Promise.resolve({
          ok: true,
          status: 200,
          json: () => Promise.resolve({ clusters: [CLUSTER] }),
        });
      }
      if (path.includes("/leaf-scores")) {
        return Promise.resolve({
          ok: true,
          status: 200,
          json: () => Promise.resolve({ leaf_scores: [LEAF] }),
        });
      }
      return Promise.resolve({
        ok: true,
        status: 200,
        json: () => Promise.resolve({ passes: [] }),
      });
    });
    vi.stubGlobal("fetch", fetchMock);

    const { result } = renderHook(() => useRdrArtifacts("prj_late_data", true, 10));

    // Cycle 1 (all 404)
    await act(async () => { await Promise.resolve(); });
    // Cycle 2 (all 404 — counter = 2, not yet stopped)
    await act(async () => { vi.advanceTimersByTime(10); await Promise.resolve(); });
    // Cycle 3 (data arrives — counter resets to 0)
    await act(async () => { vi.advanceTimersByTime(10); await Promise.resolve(); });

    expect(result.current.noRdrArtifacts).toBe(false);
    expect(result.current.clusters).toEqual([CLUSTER]);
    expect(result.current.leafScores).toEqual([LEAF]);

    // Polling should continue after data arrives — advance one more interval.
    const callsBeforeExtra = fetchMock.mock.calls.length;
    await act(async () => { vi.advanceTimersByTime(10); await Promise.resolve(); });
    expect(fetchMock.mock.calls.length).toBeGreaterThan(callsBeforeExtra);
  });

  it("test_hook_handles_mixed_404_and_data: continues polling when at least one endpoint returns data", async () => {
    vi.useFakeTimers();

    // /leaf-scores returns 200; the other two return 404.
    const fetchMock = vi.fn((url: string | URL | Request) => {
      const path = typeof url === "string" ? url : url.toString();
      if (path.includes("/leaf-scores")) {
        return Promise.resolve({
          ok: true,
          status: 200,
          json: () => Promise.resolve({ leaf_scores: [LEAF] }),
        });
      }
      return Promise.resolve({ ok: false, status: 404, json: () => Promise.resolve(null) });
    });
    vi.stubGlobal("fetch", fetchMock);

    const { result } = renderHook(() => useRdrArtifacts("prj_mixed", true, 10));

    await act(async () => { await Promise.resolve(); });

    // noRdrArtifacts must remain false — not all 3 returned 404.
    expect(result.current.noRdrArtifacts).toBe(false);
    expect(result.current.leafScores).toEqual([LEAF]);

    // Polling continues — advance 3 intervals and confirm more fetches happen.
    const callsBefore = fetchMock.mock.calls.length;
    await act(async () => { vi.advanceTimersByTime(30); await Promise.resolve(); });
    expect(fetchMock.mock.calls.length).toBeGreaterThan(callsBefore);
  });
});
