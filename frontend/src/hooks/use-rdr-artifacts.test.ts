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
        json: () => Promise.resolve(clustersData),
      });
    }
    if (path.includes("/leaf-scores")) {
      return Promise.resolve({
        ok: leafData !== null,
        json: () => Promise.resolve(leafData),
      });
    }
    if (path.includes("/repair-iterations")) {
      return Promise.resolve({
        ok: repairData !== null,
        json: () => Promise.resolve(repairData),
      });
    }
    return Promise.resolve({ ok: false, json: () => Promise.resolve(null) });
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
      Promise.resolve({ ok: false, json: () => Promise.resolve(null) })
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
});
