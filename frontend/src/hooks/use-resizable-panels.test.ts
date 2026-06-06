import { renderHook, act } from "@testing-library/react";
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { useResizablePanels } from "./use-resizable-panels";

// ── localStorage mock ─────────────────────────────────────────────────────────

const localStorageMock = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: (key: string) => store[key] ?? null,
    setItem: (key: string, value: string) => { store[key] = value; },
    removeItem: (key: string) => { delete store[key]; },
    clear: () => { store = {}; },
  };
})();

// ── matchMedia mock ───────────────────────────────────────────────────────────

type MQListener = (e: { matches: boolean }) => void;
interface MockMQ {
  matches: boolean;
  listeners: MQListener[];
  addEventListener: (evt: string, fn: MQListener) => void;
  removeEventListener: (evt: string, fn: MQListener) => void;
  fire: (matches: boolean) => void;
}

function makeMQ(matches: boolean): MockMQ {
  const mq: MockMQ = {
    matches,
    listeners: [],
    addEventListener(_evt: string, fn: MQListener) { this.listeners.push(fn); },
    removeEventListener(_evt: string, fn: MQListener) {
      this.listeners = this.listeners.filter((l) => l !== fn);
    },
    fire(newMatches: boolean) {
      this.matches = newMatches;
      this.listeners.forEach((fn) => fn({ matches: newMatches }));
    },
  };
  return mq;
}

describe("useResizablePanels", () => {
  let mq1200: MockMQ;
  let mq900: MockMQ;

  beforeEach(() => {
    vi.useFakeTimers();
    Object.defineProperty(window, "localStorage", { value: localStorageMock, writable: true });
    localStorageMock.clear();

    mq1200 = makeMQ(false);
    mq900  = makeMQ(false);
    vi.spyOn(window, "matchMedia").mockImplementation((query: string) => {
      if (query === "(max-width: 1199px)") return mq1200 as unknown as MediaQueryList;
      if (query === "(max-width: 899px)")  return mq900  as unknown as MediaQueryList;
      return { matches: false, addEventListener: () => {}, removeEventListener: () => {} } as unknown as MediaQueryList;
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.useRealTimers();
  });

  // ── Default sizes ───────────────────────────────────────────────────────────

  it("returns default sizes on first render", () => {
    const { result } = renderHook(() => useResizablePanels());
    expect(result.current.sizes.replRail).toBe(240);
    expect(result.current.sizes.reportRail).toBe(280);
    expect(result.current.sizes.detailSidebar).toBe(360);
  });

  // ── setSize bounds enforcement ──────────────────────────────────────────────

  it("clamps replRail below min (180)", () => {
    const { result } = renderHook(() => useResizablePanels());
    act(() => { result.current.setSize("replRail", 50); });
    expect(result.current.sizes.replRail).toBe(180);
  });

  it("clamps replRail above max (360)", () => {
    const { result } = renderHook(() => useResizablePanels());
    act(() => { result.current.setSize("replRail", 999); });
    expect(result.current.sizes.replRail).toBe(360);
  });

  it("accepts valid replRail size", () => {
    const { result } = renderHook(() => useResizablePanels());
    act(() => { result.current.setSize("replRail", 300); });
    expect(result.current.sizes.replRail).toBe(300);
  });

  it("clamps detailSidebar to bounds [280, 520]", () => {
    const { result } = renderHook(() => useResizablePanels());
    act(() => { result.current.setSize("detailSidebar", 100); });
    expect(result.current.sizes.detailSidebar).toBe(280);
    act(() => { result.current.setSize("detailSidebar", 9999); });
    expect(result.current.sizes.detailSidebar).toBe(520);
  });

  // ── localStorage round-trip ─────────────────────────────────────────────────

  it("persists sizes to localStorage after debounce", async () => {
    const { result } = renderHook(() => useResizablePanels());
    act(() => { result.current.setSize("reportRail", 320); });
    // Before debounce fires, localStorage is not updated yet.
    expect(localStorageMock.getItem("openresearch:lab-workspace-sizes:v1")).toBeNull();
    // Advance timers past 200ms debounce.
    act(() => { vi.advanceTimersByTime(250); });
    const stored = JSON.parse(localStorageMock.getItem("openresearch:lab-workspace-sizes:v1") ?? "{}");
    expect(stored.reportRail).toBe(320);
  });

  it("restores sizes from localStorage on mount", async () => {
    localStorageMock.setItem(
      "openresearch:lab-workspace-sizes:v1",
      JSON.stringify({ replRail: 300, reportRail: 250, detailSidebar: 400 })
    );
    const { result } = renderHook(() => useResizablePanels());
    await act(async () => {
      vi.runAllTicks();
    });
    expect(result.current.sizes.replRail).toBe(300);
    expect(result.current.sizes.reportRail).toBe(250);
    expect(result.current.sizes.detailSidebar).toBe(400);
  });

  it("clamps out-of-bounds localStorage values on restore", async () => {
    localStorageMock.setItem(
      "openresearch:lab-workspace-sizes:v1",
      JSON.stringify({ replRail: 10, reportRail: 9000, detailSidebar: 360 })
    );
    const { result } = renderHook(() => useResizablePanels());
    await act(async () => {
      vi.runAllTicks();
    });
    expect(result.current.sizes.replRail).toBe(180);
    expect(result.current.sizes.reportRail).toBe(360);
  });

  it("handles corrupt localStorage gracefully", () => {
    localStorageMock.setItem("openresearch:lab-workspace-sizes:v1", "{{invalid json}}");
    const { result } = renderHook(() => useResizablePanels());
    act(() => {});
    // Falls back to defaults without throwing.
    expect(result.current.sizes.replRail).toBe(240);
  });

  // ── Viewport collapse via matchMedia ────────────────────────────────────────

  it("collapsedByViewport.replRail is false by default", () => {
    const { result } = renderHook(() => useResizablePanels());
    act(() => {});
    expect(result.current.collapsedByViewport.replRail).toBe(false);
  });

  it("collapses replRail when width < 1200px", () => {
    mq1200 = makeMQ(true);
    vi.spyOn(window, "matchMedia").mockImplementation((query: string) => {
      if (query === "(max-width: 1199px)") return mq1200 as unknown as MediaQueryList;
      if (query === "(max-width: 899px)")  return mq900  as unknown as MediaQueryList;
      return { matches: false, addEventListener: () => {}, removeEventListener: () => {} } as unknown as MediaQueryList;
    });
    const { result } = renderHook(() => useResizablePanels());
    act(() => {});
    expect(result.current.collapsedByViewport.replRail).toBe(true);
  });

  it("collapses reportRail when width < 900px", () => {
    mq900 = makeMQ(true);
    vi.spyOn(window, "matchMedia").mockImplementation((query: string) => {
      if (query === "(max-width: 1199px)") return mq1200 as unknown as MediaQueryList;
      if (query === "(max-width: 899px)")  return mq900  as unknown as MediaQueryList;
      return { matches: false, addEventListener: () => {}, removeEventListener: () => {} } as unknown as MediaQueryList;
    });
    const { result } = renderHook(() => useResizablePanels());
    act(() => {});
    expect(result.current.collapsedByViewport.reportRail).toBe(true);
  });

  it("updates collapse flags reactively when matchMedia fires", () => {
    const { result } = renderHook(() => useResizablePanels());
    act(() => {});
    expect(result.current.collapsedByViewport.replRail).toBe(false);
    act(() => { mq1200.fire(true); });
    expect(result.current.collapsedByViewport.replRail).toBe(true);
    act(() => { mq1200.fire(false); });
    expect(result.current.collapsedByViewport.replRail).toBe(false);
  });
});
