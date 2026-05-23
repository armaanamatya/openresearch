import { render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";

import { LabShell } from "./lab-shell";

// useRun calls useRouter() (to keep ?projectId= in sync with the active
// run). jsdom doesn't mount Next's app-router context, so we stub it.
const routerReplaceMock = vi.fn();
vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: routerReplaceMock, push: vi.fn(), prefetch: vi.fn() }),
  // useSearchParams is used by the ?rlmFixture=1 dev path; return null params
  // (fixture mode off) in all existing tests.
  useSearchParams: () => ({ get: () => null })
}));

describe("LabShell", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    // useRun persists the active run's projectId to localStorage so a
    // refresh can auto-resume it. Without clearing between tests, the
    // previous test's projectId leaks and the next mount fires a
    // spurious GET /api/demo?projectId=… which breaks fetch-call-count
    // assertions and the "does not restore persisted" test below.
    window.localStorage.clear();
  });

  it("does not restore persisted runs without an explicit initial run", () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    render(<LabShell />);

    expect(screen.getByRole("heading", { name: "Upload PDF" })).toBeInTheDocument();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("renders RlmLab unconditionally when a run is active", () => {
    render(
      <LabShell
        initialRun={{
          projectId: "prj_rlm_test",
          outputDir: "runs/prj_rlm_test",
          runMode: "rlm" as import("@/lib/demo/demo-run-types").DemoRunMode,
          status: "running",
          sourceKind: "uploaded_pdf",
          sourceLabel: "Attention is all you need",
          sourceNote: "rlm mode run",
          payload: null,
          log: ""
        }}
      />
    );
    // RlmLab exposes a stable test id.
    expect(screen.getByTestId("rlm-lab")).toBeInTheDocument();
  });
});
