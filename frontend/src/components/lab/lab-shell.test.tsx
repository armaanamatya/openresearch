import { fireEvent, render, screen, waitFor } from "@testing-library/react";
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

  it("starts an uploaded paper run through the backend and opens the live event stream", async () => {
    const instances: Array<{ url: string }> = [];
    class FakeEventSource {
      url: string;

      constructor(url: string) {
        this.url = url;
        instances.push(this);
      }

      addEventListener = vi.fn();
      close = vi.fn();
    }

    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          projectId: "ui_sdk_uploaded_demo_1",
          outputDir: "runs/ui_sdk_uploaded_demo_1",
          runMode: "sdk",
          llmProvider: "anthropic",
          sourceKind: "uploaded_pdf",
          sourceLabel: "paper.pdf",
          sourceNote: "uploaded source",
          status: "running",
          payload: null,
          log: ""
        })
      });

    vi.stubGlobal("fetch", fetchMock);
    vi.stubGlobal("EventSource", FakeEventSource);

    render(<LabShell />);

    const file = new File(["%PDF-demo"], "paper.pdf", { type: "application/pdf" });
    fireEvent.change(screen.getByLabelText(/upload paper pdf/i), {
      target: { files: [file] }
    });

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));

    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("/api/demo");
    expect(init.method).toBe("POST");
    expect(init.body).toBeInstanceOf(FormData);
    await waitFor(() => {
      expect(instances[0]?.url).toBe("/api/demo/events?projectId=ui_sdk_uploaded_demo_1");
    });
  });

  it("does not restore persisted runs without an explicit initial run", () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    render(<LabShell />);

    expect(screen.getByRole("heading", { name: "Upload PDF" })).toBeInTheDocument();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("renders RlmLab when runMode is rlm", () => {
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
    // RlmLab exposes a stable test id; the existing 14-stage LabCanvas does not.
    expect(screen.getByTestId("rlm-lab")).toBeInTheDocument();
    // The 14-stage workflow header must NOT be present.
    expect(screen.queryByText(/agents complete/i)).not.toBeInTheDocument();
  });

  it("renders the non-rlm placeholder for sdk/offline runs and does not render RlmLab", () => {
    render(
      <LabShell
        initialRun={{
          projectId: "prj_sdk_test",
          outputDir: "runs/prj_sdk_test",
          runMode: "sdk",
          status: "running",
          sourceKind: "uploaded_pdf",
          sourceLabel: "sdk-paper.pdf",
          sourceNote: "sdk mode run",
          payload: null,
          log: ""
        }}
      />
    );
    // The honest placeholder must be shown for non-rlm runs.
    expect(screen.getByTestId("non-rlm-notice")).toBeInTheDocument();
    expect(screen.getByText(/retired.*pipeline/i)).toBeInTheDocument();
    // RlmLab must NOT be rendered.
    expect(screen.queryByTestId("rlm-lab")).toBeNull();
  });
});
