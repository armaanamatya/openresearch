import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { LiveDemoClient } from "./live-demo-client";

vi.mock("@/features/dashboard/dashboard-shell", () => ({
  DashboardShell: () => <div>dashboard shell</div>
}));

describe("LiveDemoClient", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("uses the automatic Docker-first sandbox by default", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        projectId: "ui_sdk_demo_1",
        outputDir: "runs/ui_sdk_demo_1",
        runMode: "sdk",
        llmProvider: "anthropic",
        executionMode: "efficient",
        sandboxMode: "auto",
        status: "queued",
        payload: null,
        log: ""
      })
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<LiveDemoClient initialRun={null} />);

    fireEvent.click(screen.getByRole("button", { name: /run sdk/i }));

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        "/api/demo?mode=sdk&provider=anthropic&executionMode=efficient&sandbox=auto&gpuMode=auto",
        {
          method: "POST"
        }
      );
    });
  });

  it("passes the selected SDK provider when starting an SDK run", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        projectId: "ui_sdk_openai_demo_1",
        outputDir: "runs/ui_sdk_openai_demo_1",
        runMode: "sdk",
        llmProvider: "openai",
        status: "queued",
        payload: null,
        log: ""
      })
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<LiveDemoClient initialRun={null} />);

    fireEvent.change(screen.getByLabelText(/sdk provider/i), {
      target: { value: "openai" }
    });
    fireEvent.change(screen.getByLabelText(/profile/i), {
      target: { value: "max" }
    });
    fireEvent.change(screen.getByLabelText(/sandbox/i), {
      target: { value: "docker" }
    });
    fireEvent.change(screen.getByLabelText(/review sdk/i), {
      target: { value: "anthropic" }
    });
    fireEvent.change(screen.getByLabelText(/compute/i), {
      target: { value: "prefer" }
    });
    fireEvent.click(screen.getByRole("button", { name: /run sdk/i }));

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        "/api/demo?mode=sdk&provider=openai&verificationProvider=anthropic&executionMode=max&sandbox=docker&gpuMode=prefer",
        {
          method: "POST"
        }
      );
    });
  });

  it("shows the backend setup error when Docker preflight fails", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: false,
      status: 503,
      json: async () => ({
        error: "Docker sandbox is not ready for the Python backend."
      })
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<LiveDemoClient initialRun={null} />);

    fireEvent.change(screen.getByLabelText(/sandbox/i), {
      target: { value: "docker" }
    });
    fireEvent.click(screen.getByRole("button", { name: /run sdk/i }));

    expect(
      await screen.findByText("Docker sandbox is not ready for the Python backend.")
    ).toBeInTheDocument();
  });

  it("shows the selected pdf name and posts provider form data for uploaded sdk runs", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        projectId: "prj_upload",
        outputDir: "runs/prj_upload",
        runMode: "sdk",
        llmProvider: "openai",
        status: "queued",
        payload: null,
        log: ""
      })
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<LiveDemoClient initialRun={null} />);

    fireEvent.change(screen.getByLabelText(/sdk provider/i), {
      target: { value: "openai" }
    });
    fireEvent.change(screen.getByLabelText(/profile/i), {
      target: { value: "max" }
    });
    fireEvent.change(screen.getByLabelText(/sandbox/i), {
      target: { value: "docker" }
    });

    const input = screen.getByLabelText("Upload paper PDF");
    const file = new File(["%PDF-demo"], "ppo-paper.pdf", {
      type: "application/pdf"
    });

    fireEvent.change(input, { target: { files: [file] } });

    expect(screen.getByText("ppo-paper.pdf")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Run uploaded paper (SDK)" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));

    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("/api/demo");
    expect(init.method).toBe("POST");
    expect(init.body).toBeInstanceOf(FormData);

    const formData = init.body as FormData;
    expect(formData.get("mode")).toBe("sdk");
    expect(formData.get("provider")).toBe("openai");
    expect(formData.get("verificationProvider")).toBe("same");
    expect(formData.get("executionMode")).toBe("max");
    expect(formData.get("sandbox")).toBe("docker");
    expect(formData.get("gpuMode")).toBe("auto");
    expect(formData.get("paper")).toBe(file);
  });

  it("opens a server-sent event stream for active runs", async () => {
    const instances: Array<{ url: string; close: ReturnType<typeof vi.fn> }> = [];
    class FakeEventSource {
      url: string;
      close = vi.fn();

      constructor(url: string) {
        this.url = url;
        instances.push(this);
      }

      addEventListener = vi.fn();
    }
    vi.stubGlobal("EventSource", FakeEventSource);

    render(
      <LiveDemoClient
        initialRun={{
          projectId: "prj_live",
          outputDir: "runs/prj_live",
          runMode: "sdk",
          llmProvider: "anthropic",
          status: "running",
          payload: null,
          log: ""
        }}
      />
    );

    await waitFor(() => {
      expect(instances[0]?.url).toBe("/api/demo/events?projectId=prj_live");
    });
  });
});
