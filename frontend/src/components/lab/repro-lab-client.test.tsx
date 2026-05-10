import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";

import { ReproLabClient } from "./repro-lab-client";

describe("ReproLabClient", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("starts a backend run from the arxiv form and transitions into the workflow view", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce({
        ok: true,
        json: async () => null
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          projectId: "ui_sdk_fixture_demo_1",
          outputDir: "runs/ui_sdk_fixture_demo_1",
          runMode: "sdk",
          llmProvider: "anthropic",
          sourceKind: "workspace_fixture",
          sourceLabel: "In-repo PPO workspace fixture",
          sourceNote: "fixture",
          status: "queued",
          payload: null,
          log: ""
        })
      });
    vi.stubGlobal("fetch", fetchMock);

    render(<ReproLabClient />);

    expect(screen.getByRole("heading", { name: "Upload PDF" })).toBeInTheDocument();
    expect(screen.getByPlaceholderText("arxiv.org/abs/2303.04137")).toBeInTheDocument();

    fireEvent.change(screen.getByPlaceholderText("arxiv.org/abs/2303.04137"), {
      target: { value: "arxiv.org/abs/2303.04137" }
    });
    fireEvent.click(screen.getByRole("button", { name: /begin/i }));

    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(
        "/api/demo?mode=sdk&provider=anthropic&executionMode=efficient&sandbox=runpod&gpuMode=auto",
        { method: "POST" }
      )
    );

    expect(
      await screen.findByRole("heading", { name: /in-repo ppo workspace fixture/i })
    ).toBeInTheDocument();
    expect(screen.getByText(/agents complete/i)).toBeInTheDocument();
    expect(screen.getByText("Live activity")).toBeInTheDocument();
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
        json: async () => null
      })
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

    render(<ReproLabClient />);

    const file = new File(["%PDF-demo"], "paper.pdf", { type: "application/pdf" });
    fireEvent.change(screen.getByLabelText(/upload paper pdf/i), {
      target: { files: [file] }
    });

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));

    const [url, init] = fetchMock.mock.calls[1] as [string, RequestInit];
    expect(url).toBe("/api/demo");
    expect(init.method).toBe("POST");
    expect(init.body).toBeInstanceOf(FormData);
    await waitFor(() => {
      expect(instances[0]?.url).toBe("/api/demo/events?projectId=ui_sdk_uploaded_demo_1");
    });
  });

  it("returns to the upload screen when the ReproLab brand is clicked from a run", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        projectId: "still-latest-run",
        outputDir: "runs/still-latest-run",
        runMode: "sdk",
        llmProvider: "anthropic",
        sourceKind: "workspace_fixture",
        sourceLabel: "Still latest run",
        sourceNote: "latest",
        status: "running",
        payload: null,
        log: ""
      })
    });
    vi.stubGlobal("fetch", fetchMock);

    render(
      <ReproLabClient
        initialRun={{
          projectId: "active-run",
          outputDir: "runs/active-run",
          runMode: "sdk",
          llmProvider: "anthropic",
          sourceKind: "uploaded_pdf",
          sourceLabel: "paper.pdf",
          sourceNote: "uploaded",
          status: "running",
          payload: null,
          log: ""
        }}
      />
    );

    expect(screen.getByRole("heading", { name: /paper\.pdf/i })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /^reprolab$/i }));

    expect(await screen.findByRole("heading", { name: "Upload PDF" })).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: /still latest run/i })).not.toBeInTheDocument();
  });
});
