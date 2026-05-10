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

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));

    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
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

  it("does not restore persisted runs without an explicit initial run", () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    render(<ReproLabClient />);

    expect(screen.getByRole("heading", { name: "Upload PDF" })).toBeInTheDocument();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("renders dashboard events as live activity and maps completed agents to graph progress", () => {
    vi.stubGlobal("fetch", vi.fn());

    render(
      <ReproLabClient
        initialRun={{
          projectId: "event-backed-run",
          outputDir: "runs/event-backed-run",
          runMode: "sdk",
          llmProvider: "anthropic",
          sourceKind: "uploaded_pdf",
          sourceLabel: "paper.pdf",
          sourceNote: "uploaded",
          status: "running",
          payload: {
            projectId: "event-backed-run",
            outputDir: "runs/event-backed-run",
            sourceKind: "uploaded_pdf",
            runMode: "sdk",
            sourceLabel: "paper.pdf",
            sourceNote: "uploaded",
            generatedAt: "2026-05-10T10:00:00.000Z",
            log: "",
            summary: {
              stage: "artifacts_discovered",
              meanReward: null,
              improvementCount: 0,
              runModeLabel: "SDK: Anthropic",
              llmProvider: "anthropic",
              sourceLabel: "paper.pdf"
            },
            initialSnapshot: {
              agents: [],
              reasoning: [],
              messages: [],
              citations: [],
              approvals: [],
              progress: [],
              dataPanels: [],
              hermesPanel: null,
              conceptCard: null
            },
            events: [
              {
                event: "agent_completed",
                timestamp: "2026-05-10T10:00:01.000Z",
                agentId: "paper-understanding",
                agent: {
                  id: "paper-understanding",
                  label: "Paper Understanding",
                  type: "builder",
                  status: "completed",
                  currentTask: "Claim map published",
                  lastUpdated: "2026-05-10T10:00:01.000Z",
                  outputTargetIds: ["artifact-discovery"],
                  contextVariables: ["paper_claim_map"]
                }
              },
              {
                event: "agent_started",
                timestamp: "2026-05-10T10:00:02.000Z",
                agentId: "artifact-discovery",
                agent: {
                  id: "artifact-discovery",
                  label: "Artifact Discovery",
                  type: "builder",
                  status: "running",
                  currentTask: "Searching repositories",
                  lastUpdated: "2026-05-10T10:00:02.000Z",
                  outputTargetIds: ["environment-detective"],
                  contextVariables: ["paper_claim_map"]
                }
              }
            ]
          },
          log: ""
        }}
      />
    );

    expect(screen.getByText("Claim map published")).toBeInTheDocument();
    expect(screen.getAllByText("Searching repositories").length).toBeGreaterThan(0);
    expect(screen.getByText(/2\/12 agents complete/i)).toBeInTheDocument();
  });

  it("maps canonical backend pipeline stages without leaving paper understanding stuck", () => {
    vi.stubGlobal("fetch", vi.fn());

    render(
      <ReproLabClient
        initialRun={{
          projectId: "stage-backed-run",
          outputDir: "runs/stage-backed-run",
          runMode: "sdk",
          llmProvider: "anthropic",
          sourceKind: "uploaded_pdf",
          sourceLabel: "paper.pdf",
          sourceNote: "uploaded",
          status: "running",
          payload: {
            projectId: "stage-backed-run",
            outputDir: "runs/stage-backed-run",
            sourceKind: "uploaded_pdf",
            runMode: "sdk",
            sourceLabel: "paper.pdf",
            sourceNote: "uploaded",
            generatedAt: "2026-05-10T10:00:00.000Z",
            log: "",
            summary: {
              stage: "paper_understood",
              meanReward: null,
              improvementCount: 0,
              runModeLabel: "SDK: Anthropic",
              llmProvider: "anthropic",
              sourceLabel: "paper.pdf"
            },
            initialSnapshot: {
              agents: [],
              reasoning: [],
              messages: [],
              citations: [],
              approvals: [],
              progress: [],
              dataPanels: [],
              hermesPanel: null,
              conceptCard: null
            },
            events: []
          },
          log: ""
        }}
      />
    );

    expect(screen.getByText(/2\/12 agents complete/i)).toBeInTheDocument();
  });

  it("shows the exact failing agent without marking completed nodes as failed", () => {
    vi.stubGlobal("fetch", vi.fn());

    render(
      <ReproLabClient
        initialRun={{
          projectId: "failed-run",
          outputDir: "runs/failed-run",
          runMode: "sdk",
          llmProvider: "anthropic",
          sourceKind: "uploaded_pdf",
          sourceLabel: "paper.pdf",
          sourceNote: "uploaded",
          status: "failed",
          error: "No JSON found in agent output",
          payload: {
            projectId: "failed-run",
            outputDir: "runs/failed-run",
            sourceKind: "uploaded_pdf",
            runMode: "sdk",
            sourceLabel: "paper.pdf",
            sourceNote: "uploaded",
            generatedAt: "2026-05-10T10:00:00.000Z",
            log: "",
            summary: {
              stage: "ingested",
              meanReward: null,
              improvementCount: 0,
              runModeLabel: "SDK: Anthropic",
              llmProvider: "anthropic",
              sourceLabel: "paper.pdf"
            },
            initialSnapshot: {
              agents: [],
              reasoning: [],
              messages: [],
              citations: [],
              approvals: [],
              progress: [],
              dataPanels: [],
              hermesPanel: null,
              conceptCard: null
            },
            events: [
              {
                event: "agent_failed",
                timestamp: "2026-05-10T10:00:01.000Z",
                agentId: "paper-understanding",
                agent: {
                  id: "paper-understanding",
                  label: "Paper Understanding",
                  type: "builder",
                  status: "failed",
                  currentTask: "No JSON found in agent output",
                  lastUpdated: "2026-05-10T10:00:01.000Z",
                  outputTargetIds: ["root-orchestrator"],
                  contextVariables: ["paper_claim_map"]
                }
              }
            ]
          },
          log: ""
        }}
      />
    );

    expect(screen.getAllByText("No JSON found in agent output").length).toBeGreaterThan(0);
    expect(screen.getByText("Failed")).toBeInTheDocument();
  });
});
