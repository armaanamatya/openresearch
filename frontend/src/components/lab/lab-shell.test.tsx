import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";

import type { LiveDemoRunState } from "@/lib/demo/demo-run-types";
import { defaultTopologyFixture } from "@/lib/pipeline/__fixtures__/default-topology";

import { LabShell, stateMapForRun } from "./lab-shell";

// useRun calls useRouter() (to keep ?projectId= in sync with the active
// run). jsdom doesn't mount Next's app-router context, so we stub it.
const routerReplaceMock = vi.fn();
vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: routerReplaceMock, push: vi.fn(), prefetch: vi.fn() })
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

    render(<LabShell initialTopology={defaultTopologyFixture} />);

    expect(screen.getByRole("heading", { name: "Upload PDF" })).toBeInTheDocument();
    expect(screen.getByPlaceholderText("arxiv.org/abs/2303.04137")).toBeInTheDocument();

    fireEvent.change(screen.getByPlaceholderText("arxiv.org/abs/2303.04137"), {
      target: { value: "arxiv.org/abs/2303.04137" }
    });
    fireEvent.click(screen.getByRole("button", { name: /begin/i }));

    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(
        "/api/demo?mode=sdk&provider=anthropic&executionMode=efficient&sandbox=runpod&gpuMode=auto&model=sonnet",
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

    render(<LabShell initialTopology={defaultTopologyFixture} />);

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
      <LabShell
        initialTopology={defaultTopologyFixture}
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

  it("renders the stored PDF and PaperBench-style benchmark in the script panel", () => {
    render(
      <LabShell
        initialTopology={defaultTopologyFixture}
        initialRun={{
          projectId: "prj_demo",
          outputDir: "runs/prj_demo",
          runMode: "sdk",
          llmProvider: "anthropic",
          sourceKind: "workspace_fixture",
          sourceLabel: "ReproLab PPO demo paper",
          sourceNote: "demo",
          sourcePdf: {
            fileName: "reprolab-demo-paper.pdf",
            title: "ReproLab PPO Reproducibility Demo",
            sizeBytes: 2048,
            sha256: "abcdef1234567890",
            pageCount: 6,
            runPath: "runs/prj_demo/raw_paper.pdf",
            codePath: "runs/prj_demo/code/paper.pdf"
          },
          benchmark: {
            benchmarkName: "PaperBench-style final benchmark",
            paperbenchTaskId: "reprolab-demo/ppo-cartpole-v1",
            overallScore: 91.4,
            targetMetric: "mean_reward",
            targetValue: 475,
            reproducedValue: 492.3,
            deltaValue: 17.3,
            verdict: "reproduced_with_caveats",
            reportPath: "runs/prj_demo/code/final_benchmark_report.md",
            comparisonPath: "runs/prj_demo/code/paperbench_comparison.json",
            logPath: "runs/prj_demo/code/logs/paperbench_eval.log"
          },
          status: "completed",
          payload: null,
          log: ""
        }}
      />
    );

    expect(screen.getByText("Script panel")).toBeInTheDocument();
    expect(screen.getByText("ReproLab PPO Reproducibility Demo")).toBeInTheDocument();
    expect(screen.getByText("91.4%")).toBeInTheDocument();
    expect(screen.getByText("runs/prj_demo/code/paper.pdf")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Preview PDF" })).toHaveAttribute(
      "href",
      "/api/demo/source-pdf?projectId=prj_demo"
    );
    expect(screen.getByRole("link", { name: /open final report/i })).toHaveAttribute(
      "href",
      "/api/demo/final-report?projectId=prj_demo"
    );

    // After Task C.1.2 the side-panel title also uses the internal-mode
    // label, so all four agent-label render sites are consistent on /lab.
    const reportNodeLabel = screen
      .getAllByText("report-generator")
      .find((el) => el.className === "node-agent");
    expect(reportNodeLabel).toBeDefined();
    fireEvent.click(reportNodeLabel!);

    expect(screen.getByText("report-generator activity")).toBeInTheDocument();
    expect(screen.getByText("Source PDF and final benchmark")).toBeInTheDocument();
  });

  it("uses needs-attention language for interrupted run states", () => {
    // Log lines render verbatim — issueText only euphemises status labels
    // and the surfaced `error` field, not the raw log stream. Use a log
    // line free of "failed" so the assertion checks the structural UI
    // (status pill + Issue section) rather than the log-line passthrough.
    render(
      <LabShell
        initialTopology={defaultTopologyFixture}
        initialRun={{
          projectId: "prj_attention",
          outputDir: "runs/prj_attention",
          runMode: "sdk",
          llmProvider: "anthropic",
          sourceKind: "uploaded_pdf",
          sourceLabel: "paper.pdf",
          sourceNote: "uploaded",
          status: "failed",
          error: "Scribe failed while writing final report",
          payload: null,
          log: "Scribe could not write the report"
        }}
      />
    );

    expect(screen.getAllByText(/needs attention/i).length).toBeGreaterThan(0);
    expect(screen.queryByText(/failed/i)).not.toBeInTheDocument();
    expect(document.querySelector(".status-dot")).not.toHaveStyle({
      background: "var(--err)"
    });
  });

  it("does not restore persisted runs without an explicit initial run", () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    render(<LabShell initialTopology={defaultTopologyFixture} />);

    expect(screen.getByRole("heading", { name: "Upload PDF" })).toBeInTheDocument();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("advances workflow nodes past read when payload.summary.stage updates", () => {
    // Regression guard: pre-fix the UI was permanently stuck at "src done /
    // read running" because payload was null. With the SSE-side enrichment
    // bridging pipeline_state.json into payload, this should no longer happen.
    const baseRun = {
      projectId: "ui_demo_stage_test",
      outputDir: "runs/ui_demo_stage_test",
      runMode: "sdk" as const,
      llmProvider: "anthropic" as const,
      executionMode: "efficient" as const,
      sandboxMode: "runpod" as const,
      gpuMode: "auto" as const,
      status: "running" as const,
      sourceKind: "workspace_fixture" as const,
      sourceLabel: "Test paper",
      sourceNote: "regression test",
      log: "",
      telemetry: []
    };

    // First mount: payload null → UI stuck on read (1 of 12 nodes done — only src)
    const stuck = render(
      <LabShell
        initialTopology={defaultTopologyFixture}
        initialRun={{ ...baseRun, payload: null }}
      />
    );
    expect(screen.getByText(/1\/12 agents complete/i)).toBeInTheDocument();
    stuck.unmount();

    // Fresh mount with enriched payload — must mount fresh because LabShell
    // takes initialRun only on first render. (Live updates flow through SSE; this
    // test just exercises the stateMapForRun branch for the new stage.)
    render(
      <LabShell
        initialTopology={defaultTopologyFixture}
        initialRun={{
          ...baseRun,
          payload: {
            projectId: baseRun.projectId,
            outputDir: baseRun.outputDir,
            runMode: "sdk",
            sourceKind: "workspace_fixture",
            sourceLabel: "Test paper",
            sourceNote: "regression test",
            generatedAt: new Date().toISOString(),
            log: "",
            initialSnapshot: {
              agents: [], reasoning: [], messages: [], citations: [],
              approvals: [], progress: [], dataPanels: [],
              hermesPanel: null, conceptCard: null
            },
            events: [],
            pathStates: { opt: "upcoming", bb: "upcoming", aug: "upcoming", hor: "upcoming", div: "upcoming" },
            decisionLog: [],
            assumptionCount: 0,
            gates: {},
            hermes: { stepReports: {}, checkpointReports: {}, interventions: [] },
            summary: {
              stage: "paper_understood",
              meanReward: null,
              improvementCount: 0,
              runModeLabel: "SDK: Anthropic",
              llmProvider: "anthropic",
              executionMode: "efficient",
              sandboxMode: "runpod",
              gpuMode: "auto",
              sourceLabel: "Test paper"
            }
          }
        }}
      />
    );

    // doneCount should advance: src + read = 2 nodes.
    expect(screen.getByText(/2\/12 agents complete/i)).toBeInTheDocument();
  });

  it("keeps the workflow done-count monotonic across every pipeline stage", () => {
    // Regression guard for the non-monotonic counter bug: at `plan_created`
    // stateMapForRun used to mark `env` back to `running` after it had already
    // shown `done` at `environment_built`, so the counter went 3 → 2 → 4
    // across environment_built → plan_created → gate_1_passed. The contract is
    // that the done-count never decreases as the pipeline stage advances.
    const STAGES = [
      "ingested",
      "paper_understood",
      "artifacts_discovered",
      "environment_built",
      "plan_created",
      "gate_1_passed",
      "baseline_implemented",
      "baseline_run",
      "gate_2_passed",
      "improvements_selected",
      "improvements_run",
      "gate_3_passed",
      "research_map_generated",
      "complete"
    ] as const;

    const runAtStage = (stage: string): LiveDemoRunState => ({
      projectId: "ui_monotonic_test",
      outputDir: "runs/ui_monotonic_test",
      runMode: "sdk",
      llmProvider: "anthropic",
      status: "running",
      sourceKind: "workspace_fixture",
      sourceLabel: "Test paper",
      sourceNote: "monotonic guard",
      log: "",
      telemetry: [],
      payload: {
        projectId: "ui_monotonic_test",
        outputDir: "runs/ui_monotonic_test",
        runMode: "sdk",
        sourceKind: "workspace_fixture",
        sourceLabel: "Test paper",
        sourceNote: "monotonic guard",
        generatedAt: new Date().toISOString(),
        log: "",
        initialSnapshot: {
          agents: [], reasoning: [], messages: [], citations: [],
          approvals: [], progress: [], dataPanels: [],
          hermesPanel: null, conceptCard: null
        },
        events: [],
        // Worst case for monotonicity: path nodes never report progress, so
        // they only flip to `done` at gate_3_passed.
        pathStates: { opt: "upcoming", bb: "upcoming", aug: "upcoming", hor: "upcoming", div: "upcoming" },
        decisionLog: [],
        assumptionCount: 0,
        gates: {},
        hermes: { stepReports: {}, checkpointReports: {}, interventions: [] },
        summary: {
          stage,
          meanReward: null,
          improvementCount: 0,
          runModeLabel: "SDK: Anthropic",
          llmProvider: "anthropic",
          executionMode: "efficient",
          sandboxMode: "runpod",
          gpuMode: "auto",
          sourceLabel: "Test paper"
        }
      }
    });

    const doneCount = (stage: string) =>
      Object.values(
        stateMapForRun(
          runAtStage(stage),
          defaultTopologyFixture.nodes.map((n) => ({ ...n, x: 0, y: 0 })),
          defaultTopologyFixture.improvement_path_ids
        )
      ).filter((s) => s === "done").length;

    const counts = STAGES.map(doneCount);
    for (let i = 1; i < counts.length; i += 1) {
      expect(
        counts[i],
        `done-count regressed at ${STAGES[i]}: ${counts[i - 1]} → ${counts[i]}`
      ).toBeGreaterThanOrEqual(counts[i - 1]);
    }

    // The specific root cause: `env` finished at environment_built and must
    // stay `done` through plan_created.
    expect(
      stateMapForRun(
        runAtStage("plan_created"),
        defaultTopologyFixture.nodes.map((n) => ({ ...n, x: 0, y: 0 })),
        defaultTopologyFixture.improvement_path_ids
      ).env
    ).toBe("done");
    // Sanity: a completed run shows all 12 nodes done.
    expect(doneCount("complete")).toBe(12);
  });
});
