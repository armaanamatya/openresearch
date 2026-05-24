import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { ReportRail } from "./report-rail";

const areas = [
  { area: "Architecture matches paper", score: 0.9, weight: 0.2, status: "pass" as const },
  { area: "Attention-mask leak-free", score: 0.2, weight: 0.1, status: "fail" as const },
];
describe("ReportRail", () => {
  it("shows the verdict and stat grid for a completed run", () => {
    render(<ReportRail status="completed" elapsedMs={4320000}
      report={{ finalReportPath: "x", costUsd: 18.4,
        counts: { iterations: 13, primitiveCalls: 21, proposed: 7, promoted: 2 } }}
      rubric={{ current: 0.53, baseline: 0.22, target: 0.4, series: [], areas, previousAreas: [], attributableCandidate: null }} />);
    expect(screen.getByText(/0\.53/)).toBeInTheDocument();
    expect(screen.getByText("13")).toBeInTheDocument();
    expect(screen.getByText("7")).toBeInTheDocument();
  });
  it("renders one breakdown row per rubric area with its status", () => {
    render(<ReportRail status="running" elapsedMs={0} report={null}
      rubric={{ current: 0.3, baseline: 0.22, target: 0.4, series: [], areas, previousAreas: [], attributableCandidate: null }} />);
    expect(screen.getByText("Architecture matches paper")).toBeInTheDocument();
    expect(screen.getByText("Attention-mask leak-free")).toBeInTheDocument();
    // With report=null, count stat tiles must render "—" placeholders, not fabricated numbers.
    const placeholders = screen.getAllByText("—");
    expect(placeholders.length).toBeGreaterThanOrEqual(1);
  });
  it("shows a degraded-score note when the score is at the 0.35 cap", () => {
    render(<ReportRail status="completed" elapsedMs={1000}
      report={{ finalReportPath: "x", costUsd: null,
        counts: { iterations: 4, primitiveCalls: 3, proposed: 0, promoted: 0 } }}
      rubric={{ current: 0.35, baseline: 0.35, target: 0.4, series: [], areas: [], previousAreas: [], attributableCandidate: null }} />);
    expect(screen.getByText(/degraded|no real metrics/i)).toBeInTheDocument();
  });
  it("renders worker report fields", () => {
    render(<ReportRail status="completed" elapsedMs={1000}
      report={{ finalReportPath: "x", costUsd: null,
        counts: { iterations: 4, primitiveCalls: 3, proposed: 0, promoted: 0 } }}
      rubric={{ current: 0.62, baseline: 0.35, target: 0.4, series: [], areas: [], previousAreas: [], attributableCandidate: null }}
      workerReports={[{
        report_id: "wr-1",
        agent_id: "baseline-implementation",
        status: "completed",
        implemented: ["Added train.py"],
        left_undone: [],
        commands: [{ command: "python train.py", exit_code: 0 }],
        issues: ["Dataset mirror was slow"],
        procedures_followed: true
      }]} />);
    expect(screen.getByText("worker reports")).toBeInTheDocument();
    // agent_id appears in both the card title and the raw JSON disclosure
    expect(screen.getAllByText(/baseline-implementation/).length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText(/Added train\.py/).length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText(/exit 0/).length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText(/followed/).length).toBeGreaterThanOrEqual(1);
  });
  it("renders extended worker report with blockers, assignment, and raw JSON", () => {
    render(<ReportRail status="failed" elapsedMs={5000}
      report={null}
      rubric={{ current: null, baseline: null, target: 0.4, series: [], areas: [], previousAreas: [], attributableCandidate: null }}
      workerReports={[{
        report_id: "wr-ext-1",
        worker_id: "w-1",
        worker_type: "rdr_cluster",
        agent_id: "baseline-implementation",
        status: "failed",
        commands: [{ command: "python train.py", exit_code: 1 }],
        blockers: [{
          title: "SDK success-with-no-text",
          description: "Claude Code returned exit status success but no text",
          severity: "critical",
          source: "claude_agent_sdk"
        }],
        assignment: { summary: "Reproduce SDAR training loop" },
        duration_ms: 12000,
        error: "Exception: Claude Code returned an error result: success"
      }]} />);
    expect(screen.getByTestId("worker-report-card")).toBeInTheDocument();
    expect(screen.getByTestId("worker-status-badge")).toHaveTextContent("failed");
    // Blocker text appears in both the card and the raw JSON — use getAllByText
    expect(screen.getAllByText(/SDK success-with-no-text/).length).toBeGreaterThanOrEqual(1);
    expect(screen.getByTestId("worker-blocker")).toBeInTheDocument();
    expect(screen.getByText("Reproduce SDAR training loop")).toBeInTheDocument();
    expect(screen.getByTestId("command-exit-code")).toHaveTextContent("exit 1");
    expect(screen.getByTestId("raw-json-disclosure")).toBeInTheDocument();
  });
  it("renders empty state when no worker reports", () => {
    render(<ReportRail status="running" elapsedMs={0}
      report={null}
      rubric={{ current: null, baseline: null, target: 0.4, series: [], areas: [], previousAreas: [], attributableCandidate: null }}
      workerReports={[]} />);
    expect(screen.getByText(/Worker summaries appear/)).toBeInTheDocument();
  });
  it("renders summary card when reportsSummary is provided", () => {
    render(<ReportRail status="completed" elapsedMs={1000}
      report={null}
      rubric={{ current: null, baseline: null, target: 0.4, series: [], areas: [], previousAreas: [], attributableCandidate: null }}
      workerReports={[]}
      reportsSummary={{
        total_workers: 5,
        by_status: { completed: 3, failed: 2 },
        critical_blockers: [{ title: "SDK error", description: "", severity: "critical", source: "sdk" }],
        commands_run: 10,
        failed_commands: 3,
      }} />);
    expect(screen.getByText(/5 total/)).toBeInTheDocument();
    expect(screen.getByText(/SDK error/)).toBeInTheDocument();
  });
});
