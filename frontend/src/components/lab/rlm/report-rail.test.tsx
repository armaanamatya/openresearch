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
});
