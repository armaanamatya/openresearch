import {
  STALL_THRESHOLD_SECONDS,
  computeProgress,
  formatDuration,
  parseLog
} from "../lib/demo/progress";
import type { LiveDemoRunState } from "../lib/demo/demo-run-types";

const SAMPLE_LOG = `
[ingest 6/6] Building workspace

==================================================
  > Starting: paper_understood
==================================================
  [paper-understanding] starting...
  [paper-understanding] (5s) tool: Bash \`ls /tmp\`
  [paper-understanding] (12s) tool: Read /tmp/parsed_full_text.txt
  ✓ paper_understood done in 66s

==================================================
  > Starting: environment_resolved
==================================================
  [environment-detective] (8s) tool: Read pyproject.toml
`;

function makeRun(overrides: Partial<LiveDemoRunState> = {}): LiveDemoRunState {
  return {
    projectId: "prj_test",
    outputDir: "/tmp/runs/prj_test",
    runMode: "sdk",
    status: "running",
    startedAt: "2026-05-09T00:00:00.000Z",
    updatedAt: "2026-05-09T00:00:30.000Z",
    payload: null,
    log: SAMPLE_LOG,
    ...overrides
  };
}

describe("parseLog", () => {
  it("extracts the most recent stage start, completed stages, and last activity line", () => {
    const parsed = parseLog(SAMPLE_LOG);
    expect(parsed.currentStageKey).toBe("environment_resolved");
    expect(parsed.completedStages.has("paper_understood")).toBe(true);
    expect(parsed.lastActivityText).toMatch(/Read pyproject.toml/);
  });

  it("captures failure messages from log lines like 'X FAILED:'", () => {
    const parsed = parseLog(
      "  X FAILED: paper_understood -- ValueError: Invalid IPv6 URL"
    );
    expect(parsed.failureText).toMatch(/paper_understood/);
    expect(parsed.failureText).toMatch(/Invalid IPv6 URL/);
  });
});

describe("computeProgress", () => {
  const NOW = Date.parse("2026-05-09T00:01:00.000Z"); // 60s after startedAt, 30s after updatedAt

  it("derives elapsed and last-activity seconds from timestamps", () => {
    const snap = computeProgress(makeRun(), NOW);
    expect(snap.elapsedSeconds).toBe(60);
    expect(snap.lastActivitySeconds).toBe(30);
  });

  it("counts completed stages and gives half-credit for the in-flight stage", () => {
    const snap = computeProgress(makeRun(), NOW);
    expect(snap.completedStageCount).toBe(1);
    // 1 done + 0.5 in-flight, out of 6 stages = 0.25
    expect(snap.percentComplete).toBeCloseTo(0.25, 5);
  });

  it("flags isStalled when running and last activity exceeds threshold", () => {
    const stalledRun = makeRun({
      updatedAt: new Date(NOW - (STALL_THRESHOLD_SECONDS + 5) * 1000).toISOString()
    });
    const snap = computeProgress(stalledRun, NOW);
    expect(snap.isStalled).toBe(true);
  });

  it("does not flag isStalled when status is completed", () => {
    const finishedRun = makeRun({
      status: "completed",
      updatedAt: new Date(NOW - 600 * 1000).toISOString()
    });
    const snap = computeProgress(finishedRun, NOW);
    expect(snap.isStalled).toBe(false);
  });

  it("falls back gracefully when the log is empty", () => {
    const snap = computeProgress(makeRun({ log: "" }), NOW);
    expect(snap.lastActivityText).toBeNull();
    expect(snap.completedStageCount).toBe(0);
  });
});

describe("formatDuration", () => {
  it.each([
    [null, "—"],
    [0, "0s"],
    [42, "42s"],
    [60, "1m 00s"],
    [3661, "1h 01m"]
  ])("formats %p as %p", (input, expected) => {
    expect(formatDuration(input)).toBe(expected);
  });
});
