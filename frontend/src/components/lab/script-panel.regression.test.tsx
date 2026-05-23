import { describe, expect, it } from "vitest";
import { render } from "@testing-library/react";

import type { LiveDemoRunState } from "@/lib/demo/demo-run-types";

import { ScriptPanel } from "./script-panel";

// Pins the historical fixture-leak documented at script-panel.tsx:22-30 — a
// real run that FAILED at Gate 2 once rendered "Reproduced With Caveats /
// 91.4% / 492.3" because the workspace_fixture's demo_status template shipped
// with preset success values and nothing overwrote them. The current fix gates
// every success-shaped display on `payload.summary.stage === "complete"`. This
// test asserts the gate stays in place by constructing the exact inverse-
// shaped fixture (complete-looking benchmark numbers + a non-complete stage)
// and confirming none of the leak markers reach the DOM.
//
// If this test ever fails, the gate has regressed and a real user can again
// see fabricated success values for a halted run. The fix is in
// script-panel.tsx's `pipelineComplete = run.payload?.summary.stage ===
// "complete"` guard and the surrounding ternaries.

describe("ScriptPanel — historical fixture-leak regression", () => {
  it("does not render '91.4' / '492.3' / 'Reproduced With Caveats' when stage !== 'complete'", () => {
    const run = {
      projectId: "test-fixture-leak",
      outputDir: "runs/test-fixture-leak",
      runMode: "sdk",
      status: "failed",
      sourcePdf: null,
      startedAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      log: "",
      telemetry: [],
      benchmark: {
        benchmarkName: "PaperBench-style",
        paperbenchTaskId: "task-1",
        overallScore: 91.4,
        targetMetric: "accuracy",
        targetValue: 500.0,
        reproducedValue: 492.3,
        deltaValue: -7.7,
        verdict: "verified_with_caveats",
        reportPath: "",
        comparisonPath: "",
        logPath: "",
      },
      payload: {
        summary: { stage: "gate_2_failed" },
      } as never,
    } as unknown as LiveDemoRunState;

    const { container } = render(<ScriptPanel run={run} />);
    const text = container.textContent ?? "";

    // The verdict slot must show the gated state, not the success label.
    expect(text).toContain("Pending");

    // The hard guarantee — none of the historical leak markers may appear.
    expect(text).not.toContain("91.4");
    expect(text).not.toContain("492.3");
    expect(text).not.toContain("Reproduced With Caveats");
  });
});
