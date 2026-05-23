import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { LiveDemoRunState } from "@/lib/demo/demo-run-types";

import { ScriptPanel } from "./script-panel";

// ------------------------------------------------------------------ helpers

const baseRun = (overrides: Partial<LiveDemoRunState> = {}): LiveDemoRunState => ({
  projectId: "prj_test",
  outputDir: "runs/prj_test",
  runMode: "sdk",
  status: "running",
  payload: null,
  log: "",
  ...overrides,
});

// ------------------------------------------------------------------
// Item 1 — pdf?.fileName ?? "paper.pdf"  (script-panel.tsx:71)
// ------------------------------------------------------------------
// NOTE: items are listed in order; tests for subsequent items are appended below.

describe("ScriptPanel — item 1: pdf.fileName fallback", () => {
  it("renders the real filename when sourcePdf is present", () => {
    render(
      <ScriptPanel
        run={baseRun({
          sourcePdf: {
            fileName: "my-real-paper.pdf",
            title: "Real Paper Title",
            sizeBytes: 1024,
            sha256: "abc123",
            runPath: "runs/prj_test/raw_paper.pdf",
            codePath: "runs/prj_test/code/"
          }
        })}
      />
    );
    const meta = document.querySelector(".pdf-meta");
    expect(meta?.textContent).toContain("my-real-paper.pdf");
  });

  it("renders '—' (not 'paper.pdf') when sourcePdf is absent", () => {
    render(<ScriptPanel run={baseRun({ sourcePdf: null })} />);
    const meta = document.querySelector(".pdf-meta");
    // Must NOT show the hardcoded literal
    expect(meta?.textContent).not.toContain("paper.pdf");
    // Must show honest-empty dash
    expect(meta?.textContent).toMatch(/^—/);
  });
});

// ------------------------------------------------------------------
// Item 2 — pdf?.codePath ?? `${run.outputDir}/code/paper.pdf`  (script-panel.tsx:88)
// ------------------------------------------------------------------

describe("ScriptPanel — item 2: code root path fallback", () => {
  it("renders the real codePath when sourcePdf is present", () => {
    render(
      <ScriptPanel
        run={baseRun({
          sourcePdf: {
            fileName: "paper.pdf",
            title: "Title",
            sizeBytes: 1024,
            sha256: "abc123",
            runPath: "runs/prj_test/raw_paper.pdf",
            codePath: "runs/prj_test/code/"
          }
        })}
      />
    );
    const path = document.querySelector(".code-root-path");
    expect(path?.textContent).toBe("runs/prj_test/code/");
  });

  it("renders '—' (not the synthesized path) when sourcePdf is absent", () => {
    render(<ScriptPanel run={baseRun({ outputDir: "runs/prj_test", sourcePdf: null })} />);
    const path = document.querySelector(".code-root-path");
    // Must NOT render any synthesized path containing outputDir
    expect(path?.textContent).not.toContain("runs/prj_test");
    // Must show honest-empty dash
    expect(path?.textContent).toBe("—");
  });
});

// ------------------------------------------------------------------
// Item 3 — benchmark?.benchmarkName ?? "PaperBench-style final benchmark"  (script-panel.tsx:106)
// ------------------------------------------------------------------

describe("ScriptPanel — item 3: benchmark name fallback", () => {
  it("renders the real benchmarkName when benchmark is present", () => {
    render(
      <ScriptPanel
        run={baseRun({
          benchmark: {
            benchmarkName: "My Custom Benchmark",
            paperbenchTaskId: "custom/task-1",
            overallScore: 80,
            targetMetric: "accuracy",
            targetValue: 90,
            reproducedValue: 80,
            deltaValue: -10,
            verdict: "partial_reproduction",
            reportPath: "runs/prj_test/code/final_benchmark_report.md",
            comparisonPath: "runs/prj_test/code/comparison.json",
            logPath: "runs/prj_test/code/logs/eval.log"
          },
          payload: { summary: { stage: "complete" } } as never
        })}
      />
    );
    const title = document.querySelector(".benchmark-title");
    expect(title?.textContent).toBe("My Custom Benchmark");
  });

  it("renders '—' (not 'PaperBench-style final benchmark') when benchmark is absent", () => {
    render(<ScriptPanel run={baseRun({ benchmark: null })} />);
    const title = document.querySelector(".benchmark-title");
    // Must NOT show the hardcoded descriptive literal
    expect(title?.textContent).not.toBe("PaperBench-style final benchmark");
    // Must show honest-empty dash
    expect(title?.textContent).toBe("—");
  });
});

// ------------------------------------------------------------------
// Item 4 — benchmark?.paperbenchTaskId ?? "pending evaluator output"  (script-panel.tsx:109)
// ------------------------------------------------------------------

describe("ScriptPanel — item 4: benchmark task-id fallback", () => {
  it("renders the real paperbenchTaskId when benchmark is present", () => {
    render(
      <ScriptPanel
        run={baseRun({
          benchmark: {
            benchmarkName: "My Benchmark",
            paperbenchTaskId: "reprolab/ppo-cartpole-v1",
            overallScore: 80,
            targetMetric: "accuracy",
            targetValue: 90,
            reproducedValue: 80,
            deltaValue: -10,
            verdict: "partial_reproduction",
            reportPath: "runs/prj_test/code/final_benchmark_report.md",
            comparisonPath: "runs/prj_test/code/comparison.json",
            logPath: "runs/prj_test/code/logs/eval.log"
          },
          payload: { summary: { stage: "complete" } } as never
        })}
      />
    );
    const subtitle = document.querySelector(".benchmark-subtitle");
    expect(subtitle?.textContent).toBe("reprolab/ppo-cartpole-v1");
  });

  it("renders '—' (not 'pending evaluator output') when benchmark is absent", () => {
    render(<ScriptPanel run={baseRun({ benchmark: null })} />);
    const subtitle = document.querySelector(".benchmark-subtitle");
    // Must NOT show the fabricated evaluator-state framing
    expect(subtitle?.textContent).not.toBe("pending evaluator output");
    // Must show honest-empty dash
    expect(subtitle?.textContent).toBe("—");
  });
});

// ------------------------------------------------------------------
// Item 5 — shortHash() returns "pending" → sha256:pending  (script-panel.tsx:74)
// ------------------------------------------------------------------

describe("ScriptPanel — item 5: sha256 hash row visibility", () => {
  it("renders the sha256 hash row when sha256 is present", () => {
    render(
      <ScriptPanel
        run={baseRun({
          sourcePdf: {
            fileName: "paper.pdf",
            title: "Title",
            sizeBytes: 1024,
            sha256: "abcdef1234567890",
            runPath: "runs/prj_test/raw_paper.pdf",
            codePath: "runs/prj_test/code/"
          }
        })}
      />
    );
    const hashDiv = document.querySelector(".pdf-hash");
    expect(hashDiv).not.toBeNull();
    // Shows the truncated hash, not "sha256:pending"
    expect(hashDiv?.textContent).toContain("abcdef123456");
    expect(hashDiv?.textContent).not.toContain("pending");
  });

  it("does not render the sha256 hash row at all when sha256 is absent", () => {
    render(<ScriptPanel run={baseRun({ sourcePdf: null })} />);
    const hashDiv = document.querySelector(".pdf-hash");
    // The row must be suppressed entirely — no "sha256:pending" rendered
    expect(hashDiv).toBeNull();
  });
});
