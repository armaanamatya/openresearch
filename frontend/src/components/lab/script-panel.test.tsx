import { render, screen } from "@testing-library/react";
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
