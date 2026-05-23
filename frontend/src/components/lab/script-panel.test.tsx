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
