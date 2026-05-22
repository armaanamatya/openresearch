// @vitest-environment node

import { describe, expect, it } from "vitest";
import { vi } from "vitest";

vi.mock("server-only", () => ({}));

import { __test__ } from "./node-runner";

describe("node-runner uploaded paper helpers", () => {
  it("derives the same deterministic project id shape used by pdf-path intake", () => {
    const projectId = __test__.projectIdForUploadedPdfPath(
      "C:\\runs\\.lab_uploads\\paper.pdf"
    );

    expect(projectId).toMatch(/^prj_[0-9a-f]{16}$/);
  });

  it("builds an uploaded-paper python script that routes through the pdf ingestion path", () => {
    const script = __test__.buildPythonScript(
      "prj_1234567890abcdef",
      "rlm",
      "anthropic",
      undefined,
      "max",
      "docker",
      "prefer",
      {
        sourcePath: "C:\\runs\\.lab_uploads\\paper.pdf",
        fileName: "paper.pdf"
      }
    );

    expect(script).toContain("from backend.cli import cmd_reproduce");
    expect(script).toContain("exit_code = cmd_reproduce");
    expect(script).toContain('write_status("failed", error=f"Pipeline exited with status {exit_code}"');
    expect(script).toContain('mode="rlm"');
    expect(script).toContain('execution_mode=execution_mode');
    expect(script).toContain('sandbox=sandbox_mode');
    expect(script).toContain('gpu_mode=gpu_mode');
    expect(script).toContain('"executionMode": execution_mode');
    expect(script).toContain('"sandboxMode": sandbox_mode');
    expect(script).toContain('"gpuMode": gpu_mode');
    expect(script).toContain('source_kind="pdf_path"');
    expect(script).toContain('uploaded_paper = Path("C:\\\\runs\\\\.lab_uploads\\\\paper.pdf").resolve()');
    expect(script).not.toContain("workspace = json.loads");
  });

  it("escapes uploaded paper paths and names before embedding them in python", () => {
    const script = __test__.buildPythonScript(
      "prj_1234567890abcdef",
      "rlm",
      "anthropic",
      undefined,
      "efficient",
      "auto",
      "auto",
      {
        sourcePath: "C:\\runs\\.lab_uploads\\evil'''paper.pdf",
        fileName: "evil'''paper.pdf"
      }
    );

    expect(script).toContain('uploaded_paper = Path("C:\\\\runs\\\\.lab_uploads\\\\evil\'\'\'paper.pdf").resolve()');
    expect(script).toContain('uploaded_file_name = "evil\'\'\'paper.pdf"');
    expect(script).not.toContain('r\'\'\'evil\'\'\'paper.pdf\'\'\'');
  });
});
