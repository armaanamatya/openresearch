// @vitest-environment node

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

describe("/api/demo backend proxy", () => {
  beforeEach(() => {
    vi.stubEnv("REPROLAB_BACKEND_URL", "http://backend.test");
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        Response.json(
          {
            projectId: "prj_upload",
            outputDir: "runs/prj_upload",
            runMode: "sdk",
            status: "queued",
            payload: null,
            log: ""
          },
          { status: 202 }
        )
      )
    );
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
  });

  it("forwards uploaded-paper runs to the FastAPI upload route", async () => {
    const { POST } = await import("./route");
    const formData = new FormData();
    formData.set("mode", "sdk");
    formData.set("provider", "anthropic");
    formData.set("verificationProvider", "openai");
    formData.set(
      "paper",
      new File([new Uint8Array([0x25, 0x50, 0x44, 0x46, 0x2d])], "paper.pdf", {
        type: "application/pdf"
      })
    );

    const response = await POST(
      new Request("http://localhost:3000/api/demo", {
        method: "POST",
        body: formData
      })
    );

    expect(response.status).toBe(202);
    expect(fetch).toHaveBeenCalledWith(
      "http://backend.test/runs/upload",
      expect.objectContaining({
        method: "POST",
        body: expect.any(FormData)
      })
    );
  });

  it("starts fixture runs through the FastAPI JSON route", async () => {
    const { POST } = await import("./route");

    const response = await POST(
      new Request(
        "http://localhost:3000/api/demo?mode=sdk&provider=openai&verificationProvider=anthropic&executionMode=max&sandbox=docker&gpuMode=prefer",
        { method: "POST" }
      )
    );

    expect(response.status).toBe(202);
    expect(fetch).toHaveBeenCalledWith(
      "http://backend.test/runs",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          mode: "sdk",
          provider: "openai",
          verificationProvider: "anthropic",
          executionMode: "max",
          sandbox: "docker",
          gpuMode: "prefer"
        })
      })
    );
  });

  it("rejects multipart requests without a pdf file before proxying", async () => {
    const { POST } = await import("./route");
    const formData = new FormData();
    formData.set("mode", "offline");

    const response = await POST(
      new Request("http://localhost:3000/api/demo", {
        method: "POST",
        body: formData
      })
    );

    expect(response.status).toBe(400);
    await expect(response.json()).resolves.toEqual({
      error: "Upload a PDF before starting a lab run."
    });
    expect(fetch).not.toHaveBeenCalled();
  });

  it("surfaces sandbox preflight failures with a setup status", async () => {
    const error = Object.assign(new Error("Docker sandbox is not ready"), {
      code: "sandbox_preflight_failed",
      statusCode: 503
    });
    startDemoRun.mockRejectedValue(error);

    const { POST } = await import("./route");
    const response = await POST(
      new Request("http://localhost:3000/api/demo?mode=sdk&sandbox=docker", {
        method: "POST"
      })
    );

    expect(response.status).toBe(503);
    await expect(response.json()).resolves.toEqual({
      error: "Docker sandbox is not ready",
      code: "sandbox_preflight_failed"
    });
  });
});
