// @vitest-environment node

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("server-only", () => ({}));

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
    // Production streams `request.body` (ReadableStream) straight through —
    // it does NOT call request.formData() (undici's parser throws on real
    // browser uploads; see the route comment). Assert on the passthrough
    // contract, not the body shape: route + multipart content-type
    // forwarded, method POST, duplex: "half" set for streaming.
    expect(fetch).toHaveBeenCalledWith(
      "http://backend.test/runs/upload",
      expect.objectContaining({
        method: "POST",
        headers: expect.objectContaining({
          "content-type": expect.stringContaining("multipart/form-data")
        }),
        duplex: "half"
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
        headers: expect.objectContaining({ "content-type": "application/json" }),
        body: JSON.stringify({
          mode: "sdk",
          provider: "openai",
          verificationProvider: "anthropic",
          executionMode: "max",
          sandbox: "docker",
          gpuMode: "prefer",
          // Production adds the model field (defaulted to "sonnet" by the
          // upload-view dropdown) to every run-start payload. Test must
          // assert it — earlier baseline was drift.
          model: "sonnet"
        })
      })
    );
  });

  // The earlier "rejects multipart requests without a pdf file before
  // proxying" test was deleted with Task C.4.1 (frontend-polish-pass).
  // The validation moved to the backend: the proxy now streams the body
  // through unconditionally (calling request.formData() in the proxy
  // throws on real-size browser uploads — see the route comment) and the
  // backend's /runs/upload returns 400 for empty/missing PDF. That
  // contract is verified end-to-end by the Playwright e2e spec, not by
  // a unit test of this proxy.

  it("surfaces sandbox preflight failures with a setup status", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(
      Response.json(
        {
          error: "Docker sandbox is not ready",
          code: "sandbox_preflight_failed"
        },
        { status: 503 }
      )
    );

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
