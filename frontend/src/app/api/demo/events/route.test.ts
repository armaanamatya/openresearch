// @vitest-environment node

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

describe("GET /api/demo/events", () => {
  beforeEach(() => {
    vi.stubEnv("REPROLAB_BACKEND_URL", "http://backend.test");
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
  });

  it("proxies run event streams from FastAPI", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response("event: heartbeat\ndata: {}\n\n", {
          status: 200,
          headers: { "content-type": "text/event-stream" }
        })
      )
    );

    const { GET } = await import("./route");
    const response = await GET(
      new Request("http://localhost:3000/api/demo/events?projectId=prj_123")
    );

    expect(response.status).toBe(200);
    expect(response.headers.get("content-type")).toContain("text/event-stream");
    expect(fetch).toHaveBeenCalledWith(
      "http://backend.test/runs/prj_123/events",
      expect.objectContaining({ cache: "no-store" })
    );
    await expect(response.text()).resolves.toContain("event: heartbeat");
  });

  it("requires a project id", async () => {
    const { GET } = await import("./route");
    const response = await GET(new Request("http://localhost:3000/api/demo/events"));

    expect(response.status).toBe(400);
  });
});

