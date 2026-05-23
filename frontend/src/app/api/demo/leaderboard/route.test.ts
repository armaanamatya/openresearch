// @vitest-environment node

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("server-only", () => ({}));

describe("/api/demo/leaderboard proxy", () => {
  beforeEach(() => {
    vi.stubEnv("REPROLAB_BACKEND_URL", "http://backend.test");
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
  });

  it("forwards query params to the backend and returns its JSON", async () => {
    const captured: string[] = [];
    vi.stubGlobal("fetch", vi.fn(async (url: string) => {
      captured.push(url);
      return Response.json([{ project_id: "x", overall_score: 0.5 }], { status: 200 });
    }));
    const { GET } = await import("./route");

    const req = new Request("http://localhost/api/demo/leaderboard?paper=p1&order_by=cost");
    const res = await GET(req);
    expect(res.status).toBe(200);
    expect(captured[0]).toBe("http://backend.test/leaderboard?paper=p1&order_by=cost");
    const body = await res.json();
    expect(body[0].project_id).toBe("x");
  });

  it("propagates non-2xx status from the backend", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("boom", { status: 500 })));
    const { GET } = await import("./route");
    const req = new Request("http://localhost/api/demo/leaderboard");
    const res = await GET(req);
    expect(res.status).toBe(500);
  });

  it("returns 502 when fetch throws", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => { throw new Error("ECONNREFUSED"); }));
    const { GET } = await import("./route");
    const req = new Request("http://localhost/api/demo/leaderboard");
    const res = await GET(req);
    expect(res.status).toBe(502);
  });
});
