/**
 * T10 — thread the `autonomous` toggle + `repoUrl` through the three run
 * launchers. Mirrors the existing `minimizeCompute` wiring: camelCase
 * FormData key for the multipart upload path, snake_case JSON key for the
 * arXiv JSON path, camelCase query param for the dev/test fixture path.
 *
 * The backend consumers already exist (T2/T3): `_optional_form_bool(form,
 * "autonomous")` + `_optional_form_value(form, "repoUrl")` on
 * POST /runs/upload, and `StartArxivRunRequest.autonomous`/`.repo_url` on
 * POST /runs/arxiv — this test only asserts the frontend sends them.
 */
import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const routerReplaceMock = vi.fn();
vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: routerReplaceMock, push: vi.fn(), prefetch: vi.fn() }),
}));

import { useRun, type ProviderRunOptions } from "./use-run";

// A terminal ("completed") run state — the poll/SSE effect early-returns for
// any non queued/running status, so no EventSource/timer gets scheduled in
// jsdom (which has no EventSource). Keeps these tests focused purely on the
// outbound request the launcher sent.
function okRunStateResponse(projectId = "prj_test") {
  return {
    ok: true,
    status: 202,
    json: async () => ({
      projectId,
      outputDir: `runs/${projectId}`,
      runMode: "rlm",
      status: "completed",
      payload: null,
      log: "",
    }),
  };
}

function renderRun(options: ProviderRunOptions) {
  return renderHook(() => useRun(null, options));
}

describe("useRun launchers — autonomous + repoUrl threading", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    window.localStorage.clear();
  });

  describe("startUploadedRun", () => {
    it('sets FormData "autonomous" to "true" when options.autonomous is true', async () => {
      const fetchMock = vi.fn().mockResolvedValue(okRunStateResponse());
      vi.stubGlobal("fetch", fetchMock);
      const { result } = renderRun({ autonomous: true });

      await act(async () => {
        await result.current.startUploadedRun(new File(["dummy"], "paper.pdf"), "sonnet");
      });

      const call = fetchMock.mock.calls.find((c: unknown[]) => c[0] === "/api/demo");
      expect(call).toBeDefined();
      const formData = call![1].body as FormData;
      expect(formData.get("autonomous")).toBe("true");
    });

    it("omits FormData autonomous when options.autonomous is unset", async () => {
      const fetchMock = vi.fn().mockResolvedValue(okRunStateResponse());
      vi.stubGlobal("fetch", fetchMock);
      const { result } = renderRun({});

      await act(async () => {
        await result.current.startUploadedRun(new File(["dummy"], "paper.pdf"), "sonnet");
      });

      const call = fetchMock.mock.calls.find((c: unknown[]) => c[0] === "/api/demo");
      const formData = call![1].body as FormData;
      expect(formData.get("autonomous")).toBeNull();
    });

    it("sets FormData repoUrl (camelCase) when options.repoUrl is set", async () => {
      const fetchMock = vi.fn().mockResolvedValue(okRunStateResponse());
      vi.stubGlobal("fetch", fetchMock);
      const { result } = renderRun({ repoUrl: "https://github.com/foo/bar" });

      await act(async () => {
        await result.current.startUploadedRun(new File(["dummy"], "paper.pdf"), "sonnet");
      });

      const call = fetchMock.mock.calls.find((c: unknown[]) => c[0] === "/api/demo");
      const formData = call![1].body as FormData;
      expect(formData.get("repoUrl")).toBe("https://github.com/foo/bar");
    });

    it("resolves with the launched run's projectId on success", async () => {
      vi.stubGlobal("fetch", vi.fn().mockResolvedValue(okRunStateResponse("prj_upload_ok")));
      const { result } = renderRun({});

      let resolved: string | undefined;
      await act(async () => {
        resolved = await result.current.startUploadedRun(new File(["dummy"], "paper.pdf"), "sonnet");
      });

      expect(resolved).toBe("prj_upload_ok");
    });

    it("resolves with undefined when the request fails", async () => {
      vi.stubGlobal(
        "fetch",
        vi.fn().mockResolvedValue({ ok: false, status: 500, json: async () => ({ error: "boom" }) })
      );
      const { result } = renderRun({});

      let resolved: string | undefined;
      await act(async () => {
        resolved = await result.current.startUploadedRun(new File(["dummy"], "paper.pdf"), "sonnet");
      });

      expect(resolved).toBeUndefined();
    });
  });

  describe("startArxivRun", () => {
    it("includes autonomous:true and repo_url (snake_case) in the JSON body when both are set", async () => {
      const fetchMock = vi.fn().mockResolvedValue(okRunStateResponse());
      vi.stubGlobal("fetch", fetchMock);
      const { result } = renderRun({ autonomous: true, repoUrl: "https://github.com/foo/bar" });

      await act(async () => {
        await result.current.startArxivRun("arxiv.org/abs/1234.56789", "sonnet");
      });

      const call = fetchMock.mock.calls.find((c: unknown[]) => c[0] === "/api/demo/arxiv");
      expect(call).toBeDefined();
      const body = JSON.parse(call![1].body as string);
      expect(body.autonomous).toBe(true);
      expect(body.repo_url).toBe("https://github.com/foo/bar");
    });

    it("omits autonomous/repo_url keys from the JSON body when unset", async () => {
      const fetchMock = vi.fn().mockResolvedValue(okRunStateResponse());
      vi.stubGlobal("fetch", fetchMock);
      const { result } = renderRun({});

      await act(async () => {
        await result.current.startArxivRun("arxiv.org/abs/1234.56789", "sonnet");
      });

      const call = fetchMock.mock.calls.find((c: unknown[]) => c[0] === "/api/demo/arxiv");
      const body = JSON.parse(call![1].body as string);
      expect("autonomous" in body).toBe(false);
      expect("repo_url" in body).toBe(false);
    });

    it("resolves with the launched run's projectId on success", async () => {
      vi.stubGlobal("fetch", vi.fn().mockResolvedValue(okRunStateResponse("prj_arxiv_ok")));
      const { result } = renderRun({});

      let resolved: string | undefined;
      await act(async () => {
        resolved = await result.current.startArxivRun("arxiv.org/abs/1234.56789", "sonnet");
      });

      expect(resolved).toBe("prj_arxiv_ok");
    });

    it("resolves with undefined when the request fails", async () => {
      vi.stubGlobal(
        "fetch",
        vi.fn().mockResolvedValue({
          ok: false,
          status: 500,
          text: async () => JSON.stringify({ error: "boom" }),
        })
      );
      const { result } = renderRun({});

      let resolved: string | undefined;
      await act(async () => {
        resolved = await result.current.startArxivRun("arxiv.org/abs/1234.56789", "sonnet");
      });

      expect(resolved).toBeUndefined();
    });
  });

  describe("startFixtureRun", () => {
    it("sets autonomous/repoUrl as camelCase query params when set", async () => {
      const fetchMock = vi.fn().mockResolvedValue(okRunStateResponse());
      vi.stubGlobal("fetch", fetchMock);
      const { result } = renderRun({ autonomous: true, repoUrl: "https://github.com/foo/bar" });

      await act(async () => {
        await result.current.startFixtureRun("sonnet");
      });

      const call = fetchMock.mock.calls.find(
        (c: unknown[]) => typeof c[0] === "string" && (c[0] as string).startsWith("/api/demo?")
      );
      expect(call).toBeDefined();
      const url = new URL(call![0] as string, "http://localhost");
      expect(url.searchParams.get("autonomous")).toBe("true");
      expect(url.searchParams.get("repoUrl")).toBe("https://github.com/foo/bar");
    });
  });
});
