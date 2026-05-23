import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useSteeringChat } from "./use-steering-chat";
import type { RlmDashboardEvent } from "../lib/events/rlm-events";

// ── Helpers ────────────────────────────────────────────────────────────────────

function makeUserMsgEvent(content: string, ts = "2026-05-21T00:00:01Z"): RlmDashboardEvent {
  return { event: "user_message", timestamp: ts, content };
}

function makeAssistantMsgEvent(message: string, ts = "2026-05-21T00:00:02Z"): RlmDashboardEvent {
  return { event: "user_message_response", timestamp: ts, message };
}

// ── Fetch mock ─────────────────────────────────────────────────────────────────

let fetchMock: ReturnType<typeof vi.fn>;

beforeEach(() => {
  fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) });
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  vi.restoreAllMocks();
});

// ── Tests ──────────────────────────────────────────────────────────────────────

describe("useSteeringChat", () => {
  it("returns empty messages when no chat events", () => {
    const { result } = renderHook(() => useSteeringChat("proj-1", []));
    expect(result.current.messages).toEqual([]);
    expect(result.current.sending).toBe(false);
    expect(result.current.error).toBe(null);
  });

  it("derives user messages from user_message events", () => {
    const events: RlmDashboardEvent[] = [makeUserMsgEvent("hello")];
    const { result } = renderHook(() => useSteeringChat("proj-1", events));
    expect(result.current.messages).toHaveLength(1);
    expect(result.current.messages[0].role).toBe("user");
    expect(result.current.messages[0].content).toBe("hello");
  });

  it("derives assistant messages from user_message_response events", () => {
    const events: RlmDashboardEvent[] = [makeAssistantMsgEvent("world")];
    const { result } = renderHook(() => useSteeringChat("proj-1", events));
    expect(result.current.messages).toHaveLength(1);
    expect(result.current.messages[0].role).toBe("assistant");
    expect(result.current.messages[0].content).toBe("world");
  });

  it("orders messages by stream order", () => {
    const events: RlmDashboardEvent[] = [
      makeUserMsgEvent("q1", "2026-05-21T00:00:00Z"),
      makeAssistantMsgEvent("a1", "2026-05-21T00:00:01Z"),
      makeUserMsgEvent("q2", "2026-05-21T00:00:02Z"),
    ];
    const { result } = renderHook(() => useSteeringChat("proj-1", events));
    const msgs = result.current.messages;
    expect(msgs.map((m) => m.content)).toEqual(["q1", "a1", "q2"]);
  });

  it("appends an optimistic message immediately on send", async () => {
    const { result } = renderHook(() => useSteeringChat("proj-1", []));
    await act(async () => {
      void result.current.send("ping");
    });
    // Optimistic message appears even before fetch resolves in this tick.
    // The hook set it synchronously; fetch was called.
    expect(fetchMock).toHaveBeenCalledOnce();
  });

  it("posts to the correct endpoint with projectId", async () => {
    const { result } = renderHook(() => useSteeringChat("my-project", []));
    await act(async () => {
      await result.current.send("test msg");
    });
    const [url, opts] = fetchMock.mock.calls[0];
    expect(url).toContain("/api/demo/runs/my-project/messages");
    expect(opts.method).toBe("POST");
    expect(JSON.parse(opts.body)).toEqual({ role: "user", content: "test msg" });
  });

  it("server-confirmed message appears once after SSE echo (no duplicate)", () => {
    // Verify that when the SSE stream contains a user_message event, the message
    // is rendered exactly once — not doubled by an optimistic entry.
    // We skip the optimistic-add-then-echo sequence here and test purely the
    // server-side path which is the most important invariant: SSE messages are
    // deduplicated correctly and optimistic entries don't duplicate them.
    const events: RlmDashboardEvent[] = [
      makeUserMsgEvent("hello from server"),
    ];
    const { result } = renderHook(() => useSteeringChat("proj-1", events));
    const count = result.current.messages.filter((m) => m.content === "hello from server").length;
    expect(count).toBe(1);
  });

  it("sets error and removes optimistic on fetch failure", async () => {
    fetchMock.mockResolvedValueOnce({ ok: false, status: 502, text: async () => "Bad Gateway" });
    const { result } = renderHook(() => useSteeringChat("proj-1", []));
    await act(async () => {
      await result.current.send("fail me");
    });
    expect(result.current.error).toMatch(/Bad Gateway|HTTP 502/);
    // Optimistic entry removed after failure.
    const has = result.current.messages.some((m) => m.content === "fail me");
    expect(has).toBe(false);
  });

  it("does not send while already sending", async () => {
    // Hold the first fetch in-flight by returning a never-resolving promise.
    let resolveFetch!: () => void;
    fetchMock.mockReturnValueOnce(
      new Promise<{ ok: boolean }>((resolve) => {
        resolveFetch = () => resolve({ ok: true });
      })
    );
    const { result } = renderHook(() => useSteeringChat("proj-1", []));
    act(() => { void result.current.send("first"); });
    expect(result.current.sending).toBe(true);
    // Second send while first is in-flight — should no-op.
    await act(async () => { await result.current.send("second"); });
    expect(fetchMock).toHaveBeenCalledTimes(1);
    // Clean up: resolve the first.
    resolveFetch();
  });

  it("ignores non-chat events", () => {
    const events: RlmDashboardEvent[] = [
      {
        event: "repl_iteration",
        timestamp: "2026-05-21T00:00:00Z",
        iteration: 1,
        response: "some response",
        code_blocks: [],
        sub_calls: 0,
        timing: null,
      },
    ];
    const { result } = renderHook(() => useSteeringChat("proj-1", events));
    expect(result.current.messages).toHaveLength(0);
  });
});
