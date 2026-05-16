"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";

import type { DemoModelChoice, LiveDemoRunState } from "@/lib/demo/demo-run-types";
import type { DashboardLiveEvent } from "@/components/lab/agent-timeline-rail";
import { issueText } from "@/components/lab/shared-helpers";
import { readUserPrefs } from "@/lib/user-prefs";

const MAX_DASHBOARD_EVENTS = 200;
const POLL_INTERVAL_MS = 3000;
const LAST_RUN_KEY = "reprolab:lastRun";

type EventSourceLike = {
  addEventListener: (type: string, listener: EventListenerOrEventListenerObject) => void;
  close: () => void;
  onerror: ((this: EventSource, ev: Event) => unknown) | null;
};

function writeLastRun(projectId: string): void {
  try {
    window.localStorage.setItem(LAST_RUN_KEY, projectId);
  } catch {
    // localStorage may be disabled (private mode etc.) — non-fatal.
  }
}

function clearLastRun(): void {
  try {
    window.localStorage.removeItem(LAST_RUN_KEY);
  } catch {
    // non-fatal
  }
}

function readLastRun(): string | null {
  try {
    return window.localStorage.getItem(LAST_RUN_KEY);
  } catch {
    return null;
  }
}

// fetch() rejects with a TypeError on a network-level failure — the
// connection dropped before the request completed (DNS, reset, a flaky
// localhost relay on WSL2 choking on a large upload body). That is
// distinct from an HTTP error *response*. A single quick retry recovers
// a genuine transient without masking a real server error.
async function postRunRequest(
  input: string,
  init: RequestInit,
  attempts = 2
): Promise<Response> {
  let lastError: unknown;
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    try {
      return await fetch(input, init);
    } catch (error) {
      lastError = error;
      if (attempt < attempts - 1) {
        await new Promise((resolve) => setTimeout(resolve, 600 * (attempt + 1)));
      }
    }
  }
  throw lastError;
}

function describeStartError(error: unknown, fallback: string): string {
  if (error instanceof TypeError) {
    return "Couldn't reach the server — the connection dropped before the request finished. Check your connection and try again.";
  }
  return error instanceof Error ? error.message : fallback;
}

// Merge a freshly-received run_state frame onto the current one. The GET
// /api/demo route caps payload enrichment and on timeout returns an
// un-enriched LiveDemoRunState (payload === null). The graph reads
// payload heavily, so applying the un-enriched frame verbatim would
// regress the graph for one tick. coalesceRunState merges, carrying the
// last enriched payload forward when the new frame lacks one.
export function coalesceRunState(
  prev: LiveDemoRunState | null,
  next: LiveDemoRunState
): LiveDemoRunState {
  if (!prev || prev.projectId !== next.projectId) {
    return next;
  }
  if (!next.payload && prev.payload && process.env.NODE_ENV !== "production") {
    console.warn(
      "[reprolab] un-enriched run_state frame (no payload) — retaining the last enriched payload so the graph does not regress"
    );
  }
  return {
    ...next,
    payload: next.payload ?? prev.payload,
    telemetry: next.telemetry?.length ? next.telemetry : prev.telemetry,
    log: next.log || prev.log
  };
}

export interface UseRunResult {
  run: LiveDemoRunState | null;
  busy: boolean;
  error: string | null;
  dashboardEvents: DashboardLiveEvent[];
  startFixtureRun: (model: DemoModelChoice) => Promise<void>;
  startUploadedRun: (file: File, model: DemoModelChoice) => Promise<void>;
  clearRun: () => Promise<void>;
  resetToUpload: () => void;
}

export function useRun(initialRun: LiveDemoRunState | null = null): UseRunResult {
  const [run, setRun] = useState<LiveDemoRunState | null>(initialRun);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dashboardEvents, setDashboardEvents] = useState<DashboardLiveEvent[]>([]);
  const eventSourceRef = useRef<EventSourceLike | null>(null);
  const pollTimer = useRef<number | null>(null);
  const dashboardProjectIdRef = useRef<string | null>(null);
  const router = useRouter();
  const didAutoResume = useRef(false);

  // Keep the URL in sync with the active run so a refresh or a shared
  // link restores it. `replace` (not `push`) avoids a history pile-up;
  // `scroll: false` keeps the viewport steady.
  const setRunUrl = useCallback(
    (projectId: string | null) => {
      router.replace(projectId ? `/lab?projectId=${encodeURIComponent(projectId)}` : "/lab", {
        scroll: false
      });
    },
    [router]
  );

  // Restore an in-flight run on mount.
  useEffect(() => {
    if (didAutoResume.current) {
      return;
    }
    didAutoResume.current = true;

    if (initialRun) {
      writeLastRun(initialRun.projectId);
      return;
    }

    const urlPid = new URLSearchParams(window.location.search).get("projectId");
    const candidate = urlPid ?? readLastRun();
    if (!candidate) {
      return;
    }

    void (async () => {
      try {
        const response = await fetch(`/api/demo?projectId=${encodeURIComponent(candidate)}`, {
          cache: "no-store"
        });
        if (response.status === 504) {
          return;
        }
        if (!response.ok) {
          clearLastRun();
          if (urlPid) {
            setRunUrl(null);
          }
          return;
        }
        const restored = (await response.json()) as LiveDemoRunState | null;
        if (!restored || !restored.projectId) {
          clearLastRun();
          if (urlPid) {
            setRunUrl(null);
          }
          return;
        }
        setRun(restored);
        setRunUrl(restored.projectId);
        writeLastRun(restored.projectId);
      } catch {
        // Network error — next visit retries.
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (run?.projectId !== dashboardProjectIdRef.current) {
      dashboardProjectIdRef.current = run?.projectId ?? null;
      setDashboardEvents([]);
    }

    eventSourceRef.current?.close();
    eventSourceRef.current = null;
    if (pollTimer.current) {
      window.clearTimeout(pollTimer.current);
      pollTimer.current = null;
    }

    if (!run || !["queued", "running"].includes(run.status)) {
      return;
    }

    if (typeof EventSource !== "undefined") {
      const source = new EventSource(
        `/api/demo/events?projectId=${encodeURIComponent(run.projectId)}`
      ) as unknown as EventSourceLike;
      eventSourceRef.current = source;
      source.addEventListener("run_state", (event) => {
        try {
          const next = JSON.parse((event as MessageEvent).data) as LiveDemoRunState;
          setRun((current) => coalesceRunState(current, next));
          if (next.status === "failed") {
            setError(next.error ? issueText(next.error) : "Run needs attention");
            setBusy(false);
          }
          if (next.status === "completed" || next.status === "stopped") {
            setBusy(false);
          }
        } catch {
          setError("Unable to parse live run update");
        }
      });
      source.addEventListener("agent_log", (event) => {
        try {
          const update = JSON.parse((event as MessageEvent).data) as {
            log?: string;
            text?: string;
          };
          setRun((current) =>
            current && current.projectId === run.projectId
              ? {
                  ...current,
                  log:
                    typeof update.log === "string"
                      ? update.log
                      : `${current.log}${update.text ?? ""}`
                }
              : current
          );
        } catch {
          setError("Unable to parse live log update");
        }
      });
      source.addEventListener("dashboard_event", (event) => {
        try {
          const evt = JSON.parse((event as MessageEvent).data) as DashboardLiveEvent;
          setDashboardEvents((prev) => {
            const next = [...prev, evt];
            return next.length > MAX_DASHBOARD_EVENTS
              ? next.slice(next.length - MAX_DASHBOARD_EVENTS)
              : next;
          });
        } catch {
          // Malformed dashboard events should never break the live UI.
        }
      });
      source.onerror = () => {
        source.close();
        if (eventSourceRef.current === source) {
          eventSourceRef.current = null;
        }
      };

      return () => {
        source.close();
        if (eventSourceRef.current === source) {
          eventSourceRef.current = null;
        }
      };
    }

    pollTimer.current = window.setTimeout(async () => {
      try {
        const response = await fetch(`/api/demo?projectId=${encodeURIComponent(run.projectId)}`, {
          cache: "no-store"
        });
        if (!response.ok) {
          throw new Error("Unable to refresh run");
        }
        const next = (await response.json()) as LiveDemoRunState | null;
        if (next) {
          setRun((current) => coalesceRunState(current, next));
        }
      } catch (pollError) {
        setError(pollError instanceof Error ? pollError.message : "Unable to refresh run");
      }
    }, POLL_INTERVAL_MS);

    return () => {
      if (pollTimer.current) {
        window.clearTimeout(pollTimer.current);
        pollTimer.current = null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [run?.projectId, run?.status]);

  const resetToUpload = useCallback(() => {
    setRun(null);
    setBusy(false);
    setError(null);
    eventSourceRef.current?.close();
    eventSourceRef.current = null;
    if (pollTimer.current) {
      window.clearTimeout(pollTimer.current);
      pollTimer.current = null;
    }
    clearLastRun();
    setRunUrl(null);
  }, [setRunUrl]);

  const startFixtureRun = useCallback(async (model: DemoModelChoice) => {
    setBusy(true);
    setError(null);
    try {
      const prefs = readUserPrefs();
      // URLSearchParams preserves insertion order; keep the key order
      // mode, provider, executionMode, sandbox, gpuMode, model so the
      // produced URL matches the contract asserted in lab-shell.test.tsx.
      const params = new URLSearchParams({
        mode: "sdk",
        provider: "anthropic",
        executionMode: prefs.executionMode ?? "efficient",
        sandbox: prefs.sandbox ?? "runpod",
        gpuMode: "auto",
        model
      });
      const response = await postRunRequest(
        `/api/demo?${params.toString()}`,
        { method: "POST" }
      );
      if (!response.ok) {
        const payload = (await response.json().catch(() => null)) as { error?: string } | null;
        throw new Error(payload?.error ?? "Unable to start run");
      }
      const next = (await response.json()) as LiveDemoRunState;
      setRun(next);
      setRunUrl(next.projectId);
      writeLastRun(next.projectId);
    } catch (startError) {
      setError(describeStartError(startError, "Unable to start run"));
      setBusy(false);
    }
  }, [setRunUrl]);

  const startUploadedRun = useCallback(async (file: File, model: DemoModelChoice) => {
    setBusy(true);
    setError(null);
    try {
      const prefs = readUserPrefs();
      const formData = new FormData();
      formData.set("mode", "sdk");
      formData.set("provider", "anthropic");
      formData.set("executionMode", prefs.executionMode ?? "efficient");
      formData.set("sandbox", prefs.sandbox ?? "runpod");
      formData.set("gpuMode", "auto");
      formData.set("model", model);
      formData.set("paper", file);
      const response = await postRunRequest("/api/demo", {
        method: "POST",
        body: formData
      });
      if (!response.ok) {
        const payload = (await response.json().catch(() => null)) as { error?: string } | null;
        throw new Error(payload?.error ?? "Unable to start uploaded run");
      }
      const next = (await response.json()) as LiveDemoRunState;
      setRun(next);
      setRunUrl(next.projectId);
      writeLastRun(next.projectId);
    } catch (startError) {
      setError(describeStartError(startError, "Unable to start uploaded run"));
      setBusy(false);
    }
  }, [setRunUrl]);

  const clearRun = useCallback(async () => {
    setBusy(true);
    try {
      if (run) {
        await fetch(`/api/demo?projectId=${encodeURIComponent(run.projectId)}`, {
          method: "DELETE"
        }).catch(() => null);
      }
    } finally {
      resetToUpload();
    }
  }, [run, resetToUpload]);

  return { run, busy, error, dashboardEvents, startFixtureRun, startUploadedRun, clearRun, resetToUpload };
}
