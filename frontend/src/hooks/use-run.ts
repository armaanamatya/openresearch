"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";

import type { DemoModelChoice, DemoRunMode, LiveDemoRunState } from "@/lib/demo/demo-run-types";
import type { DashboardLiveEvent } from "@/lib/events/dashboard-live-event";
import { isRlmEvent } from "@/lib/events/rlm-events";
import { issueText } from "@/components/lab/shared-helpers";
import { readUserPrefs } from "@/lib/user-prefs";

const MAX_DASHBOARD_EVENTS = 1000;
const POLL_INTERVAL_MS = 3000;
const LAST_RUN_KEY = "reprolab:lastRun";
const PINNED_DASHBOARD_EVENTS = new Set([
  "run_complete",
  "rubric_score",
  "candidate_proposed",
  "candidate_outcome",
  "user_message",
  "user_message_response",
  "cluster_started",
  "cluster_artifact_emitted",
  "cluster_scored",
  "repair_dispatched",
]);

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

function compactDashboardEvents(events: DashboardLiveEvent[]): DashboardLiveEvent[] {
  if (events.length <= MAX_DASHBOARD_EVENTS) return events;
  const tailStart = Math.max(0, events.length - MAX_DASHBOARD_EVENTS);
  const pinned = events
    .slice(0, tailStart)
    .filter((event) => PINNED_DASHBOARD_EVENTS.has(event.event));
  return [...pinned, ...events.slice(tailStart)];
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

// Merge a freshly-received run_state frame onto the current one. Carry
// forward the last telemetry and log when the new frame is empty, so
// transient backend responses do not regress the UI.
export function coalesceRunState(
  prev: LiveDemoRunState | null,
  next: LiveDemoRunState
): LiveDemoRunState {
  if (!prev || prev.projectId !== next.projectId) {
    return next;
  }
  return {
    ...next,
    telemetry: next.telemetry?.length ? next.telemetry : prev.telemetry,
    log: next.log || prev.log
  };
}

export interface UseRunResult {
  run: LiveDemoRunState | null;
  busy: boolean;
  error: string | null;
  dashboardEvents: DashboardLiveEvent[];
  runMode: DemoRunMode;
  setRunMode: (mode: DemoRunMode) => void;
  startFixtureRun: (model: DemoModelChoice) => Promise<void>;
  startUploadedRun: (file: File, model: DemoModelChoice) => Promise<void>;
  startArxivRun: (url: string, model: DemoModelChoice) => Promise<void>;
  resumeRun: (projectId: string, overrides?: Record<string, string>) => Promise<void>;
  clearRun: () => Promise<void>;
  resetToUpload: () => void;
}

export function useRun(initialRun: LiveDemoRunState | null = null): UseRunResult {
  const [run, setRun] = useState<LiveDemoRunState | null>(initialRun);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dashboardEvents, setDashboardEvents] = useState<DashboardLiveEvent[]>([]);
  const [runMode, setRunMode] = useState<DemoRunMode>("rlm");
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

    // ?new=1 forces the upload view — clear any persisted run and strip the
    // param from the URL so a refresh does not keep triggering it.
    if (new URLSearchParams(window.location.search).get("new") === "1") {
      clearLastRun();
      router.replace("/lab", { scroll: false });
      return;
    }

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
        // dashboardEvents are seeded in the useEffect([run?.projectId, run?.status])
        // block below — no action needed here for terminal runs.
      } catch {
        // Network error — next visit retries.
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    let seedTimer: number | null = null;

    if (run?.projectId !== dashboardProjectIdRef.current) {
      dashboardProjectIdRef.current = run?.projectId ?? null;
      // For terminal runs the events are seeded from payload.events in the
      // auto-resume effect (or will be seeded below from the current run).
      // Clearing here would wipe them — so only clear for live runs.
      const isTerminal = run?.status === "failed" || run?.status === "completed" || run?.status === "stopped";
      const rawEvents = Array.isArray(run?.payload?.events) ? run.payload.events : [];
      const rlmEvents = rawEvents.filter(isRlmEvent) as DashboardLiveEvent[];
      if (!isTerminal) {
        seedTimer = window.setTimeout(() => setDashboardEvents([]), 0);
      } else {
        // Navigation to a terminal run: seed from payload.events if present.
        seedTimer = window.setTimeout(() => setDashboardEvents(compactDashboardEvents(rlmEvents)), 0);
      }
    }

    eventSourceRef.current?.close();
    eventSourceRef.current = null;
    if (pollTimer.current) {
      window.clearTimeout(pollTimer.current);
      pollTimer.current = null;
    }

    if (!run || !["queued", "running"].includes(run.status)) {
      if (seedTimer !== null) {
        return () => window.clearTimeout(seedTimer);
      }
      return;
    }

    const pollOnce = async (projectId: string) => {
      try {
        const response = await fetch(`/api/demo?projectId=${encodeURIComponent(projectId)}`, {
          cache: "no-store"
        });
        if (!response.ok) {
          throw new Error("Unable to refresh run");
        }
        const next = (await response.json()) as LiveDemoRunState | null;
        if (next) {
          setRun((current) => coalesceRunState(current, next));
          if (next.status === "queued" || next.status === "running") {
            pollTimer.current = window.setTimeout(() => void pollOnce(projectId), POLL_INTERVAL_MS);
          }
        }
      } catch (pollError) {
        setError(pollError instanceof Error ? pollError.message : "Unable to refresh run");
        pollTimer.current = window.setTimeout(() => void pollOnce(projectId), POLL_INTERVAL_MS);
      }
    };

    const schedulePoll = (delayMs: number) => {
      if (pollTimer.current) {
        window.clearTimeout(pollTimer.current);
      }
      pollTimer.current = window.setTimeout(() => void pollOnce(run.projectId), delayMs);
    };

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
            return compactDashboardEvents(next);
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
        schedulePoll(500);
      };

      return () => {
        if (seedTimer !== null) window.clearTimeout(seedTimer);
        source.close();
        if (eventSourceRef.current === source) {
          eventSourceRef.current = null;
        }
        if (pollTimer.current) {
          window.clearTimeout(pollTimer.current);
          pollTimer.current = null;
        }
      };
    }

    schedulePoll(POLL_INTERVAL_MS);

    return () => {
      if (seedTimer !== null) window.clearTimeout(seedTimer);
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
      // mode, executionMode, sandbox, gpuMode, model so the
      // produced URL matches the contract asserted in lab-shell.test.tsx.
      const params = new URLSearchParams({
        mode: runMode,
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
  }, [setRunUrl, runMode]);

  const startUploadedRun = useCallback(async (file: File, model: DemoModelChoice) => {
    setBusy(true);
    setError(null);
    try {
      const prefs = readUserPrefs();
      const formData = new FormData();
      formData.set("mode", runMode);
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
  }, [setRunUrl, runMode]);

  const startArxivRun = useCallback(async (url: string, model: DemoModelChoice) => {
    setBusy(true);
    setError(null);
    try {
      const prefs = readUserPrefs();
      // The backend fetches the paper server-side; we just hand it the URL
      // and run-config knobs. arXiv-style paths get rewritten to /pdf on the
      // backend so users can paste either /abs/ or /pdf/ links.
      const normalisedUrl = /^https?:\/\//i.test(url.trim())
        ? url.trim()
        : `https://${url.trim()}`;
      const response = await postRunRequest("/api/demo/arxiv", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          url: normalisedUrl,
          mode: runMode,
          executionMode: prefs.executionMode ?? "efficient",
          sandbox: prefs.sandbox ?? "runpod",
          gpuMode: "auto",
          model
        })
      });
      if (!response.ok) {
        // The backend returns plain-text error bodies for httpx/PDF
        // validation failures; degrade gracefully when there's no JSON.
        const raw = await response.text().catch(() => "");
        let message = raw || "Unable to start arXiv run";
        try {
          const payload = JSON.parse(raw) as { error?: string; detail?: string };
          message = payload.error ?? payload.detail ?? message;
        } catch {
          /* keep raw text */
        }
        throw new Error(message);
      }
      const next = (await response.json()) as LiveDemoRunState;
      setRun(next);
      setRunUrl(next.projectId);
      writeLastRun(next.projectId);
    } catch (startError) {
      setError(describeStartError(startError, "Unable to start arXiv run"));
      setBusy(false);
    }
  }, [setRunUrl, runMode]);

  const resumeRun = useCallback(
    async (projectId: string, overrides: Record<string, string> = {}) => {
      // Resume an existing run from its on-disk checkpoint — the orchestrator
      // skips already-completed stages and only re-runs from the failure
      // point. Overrides (e.g. {executionMode: "max"}) let the operator push
      // past a wall-clock cap without losing the earlier agents' work.
      setBusy(true);
      setError(null);
      try {
        const response = await postRunRequest(
          `/api/demo/resume?projectId=${encodeURIComponent(projectId)}`,
          {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify(overrides)
          }
        );
        if (!response.ok) {
          const raw = await response.text().catch(() => "");
          let message = raw || "Unable to resume run";
          try {
            const payload = JSON.parse(raw) as { error?: string; detail?: string };
            message = payload.error ?? payload.detail ?? message;
          } catch {
            /* keep raw text */
          }
          throw new Error(message);
        }
        const next = (await response.json()) as LiveDemoRunState;
        setRun(next);
        setRunUrl(next.projectId);
        writeLastRun(next.projectId);
      } catch (resumeError) {
        setError(describeStartError(resumeError, "Unable to resume run"));
        setBusy(false);
      }
    },
    [setRunUrl]
  );

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

  return { run, busy, error, dashboardEvents, runMode, setRunMode, startFixtureRun, startUploadedRun, startArxivRun, resumeRun, clearRun, resetToUpload };
}
