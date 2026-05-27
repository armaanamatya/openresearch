"use client";

import { useCallback, useState } from "react";

/**
 * useResume — POST /api/demo/resume?projectId=<id> to resume an interrupted run.
 *
 * Surfaces the PR-π Module D resume offer in the UI (spec
 * `docs/superpowers/specs/2026-05-26-pr-pi-sdk-resilience-design.md`). Reused
 * by the lab header when `runStateKind === "interrupted"` so a researcher can
 * continue a run that the orphan sweeper marked terminal, without re-running
 * earlier iterations from scratch.
 *
 * On success, the backend re-spawns the orchestrator subprocess against the
 * same project id; the lab UI's existing SSE stream will pick up the new
 * dashboard events automatically — no client-side navigation is needed.
 */
export interface UseResumeResult {
  resume: () => Promise<void>;
  busy: boolean;
  error: string | null;
}

export function useResume(
  projectId: string | null | undefined,
): UseResumeResult {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const resume = useCallback(async () => {
    if (!projectId) {
      setError("No project selected.");
      return;
    }
    setBusy(true);
    setError(null);
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 10_000);
    try {
      const response = await fetch(
        `/api/demo/resume?projectId=${encodeURIComponent(projectId)}`,
        { method: "POST", signal: controller.signal },
      );
      if (!response.ok) {
        const raw = await response.text().catch(() => "");
        let message = raw || "Unable to resume";
        try {
          const payload = JSON.parse(raw) as { error?: string; detail?: string };
          message = payload.error ?? payload.detail ?? message;
        } catch {
          /* keep raw text */
        }
        setError(message);
        return;
      }
      // Successful POST — the existing SSE stream takes over. Reset busy so the
      // button re-enables once the next run_state event lands (kind:initializing).
      setBusy(false);
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") {
        setError("Resume request timed out — check the backend.");
      } else {
        setError(err instanceof Error ? err.message : "Unable to resume");
      }
      setBusy(false);
    } finally {
      clearTimeout(timer);
    }
  }, [projectId]);

  return { resume, busy, error };
}
