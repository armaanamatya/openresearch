"use client";

import { useCallback, useState } from "react";
import { useRouter } from "next/navigation";

import type { LiveDemoRunState } from "@/lib/demo/demo-run-types";

export interface UseRerunResult {
  rerun: () => Promise<void>;
  busy: boolean;
  error: string | null;
}

/**
 * useRerun — POST /api/demo/runs/<projectId>/rerun and navigate to the new run.
 *
 * Reusable across any surface (header button, sidebar context menu, etc.)
 * that needs to restart a terminal run from its original paper source.
 */
export function useRerun(projectId: string | null | undefined): UseRerunResult {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();

  const rerun = useCallback(async () => {
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
        `/api/demo/runs/${encodeURIComponent(projectId)}/rerun`,
        { method: "POST", signal: controller.signal }
      );
      if (!response.ok) {
        const raw = await response.text().catch(() => "");
        let message = raw || "Unable to rerun";
        try {
          const payload = JSON.parse(raw) as { error?: string; detail?: string };
          message = payload.error ?? payload.detail ?? message;
        } catch {
          /* keep raw text */
        }
        setError(message);
        return;
      }
      const next = (await response.json()) as LiveDemoRunState;
      // navigate to the new run; replace keeps history clean
      router.replace(`/lab?projectId=${encodeURIComponent(next.projectId)}`, { scroll: false });
      // busy stays true until navigation completes (button shows spinner through redirect)
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") {
        setError("Rerun request timed out — check the backend.");
      } else {
        setError(err instanceof Error ? err.message : "Unable to rerun");
      }
      setBusy(false);
    } finally {
      clearTimeout(timer);
    }
  }, [projectId, router]);

  return { rerun, busy, error };
}
