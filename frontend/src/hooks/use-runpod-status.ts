"use client";

import { useEffect, useState } from "react";
import type { DemoRunpodStatusResponse } from "@/lib/demo/demo-run-types";

export function useRunpodStatus(
  projectId: string,
  enabled: boolean,
  pollMs: number = 10_000,
): DemoRunpodStatusResponse | null {
  const [status, setStatus] = useState<DemoRunpodStatusResponse | null>(null);

  useEffect(() => {
    if (!enabled) {
      return;
    }
    let cancelled = false;
    let intervalId: ReturnType<typeof setInterval> | null = null;
    let controller: AbortController | null = null;

    const fetchStatus = async () => {
      controller?.abort();
      controller = new AbortController();
      try {
        const response = await fetch(
          `/api/demo/runs/${encodeURIComponent(projectId)}/runpod-status`,
          { cache: "no-store", signal: controller.signal },
        );
        if (!response.ok) return;
        const body = (await response.json()) as DemoRunpodStatusResponse;
        if (!cancelled) setStatus(body);
      } catch {
        // The event-derived chip remains visible. RunPod API polling is best-effort.
      }
    };

    void fetchStatus();
    intervalId = setInterval(() => {
      void fetchStatus();
    }, pollMs);

    return () => {
      cancelled = true;
      controller?.abort();
      if (intervalId !== null) clearInterval(intervalId);
    };
  }, [enabled, pollMs, projectId]);

  if (!enabled || status?.project_id !== projectId) return null;
  return status;
}
