"use client";

import { useState, useEffect, useRef } from "react";
import type { DemoReportsResponse, DemoWorkerReport, DemoReportsSummary } from "../lib/demo/demo-run-types";
import type { RlmDashboardEvent } from "../lib/events/rlm-events";

/**
 * Hook that fetches worker reports from the API, refreshing when
 * worker_report_* SSE events arrive or on a periodic interval while active.
 */
export function useWorkerReports(
  projectId: string,
  events: RlmDashboardEvent[],
  isActive: boolean
) {
  const [workers, setWorkers] = useState<DemoWorkerReport[]>([]);
  const [summary, setSummary] = useState<DemoReportsSummary>({});
  const lastFetchRef = useRef(0);
  const workerEventCountRef = useRef(0);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => { mountedRef.current = false; };
  }, []);

  // Single effect that handles initial fetch, SSE-triggered refetch, and periodic polling
  useEffect(() => {
    if (!projectId) return;

    let timer: ReturnType<typeof setInterval> | null = null;

    const doFetch = async () => {
      // Debounce: don't fetch more than once per 2 seconds
      if (Date.now() - lastFetchRef.current < 2000) return;
      try {
        const res = await fetch(
          `/api/demo/runs/${encodeURIComponent(projectId)}/reports`,
          { cache: "no-store" }
        );
        if (!res.ok || !mountedRef.current) return;
        const data: DemoReportsResponse = await res.json();
        if (!mountedRef.current) return;
        setWorkers(data.workers ?? []);
        setSummary(data.summary ?? {});
        lastFetchRef.current = Date.now();
      } catch {
        // Silently fail — reports are supplementary
      }
    };

    // Initial fetch (deferred to avoid synchronous setState in effect)
    const initialTimer = setTimeout(doFetch, 0);

    // Count worker_report events to detect new ones
    const workerEventCount = events.filter(
      (e) =>
        e.event === "worker_report_started" ||
        e.event === "worker_report_completed" ||
        e.event === "worker_report_failed"
    ).length;

    if (workerEventCount > workerEventCountRef.current) {
      workerEventCountRef.current = workerEventCount;
      doFetch();
    }

    // Periodic refresh while active
    if (isActive) {
      timer = setInterval(doFetch, 10_000);
    }

    return () => {
      clearTimeout(initialTimer);
      if (timer) clearInterval(timer);
    };
  }, [projectId, events, isActive]);

  return { workers, summary };
}
