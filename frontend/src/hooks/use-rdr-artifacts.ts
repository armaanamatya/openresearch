"use client";

import { useEffect, useState } from "react";

import type {
  DemoClusterStatus,
  DemoLeafScore,
  DemoRepairPass,
} from "@/lib/demo/demo-run-types";

export interface RdrArtifacts {
  clusters: DemoClusterStatus[];
  leafScores: DemoLeafScore[];
  repairPasses: DemoRepairPass[];
  /**
   * True after 3 consecutive poll cycles where all 3 endpoints returned a
   * non-2xx status (404, 5xx, or network-error status=0). Signals "this run
   * will never have rdr artifacts" — callers can use this to suppress any UI
   * placeholder rather than showing an indefinite empty state.
   */
  noRdrArtifacts: boolean;
}

const EMPTY_CLUSTERS: DemoClusterStatus[] = [];
const EMPTY_LEAVES: DemoLeafScore[] = [];
const EMPTY_PASSES: DemoRepairPass[] = [];

/** Number of consecutive all-missing poll cycles before polling stops entirely. */
const STOP_AFTER_NO_ARTIFACT_CYCLES = 3;

/**
 * Polls the three RDR artifact endpoints while the run is active.
 *
 * - Fetches immediately on mount (or when projectId changes).
 * - Continues polling at `pollMs` intervals only when `isActive` is true.
 * - Returns empty arrays (never throws) — 404s during run-start are expected
 *   because the iterations directory and final_report.json don't exist yet.
 *
 * JSON shapes match backend _read_rdr_clusters / _read_rdr_repair_iterations /
 * _read_rdr_leaf_scores (backend/app.py).
 */
export function useRdrArtifacts(
  projectId: string | null,
  isActive: boolean,
  pollMs: number = 5_000,
): RdrArtifacts {
  // State is keyed by projectId — when projectId changes the new effect fires
  // a fresh fetch and replaces the previous run's data.
  const [activeProjectId, setActiveProjectId] = useState<string | null>(null);
  const [clusters, setClusters] = useState<DemoClusterStatus[]>([]);
  const [leafScores, setLeafScores] = useState<DemoLeafScore[]>([]);
  const [repairPasses, setRepairPasses] = useState<DemoRepairPass[]>([]);
  // Count consecutive poll cycles where every endpoint returned a non-2xx status.
  // After STOP_AFTER_NO_ARTIFACT_CYCLES we stop polling — the run has no rdr artifacts.
  const [noArtifactsCount, setNoArtifactsCount] = useState(0);

  useEffect(() => {
    if (!projectId) return;

    let cancelled = false;
    // Shadow the outer count inside the effect so the interval closure always
    // reads the latest value without re-registering on every state change.
    let localNoArtifactsCount = 0;

    type FetchResult = { status: number; data: unknown };

    const fetchOne = async (endpoint: string): Promise<FetchResult> => {
      try {
        const r = await fetch(
          `/api/demo/runs/${encodeURIComponent(projectId)}/${endpoint}`,
          { cache: "no-store" },
        );
        if (r.ok) return { status: r.status, data: await r.json() };
        return { status: r.status, data: null };
      } catch {
        return { status: 0, data: null }; // network error
      }
    };

    const fetchOnce = async () => {
      // If we have already confirmed this run has no rdr artifacts, bail early
      // so the interval doesn't fire additional requests.
      if (localNoArtifactsCount >= STOP_AFTER_NO_ARTIFACT_CYCLES) return;

      const [c, l, r] = await Promise.all([
        fetchOne("clusters"),
        fetchOne("leaf-scores"),
        fetchOne("repair-iterations"),
      ]);

      if (cancelled) return;

      // Treat any non-2xx response (404, 5xx, network-error status=0) as
      // "no artifact". The proxy now normalizes timeouts/5xx → 404, but defend
      // against future regressions.
      const allMissing = [c, l, r].every((res) => res.status === 0 || res.status >= 400);
      if (allMissing) {
        localNoArtifactsCount += 1;
        setNoArtifactsCount(localNoArtifactsCount);
        return;
      }

      // At least one endpoint returned 2xx data — reset counter.
      localNoArtifactsCount = 0;
      setNoArtifactsCount(0);

      const cData = c.data as { clusters?: DemoClusterStatus[] } | null;
      const lData = l.data as { leaf_scores?: DemoLeafScore[] } | null;
      const rData = r.data as { passes?: DemoRepairPass[] } | null;

      if (Array.isArray(cData?.clusters)) setClusters(cData.clusters);
      if (Array.isArray(lData?.leaf_scores)) setLeafScores(lData.leaf_scores);
      if (Array.isArray(rData?.passes)) setRepairPasses(rData.passes);
      setActiveProjectId(projectId);
    };

    void fetchOnce();

    if (!isActive) return;

    const id = setInterval(() => void fetchOnce(), pollMs);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [projectId, isActive, pollMs]);

  // If projectId has changed since the last successful fetch, suppress stale
  // data by returning empty arrays. This avoids setState-in-effect while still
  // presenting a clean state when the project changes.
  if (!projectId || activeProjectId !== projectId) {
    return {
      clusters: EMPTY_CLUSTERS,
      leafScores: EMPTY_LEAVES,
      repairPasses: EMPTY_PASSES,
      noRdrArtifacts: noArtifactsCount >= STOP_AFTER_NO_ARTIFACT_CYCLES,
    };
  }

  return {
    clusters,
    leafScores,
    repairPasses,
    noRdrArtifacts: noArtifactsCount >= STOP_AFTER_NO_ARTIFACT_CYCLES,
  };
}
