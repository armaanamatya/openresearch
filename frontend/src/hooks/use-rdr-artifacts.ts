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
}

const EMPTY_CLUSTERS: DemoClusterStatus[] = [];
const EMPTY_LEAVES: DemoLeafScore[] = [];
const EMPTY_PASSES: DemoRepairPass[] = [];

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

  useEffect(() => {
    if (!projectId) return;

    let cancelled = false;

    const fetchOnce = async () => {
      try {
        const [c, l, r] = await Promise.all([
          fetch(`/api/demo/runs/${encodeURIComponent(projectId)}/clusters`, {
            cache: "no-store",
          }).then((x) => (x.ok ? (x.json() as Promise<unknown>) : null)),
          fetch(`/api/demo/runs/${encodeURIComponent(projectId)}/leaf-scores`, {
            cache: "no-store",
          }).then((x) => (x.ok ? (x.json() as Promise<unknown>) : null)),
          fetch(
            `/api/demo/runs/${encodeURIComponent(projectId)}/repair-iterations`,
            { cache: "no-store" },
          ).then((x) => (x.ok ? (x.json() as Promise<unknown>) : null)),
        ]);
        if (cancelled) return;

        const cData = c as { clusters?: DemoClusterStatus[] } | null;
        const lData = l as { leaf_scores?: DemoLeafScore[] } | null;
        const rData = r as { passes?: DemoRepairPass[] } | null;

        if (Array.isArray(cData?.clusters)) setClusters(cData.clusters);
        if (Array.isArray(lData?.leaf_scores)) setLeafScores(lData.leaf_scores);
        if (Array.isArray(rData?.passes)) setRepairPasses(rData.passes);
        setActiveProjectId(projectId);
      } catch {
        // Poll quietly; network errors are non-fatal.
      }
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
    return { clusters: EMPTY_CLUSTERS, leafScores: EMPTY_LEAVES, repairPasses: EMPTY_PASSES };
  }

  return { clusters, leafScores, repairPasses };
}
