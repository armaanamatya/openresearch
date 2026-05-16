"use client";
import { useEffect, useState } from "react";
import type { PipelineTopology } from "@/lib/pipeline/topology";

let cachedTopology: PipelineTopology | null = null;

/**
 * Fetches and caches the pipeline topology.
 *
 * Pass `initialTopology` from a server component to avoid a fetch on
 * first paint (SSR-friendly). Subsequent components reuse the cache.
 *
 * If the fetch fails, returns null — callers MUST handle that case
 * (typically by falling back to a "Loading…" or "Pipeline unavailable"
 * state).
 */
export function useTopology(initialTopology: PipelineTopology | null = null): PipelineTopology | null {
  const [topology, setTopology] = useState<PipelineTopology | null>(
    initialTopology ?? cachedTopology
  );

  useEffect(() => {
    if (topology) {
      cachedTopology = topology;
      return;
    }
    let cancelled = false;
    fetch("/api/pipeline/topology", { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : null))
      .then((data: PipelineTopology | null) => {
        if (cancelled) return;
        // Shape-guard against stubbed/incompatible responses (jsdom unit
        // tests that intercept global fetch return arbitrary JSON for
        // the run endpoints — without this we'd hand a malformed object
        // to layoutTopology, which crashes on .nodes iteration).
        if (
          data &&
          Array.isArray((data as Partial<PipelineTopology>).nodes) &&
          Array.isArray((data as Partial<PipelineTopology>).edges)
        ) {
          cachedTopology = data;
          setTopology(data);
        }
      })
      .catch(() => {
        // Stay null; caller renders a fallback.
      });
    return () => {
      cancelled = true;
    };
  }, [topology]);

  return topology;
}
