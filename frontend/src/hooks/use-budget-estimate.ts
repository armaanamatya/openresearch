"use client";

import { useCallback, useRef, useState } from "react";

import type { PaperBudgetEstimate } from "@/components/lab/budget/budget-panel";

export interface UseBudgetEstimateResult {
  estimate: PaperBudgetEstimate | null;
  loading: boolean;
  error: string | null;
  /** Estimate from an uploaded PDF — multipart path. */
  estimateFromFile: (file: File) => Promise<void>;
  /** Estimate from an arXiv URL or bare ID — JSON path. */
  estimateFromArxiv: (urlOrId: string) => Promise<void>;
  /** Clear current estimate (e.g. user picked a different paper). */
  reset: () => void;
}

/**
 * Fetches a paper budget estimate from `/api/demo/estimate`.
 *
 * Reset on every new paper selection; the panel re-renders with a fresh
 * spinner. Errors are captured as strings so the panel can display
 * "Estimate unavailable" + the skip-and-start fallback.
 *
 * Race-safety: a stale response from an earlier paper cannot overwrite a
 * fresh one — we tag each request with an incrementing seq and ignore
 * out-of-order resolves.
 */
export function useBudgetEstimate(): UseBudgetEstimateResult {
  const [estimate, setEstimate] = useState<PaperBudgetEstimate | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const seq = useRef(0);

  const apply = useCallback((reqSeq: number, body: unknown, errMsg?: string) => {
    if (reqSeq !== seq.current) return;
    if (errMsg) {
      setError(errMsg);
      setEstimate(null);
    } else {
      setEstimate(body as PaperBudgetEstimate);
      setError(null);
    }
    setLoading(false);
  }, []);

  const estimateFromFile = useCallback(
    async (file: File) => {
      const reqSeq = ++seq.current;
      setLoading(true);
      setError(null);
      setEstimate(null);
      const fd = new FormData();
      fd.append("paper", file);
      try {
        const resp = await fetch("/api/demo/estimate", { method: "POST", body: fd });
        const text = await resp.text();
        if (!resp.ok) {
          apply(reqSeq, null, text || `Estimate failed (HTTP ${resp.status})`);
          return;
        }
        apply(reqSeq, JSON.parse(text));
      } catch (e) {
        apply(reqSeq, null, e instanceof Error ? e.message : "Estimate request failed");
      }
    },
    [apply]
  );

  const estimateFromArxiv = useCallback(
    async (urlOrId: string) => {
      const reqSeq = ++seq.current;
      setLoading(true);
      setError(null);
      setEstimate(null);
      const trimmed = urlOrId.trim();
      const sourceKind = /^\d{4}\.\d{4,5}(v\d+)?$/.test(trimmed) ? "arxiv_id" : "arxiv_url";
      const normalised =
        sourceKind === "arxiv_url" && !/^https?:\/\//i.test(trimmed)
          ? `https://${trimmed}`
          : trimmed;
      try {
        const resp = await fetch("/api/demo/estimate", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ source_kind: sourceKind, source: normalised })
        });
        const text = await resp.text();
        if (!resp.ok) {
          apply(reqSeq, null, text || `Estimate failed (HTTP ${resp.status})`);
          return;
        }
        apply(reqSeq, JSON.parse(text));
      } catch (e) {
        apply(reqSeq, null, e instanceof Error ? e.message : "Estimate request failed");
      }
    },
    [apply]
  );

  const reset = useCallback(() => {
    seq.current++;
    setEstimate(null);
    setLoading(false);
    setError(null);
  }, []);

  return { estimate, loading, error, estimateFromFile, estimateFromArxiv, reset };
}
