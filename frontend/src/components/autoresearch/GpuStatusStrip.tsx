"use client";

import { useEffect, useState } from "react";

import type { GpuPlan, RlmRunState } from "@/hooks/use-rlm-run";
import styles from "./GpuStatusStrip.module.css";

export interface GpuStatusStripProps {
  /** The most-recently resolved GPU plan for this run — populated from the
   * `gpu_resolved` SSE event via useRlmRun's own fold (RlmRunState.gpuPlan);
   * `null` before resolve_gpu_requirements completes (or if the paper never
   * needs a GPU at all). Read straight off the session page's existing
   * useRlmRun(events) call — no new subscription, no shared-reducer edit. */
  gpuPlan: GpuPlan | null;
  /** The run's overall lifecycle status, driving the pending/active/terminal
   * wording below. */
  status: RlmRunState["status"];
  className?: string;
}

// There is no dedicated "provisioning finished, training started" SSE event
// today — gpu_resolved only fires once a SKU is *chosen*, not once the
// sandbox is actually up. Rather than fabricate that signal, this is an
// honest, data-derived approximation: show "Provisioning…" for a short grace
// window anchored on the event's own `resolved_at` timestamp, then fall back
// to the run's plain lifecycle status. Never asserts more than the wire data
// supports.
const PROVISIONING_GRACE_MS = 90_000;

function friendlyStatus(status: RlmRunState["status"]): string {
  switch (status) {
    case "queued":
      return "Queued";
    case "running":
      return "Running";
    case "completed":
      return "Completed";
    case "partial":
      return "Completed (partial)";
    case "failed":
      return "Failed";
    default:
      return status;
  }
}

/**
 * Compact GPU/compute status strip for the live session view (alphaXiv
 * screen `_5`). Surfaces the resolved GpuPlan (SKU, count, VRAM, $/hr) once
 * available, with a provisioning-vs-running indicator; degrades to a neutral
 * "resolving" chip before resolution, and disappears entirely once the run
 * is terminal without ever having touched a GPU (a CPU-only paper) — never
 * shows a fabricated or stale claim.
 *
 * Note on `cloud_type`: this is the RunPod capacity tier the resolver chose
 * ("COMMUNITY"/"SECURE"/"ONDEMAND"), not necessarily the literal cloud
 * brand (the sandbox could be RunPod, GCP, Azure, or local — that choice
 * isn't part of the gpu_resolved payload), so it's shown as-is rather than
 * relabelled to a specific provider name.
 */
export function GpuStatusStrip({ gpuPlan, status, className }: GpuStatusStripProps) {
  // `Date.now()` is an impure call, so it's never read directly during
  // render (react-hooks/purity) — `nowMs` is plain state, updated only from
  // inside async callbacks (setTimeout/setInterval), never synchronously in
  // the effect body itself (react-hooks/set-state-in-effect). The initial
  // tick is deferred a macrotask via setTimeout(0) so it's still an async
  // callback, not a direct effect-body call.
  const [nowMs, setNowMs] = useState<number | null>(null);

  useEffect(() => {
    if (!gpuPlan || status !== "running") return;
    const tick = () => setNowMs(Date.now());
    const initial = window.setTimeout(tick, 0);
    const id = window.setInterval(tick, 5000);
    return () => {
      window.clearTimeout(initial);
      window.clearInterval(id);
    };
  }, [gpuPlan, status]);

  const classes = [styles.strip, className].filter(Boolean).join(" ");

  if (!gpuPlan) {
    // Only worth mentioning while the run is actually live — a terminal run
    // that never resolved a GPU (e.g. a CPU-only paper) should not show a
    // permanently "resolving" chip.
    if (status !== "queued" && status !== "running") return null;
    return (
      <div className={classes} role="status" aria-live="polite" data-testid="gpu-status-strip">
        <span className={styles.dot} data-tone="pending" aria-hidden="true" />
        <span className={styles.label}>Resolving compute requirements…</span>
      </div>
    );
  }

  const resolvedAtMs = new Date(gpuPlan.resolved_at).getTime();
  const isProvisioning =
    status === "running" &&
    nowMs !== null &&
    !Number.isNaN(resolvedAtMs) &&
    nowMs - resolvedAtMs < PROVISIONING_GRACE_MS;

  const countLabel = gpuPlan.gpu_count > 1 ? `${gpuPlan.gpu_count}× ` : "";
  const rateLabel =
    typeof gpuPlan.total_usd_per_hr === "number" ? `$${gpuPlan.total_usd_per_hr.toFixed(2)}/hr` : null;
  const tone = status === "failed" ? "err" : isProvisioning ? "pending" : status === "running" ? "active" : "done";

  return (
    <div className={classes} role="status" aria-live="polite" data-testid="gpu-status-strip">
      <span className={styles.dot} data-tone={tone} aria-hidden="true" />
      <span className={styles.label}>
        {countLabel}
        {gpuPlan.short_name}
        {typeof gpuPlan.vram_gb === "number" ? ` · ${gpuPlan.vram_gb} GB` : ""}
      </span>
      {gpuPlan.cloud_type && <span className={styles.cloudType}>{gpuPlan.cloud_type}</span>}
      {rateLabel && <span className={styles.rate}>{rateLabel}</span>}
      {gpuPlan.source === "fallback" && (
        <span
          className={styles.fallback}
          title={gpuPlan.requirements?.reasoning ?? "fallback to default SKU"}
        >
          fallback
        </span>
      )}
      <span className={styles.statusWord}>
        {isProvisioning ? "Provisioning…" : friendlyStatus(status)}
      </span>
    </div>
  );
}
