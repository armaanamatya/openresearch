"use client";

import { useEffect, useState } from "react";
import type { LiveDemoRunState } from "@/lib/demo/demo-run-types";
import styles from "./telemetry-strip.module.css";

function formatCount(n: number): string {
  if (!Number.isFinite(n) || n < 0) return "—";
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1).replace(/\.0$/, "")}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1).replace(/\.0$/, "")}K`;
  return String(n);
}

function formatDuration(ms: number): string {
  if (!Number.isFinite(ms) || ms < 0) return "—";
  const s = Math.floor(ms / 1000);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${sec}s`;
  return `${sec}s`;
}

/**
 * Sticky bottom strip showing live run telemetry. Reads from
 * `run.telemetry` (sum across records) plus `startedAt`/`completedAt`
 * for wall-clock. Renders nothing when `run` is null — the strip is
 * for active or completed runs only, not the upload view.
 *
 * While the run is in flight (no `completedAt` yet), the wall-clock
 * cell needs to tick — a 1Hz interval keeps it honest. Once the run
 * completes, the interval is unnecessary and the effect tears down.
 */
export function TelemetryStrip({ run }: { run: LiveDemoRunState | null }) {
  const [now, setNow] = useState(() => Date.now());

  const isLive = Boolean(run && !run.completedAt && run.status !== "completed" && run.status !== "failed" && run.status !== "stopped");

  useEffect(() => {
    if (!isLive) return undefined;
    const id = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, [isLive]);

  if (!run) return null;

  const tel = run.telemetry ?? [];
  let totalMessages = 0;
  let totalChars = 0;
  for (const t of tel) {
    if (typeof t.message_count === "number") totalMessages += t.message_count;
    if (typeof t.output_chars === "number") totalChars += t.output_chars;
  }

  const startedAt = run.startedAt ? new Date(run.startedAt).getTime() : 0;
  const referenceEnd = run.completedAt ? new Date(run.completedAt).getTime() : now;
  const wall = startedAt && Number.isFinite(startedAt) ? referenceEnd - startedAt : 0;

  return (
    <div className={styles.strip} role="status" aria-label="Run telemetry">
      <div className={styles.cell}>
        <span className={styles.label}>Messages</span>
        <span className={styles.value}>{formatCount(totalMessages)}</span>
      </div>
      <div className={styles.cell}>
        <span className={styles.label}>Output</span>
        <span className={styles.value}>{formatCount(totalChars)}</span>
      </div>
      <div className={styles.cell}>
        <span className={styles.label}>Wall</span>
        <span className={styles.value}>{formatDuration(wall)}</span>
      </div>
      <div className={styles.cell}>
        <span className={styles.label}>Project</span>
        <span className={styles.value}>{run.projectId.slice(0, 14)}</span>
      </div>
    </div>
  );
}
