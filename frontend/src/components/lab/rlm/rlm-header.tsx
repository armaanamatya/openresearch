"use client";

import { useMemo, useState } from "react";
import type { RunWarning } from "../../../hooks/use-rlm-run";
import styles from "./rlm-header.module.css";

export type RlmRunStatus = "queued" | "running" | "completed" | "partial" | "failed";

/** Threshold in milliseconds after which a running-but-silent run is considered wedged. */
const HEARTBEAT_STALE_MS = 60_000;
const ERROR_TRUNCATE = 200;

interface RlmHeaderProps {
  paperTitle: string;
  paperMeta: string;
  projectId: string;
  status: RlmRunStatus;
  iterationCount: number;
  costUsd: number | null;
  warnings?: RunWarning[];
  /** ISO-8601 timestamp of the last heartbeat, or null if none received yet. */
  lastHeartbeatAt?: string | null;
  /** Client-side clock tick owned by the parent; avoids Date.now() during render. */
  heartbeatNowMs?: number | null;
  /** Error message to display when status is "failed". */
  error?: string | null;
  /** Callback to trigger a rerun; present for failed and completed runs. */
  onRerun?: () => Promise<void>;
  /** True while the rerun POST is in-flight (disables button, shows spinner). */
  rerunBusy?: boolean;
}

function statusTone(status: RlmRunStatus): { bg: string; fg: string; dot: string; pulse: boolean } {
  switch (status) {
    case "running":
      return { bg: "var(--accent-soft)", fg: "var(--accent-ink)", dot: "var(--accent)", pulse: true };
    case "completed":
      return { bg: "var(--chip)", fg: "var(--ink-2)", dot: "var(--ink)", pulse: false };
    case "partial":
      return { bg: "var(--warn-soft)", fg: "var(--warn-ink)", dot: "var(--warn)", pulse: false };
    case "failed":
      return { bg: "var(--err-soft)", fg: "var(--err)", dot: "var(--err)", pulse: false };
    default:
      // queued
      return { bg: "var(--chip)", fg: "var(--muted)", dot: "var(--muted-2)", pulse: false };
  }
}

export function RlmHeader({
  paperTitle,
  paperMeta,
  projectId,
  status,
  iterationCount,
  costUsd,
  warnings = [],
  lastHeartbeatAt = null,
  heartbeatNowMs = null,
  error = null,
  onRerun,
  rerunBusy = false,
}: RlmHeaderProps) {
  const tone = statusTone(status);
  const latestWarning = warnings.length > 0 ? warnings[warnings.length - 1] : null;

  // Transient inline error shown when the rerun POST itself fails.
  const [rerunError, setRerunError] = useState<string | null>(null);

  // Derive "wedged" signal: running + no heartbeat for >60 s.
  const noSignalSecs = useMemo(() => {
    if (status !== "running") return null;
    if (lastHeartbeatAt === null) return null;
    if (heartbeatNowMs === null) return null;
    const elapsed = heartbeatNowMs - new Date(lastHeartbeatAt).getTime();
    return elapsed > HEARTBEAT_STALE_MS ? Math.floor(elapsed / 1000) : null;
  }, [status, lastHeartbeatAt, heartbeatNowMs]);

  const handleRerun = onRerun
    ? async () => {
        setRerunError(null);
        try {
          await onRerun();
        } catch (err) {
          setRerunError(err instanceof Error ? err.message : "Rerun failed");
        }
      }
    : undefined;

  const showRerun = (status === "failed" || status === "completed") && handleRerun !== undefined;
  const rerunLabel = status === "completed" ? "Run again" : "Rerun";

  return (
    <div>
      <header className={styles.header}>
        <div className={styles.paperBlock}>
          <h1 className={styles.paperTitle}>{paperTitle}</h1>
          <p className={styles.paperMeta}>{paperMeta}</p>
        </div>

        <div className={styles.metaBlock}>
          <span className={styles.projectChip}>{projectId}</span>

          <span
            className={styles.statusPill}
            style={{ background: tone.bg, color: tone.fg }}
          >
            <span
              className={`${styles.statusDot}${tone.pulse ? ` ${styles.pulse}` : ""}`}
              style={{ background: tone.dot }}
            />
            {status}
          </span>

          {noSignalSecs !== null && (
            <span
              title={`No heartbeat received for ${noSignalSecs}s — the run may be wedged.`}
              style={{
                background: "var(--warn-soft)",
                color: "var(--warn)",
                borderRadius: "4px",
                padding: "2px 8px",
                fontSize: "0.72rem",
                fontWeight: 600,
                display: "inline-block",
                verticalAlign: "middle",
              }}
            >
              no signal {noSignalSecs}s
            </span>
          )}

          {latestWarning !== null && (
            <span
              title={latestWarning.message}
              style={{
                background: "var(--err-soft)",
                color: "var(--err)",
                borderRadius: "4px",
                padding: "2px 8px",
                fontSize: "0.72rem",
                fontWeight: 600,
                maxWidth: "220px",
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
                display: "inline-block",
                verticalAlign: "middle",
              }}
            >
              {latestWarning.message.length > 48
                ? latestWarning.message.slice(0, 48) + "…"
                : latestWarning.message}
            </span>
          )}

          <span className={styles.iterBlock}>
            <span className={styles.iterLabel}>iteration</span>
            <span className={styles.iterValue}>{iterationCount}</span>
          </span>

          {costUsd !== null && (
            <span className={styles.cost}>${costUsd.toFixed(2)}</span>
          )}

          {showRerun && (
            <button
              className={styles.rerunButton}
              onClick={handleRerun}
              disabled={rerunBusy}
              title={rerunBusy ? "Starting new run…" : `${rerunLabel} with the same paper`}
            >
              {rerunBusy ? "Starting…" : rerunLabel}
            </button>
          )}
        </div>
      </header>

      {status === "failed" && error && (
        <div className={styles.errorBanner} title={error}>
          <span className={styles.errorBannerText}>
            {error.length > ERROR_TRUNCATE ? error.slice(0, ERROR_TRUNCATE) + "…" : error}
          </span>
        </div>
      )}

      {rerunError && (
        <div className={styles.errorBanner}>
          <span className={styles.errorBannerText}>Rerun failed: {rerunError}</span>
        </div>
      )}
    </div>
  );
}
