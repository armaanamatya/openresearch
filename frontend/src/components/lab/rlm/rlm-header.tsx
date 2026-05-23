"use client";

import styles from "./rlm-header.module.css";

export type RlmRunStatus = "queued" | "running" | "completed" | "partial" | "failed";

interface RlmHeaderProps {
  paperTitle: string;
  paperMeta: string;
  projectId: string;
  status: RlmRunStatus;
  iterationCount: number;
  costUsd: number | null;
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
}: RlmHeaderProps) {
  const tone = statusTone(status);

  return (
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

        <span className={styles.iterBlock}>
          <span className={styles.iterLabel}>iteration</span>
          <span className={styles.iterValue}>{iterationCount}</span>
        </span>

        {costUsd !== null && (
          <span className={styles.cost}>${costUsd.toFixed(2)}</span>
        )}
      </div>
    </header>
  );
}
