"use client";

import Link from "next/link";
import type { LeaderboardRow } from "@/lib/leaderboard/types";
import styles from "./recent-runs-panel.module.css";

function Dash() {
  return <span className={styles.dash} data-dash="true">—</span>;
}

function StatusDot({ status }: { status: string }) {
  const cls =
    status === "completed"
      ? styles.dotGreen
      : status === "failed" || status === "interrupted"
      ? styles.dotRed
      : styles.dotAmber;
  return <span className={`${styles.dot} ${cls}`} aria-hidden="true" />;
}

interface RecentRunsPanelProps {
  rows: LeaderboardRow[];
  interruptedRows?: LeaderboardRow[];
  error?: string | null;
  limit?: number;
  onResume?: (projectId: string) => void;
}

export function RecentRunsPanel({ rows, interruptedRows = [], error = null, onResume }: RecentRunsPanelProps) {
  return (
    <section className={styles.panel}>
      <div className={styles.header}>
        <span className={styles.title}>Recent runs</span>
        <Link href="/leaderboard" className={styles.viewAll} prefetch={false}>
          View all →
        </Link>
      </div>

      {interruptedRows.length > 0 && onResume && (
        <ul className={styles.list} role="list" aria-label="Interrupted runs">
          {interruptedRows.map((r) => (
            <li key={r.project_id} className={styles.row}>
              <div className={styles.rowMain}>
                <div className={styles.rowTop}>
                  <span className={styles.paperLink}>{r.paper_title ?? r.paper_id}</span>
                  <span className={styles.modeBadge}>interrupted</span>
                </div>
              </div>
              <div className={styles.rowRight}>
                <button
                  type="button"
                  className={styles.replayLink}
                  onClick={() => onResume(r.project_id)}
                  aria-label={`Resume run for ${r.paper_title ?? r.paper_id}`}
                  title="Restarts this run under the same project — implementation work may be reused, the reasoning loop starts over."
                >
                  Resume
                </button>
              </div>
            </li>
          ))}
        </ul>
      )}

      {error ? (
        <p className={styles.empty} role="status">{error}</p>
      ) : rows.length === 0 ? (
        <p className={styles.empty}>No runs yet — start one above.</p>
      ) : (
        <ul className={styles.list} role="list">
          {rows.map((r) => {
            const score = r.compute_adjusted_score;
            const scoreCls = r.meets_target
              ? styles.scoreMet
              : r.degraded
              ? styles.scoreDegraded
              : "";
            const scoreTitle = (() => {
              const parts: string[] = [];
              if (r.attempts > 0) parts.push(`best of ${r.attempts} attempt${r.attempts === 1 ? "" : "s"}`);
              if (r.overall_score !== null) parts.push(`raw: ${r.overall_score.toFixed(2)}`);
              return parts.join(" · ") || undefined;
            })();

            return (
              <li key={r.project_id} className={styles.row}>
                <div className={styles.rowMain}>
                  <div className={styles.rowTop}>
                    <Link
                      href={`/lab?projectId=${encodeURIComponent(r.project_id)}`}
                      className={styles.paperLink}
                    >
                      {r.paper_title ?? r.paper_id}
                    </Link>
                    <span className={styles.modeBadge}>{r.mode}</span>
                  </div>

                  <div className={styles.rowMeta}>
                    <StatusDot status={r.status} />
                    <span className={styles.verdict}>{r.verdict}</span>
                    {r.completed_at && (
                      <span className={styles.completedAt}>{r.completed_at}</span>
                    )}
                  </div>
                </div>

                <div className={styles.rowRight}>
                  <span
                    className={`${styles.score} ${scoreCls}`}
                    title={scoreTitle}
                  >
                    {score !== null ? (
                      <>
                        {score.toFixed(2)}
                        {r.attempts > 1 && (
                          <sup className={styles.attempts} title={scoreTitle}>
                            ×{r.attempts}
                          </sup>
                        )}
                      </>
                    ) : (
                      <Dash />
                    )}
                  </span>

                  <Link
                    href={`/lab?replay=${encodeURIComponent(r.project_id)}`}
                    className={styles.replayLink}
                    prefetch={false}
                    aria-label={`Replay run for ${r.paper_title ?? r.paper_id}`}
                  >
                    Replay
                  </Link>
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}
