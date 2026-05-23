"use client";

import type { RlmRunState } from "../../../hooks/use-rlm-run";
import styles from "./report-rail.module.css";

export interface ReportRailProps {
  status: RlmRunState["status"];
  elapsedMs: number;
  report: RlmRunState["report"];
  rubric: RlmRunState["rubric"];
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/** Format elapsed milliseconds to a human-readable string: "1h 12m", "3m 5s", "42s", "0s". */
function fmtElapsed(ms: number): string {
  const totalSec = Math.floor(ms / 1000);
  const h = Math.floor(totalSec / 3600);
  const m = Math.floor((totalSec % 3600) / 60);
  const s = totalSec % 60;
  const parts: string[] = [];
  if (h > 0) parts.push(`${h}h`);
  if (m > 0) parts.push(`${m}m`);
  if (s > 0 || parts.length === 0) parts.push(`${s}s`);
  return parts.join(" ");
}

/** Sentence-case a status value for display. */
function verdictLabel(status: RlmRunState["status"]): string {
  switch (status) {
    case "running":
      return "in progress";
    case "queued":
      return "Queued";
    case "completed":
      return "Completed";
    case "partial":
      return "Partial";
    case "failed":
      return "Failed";
  }
}

/** Area status → glyph marker. */
function areaMarker(status: "pass" | "partial" | "fail"): string {
  switch (status) {
    case "pass":
      return "✓";
    case "partial":
      return "◐";
    case "fail":
      return "✗";
  }
}

// ─── Component ────────────────────────────────────────────────────────────────

/**
 * ReportRail — the right rail of the RLM lab (spec §7, §9, §14).
 *
 * Renders:
 *  - A verdict pill (status-driven).
 *  - Rubric score header + degraded note when current <= 0.35.
 *  - A 6-tile stat grid: iterations / primitive calls / proposed / promoted /
 *    cost (only when costUsd != null) / elapsed.
 *    When report === null (run still going), count tiles show "—".
 *  - A rubric breakdown — one row per rubric.areas entry with a ✓/◐/✗ marker.
 *
 * Honesty rules (spec §14):
 *  - Cost tile is omitted entirely when costUsd is null.
 *  - When rubric.current <= 0.35, a "degraded — no real metrics" note is shown.
 *  - When report === null, count tiles render a dimmed "—" placeholder.
 *  - No fabricated numbers.
 */
export function ReportRail({
  status,
  elapsedMs,
  report,
  rubric,
}: ReportRailProps) {
  const hasCost = report?.costUsd != null;
  const isDegraded =
    rubric.current !== null && rubric.current <= 0.35;

  // Verdict pill style class.
  const verdictClass =
    status === "completed"
      ? styles.verdictPass
      : status === "partial"
      ? styles.verdictPartial
      : status === "failed"
      ? styles.verdictFail
      : styles.verdictRunning;

  return (
    <aside className={styles.rail} aria-label="Report rail">
      {/* ── Verdict pill ─────────────────────────────────────── */}
      <div className={`${styles.verdict} ${verdictClass}`}>
        {verdictLabel(status)}
      </div>

      {/* ── Rubric score header + degraded note ──────────────── */}
      <div className={styles.scoreHeader}>
        <span className={styles.scoreLabel}>rubric score</span>
        <span className={styles.scoreValue}>
          {rubric.current !== null ? rubric.current.toFixed(2) : "—"}
        </span>
        {isDegraded && (
          <span className={styles.degradedNote}>
            ≤ 0.35 — may be a degraded run with no measured metrics
          </span>
        )}
      </div>

      {/* ── 6-tile stat grid ─────────────────────────────────── */}
      <div className={styles.grid}>
        {/* Iterations */}
        <div className={styles.tile}>
          <span className={styles.tileValue}>
            {report !== null ? (
              <span>{report.counts.iterations}</span>
            ) : (
              <span className={styles.tilePlaceholder}>—</span>
            )}
          </span>
          <span className={styles.tileLabel}>iterations</span>
        </div>

        {/* Primitive calls */}
        <div className={styles.tile}>
          <span className={styles.tileValue}>
            {report !== null ? (
              <span>{report.counts.primitiveCalls}</span>
            ) : (
              <span className={styles.tilePlaceholder}>—</span>
            )}
          </span>
          <span className={styles.tileLabel}>primitive calls</span>
        </div>

        {/* Proposed */}
        <div className={styles.tile}>
          <span className={styles.tileValue}>
            {report !== null ? (
              <span>{report.counts.proposed}</span>
            ) : (
              <span className={styles.tilePlaceholder}>—</span>
            )}
          </span>
          <span className={styles.tileLabel}>proposed</span>
        </div>

        {/* Promoted */}
        <div className={styles.tile}>
          <span className={styles.tileValue}>
            {report !== null ? (
              <span>{report.counts.promoted}</span>
            ) : (
              <span className={styles.tilePlaceholder}>—</span>
            )}
          </span>
          <span className={styles.tileLabel}>promoted</span>
        </div>

        {/* Cost — omitted entirely when costUsd is null (spec §14 / M4) */}
        {hasCost && (
          <div className={styles.tile}>
            <span className={styles.tileValue}>
              <span>${report!.costUsd!.toFixed(2)}</span>
            </span>
            <span className={styles.tileLabel}>cost</span>
          </div>
        )}

        {/* Elapsed */}
        <div className={styles.tile}>
          <span className={styles.tileValue}>
            <span>{fmtElapsed(elapsedMs)}</span>
          </span>
          <span className={styles.tileLabel}>elapsed</span>
        </div>
      </div>

      {/* ── Rubric breakdown ─────────────────────────────────── */}
      {rubric.areas.length > 0 && (
        <section className={styles.breakdown} aria-label="Rubric breakdown">
          <p className={styles.breakdownHeading}>rubric breakdown</p>
          <ul className={styles.areaList}>
            {rubric.areas.map((a, i) => (
              <li
                key={a.area || `__area_${i}`}
                className={`${styles.areaRow} ${
                  a.status === "pass"
                    ? styles.areaPass
                    : a.status === "partial"
                    ? styles.areaPartial
                    : styles.areaFail
                }`}
              >
                <span className={styles.areaMarker} aria-hidden="true">
                  {areaMarker(a.status)}
                </span>
                <span className={styles.srOnly}>{a.status}</span>
                <span className={styles.areaName}>{a.area || "—"}</span>
                <span className={styles.areaScore}>
                  {a.score.toFixed(2)}
                </span>
              </li>
            ))}
          </ul>
        </section>
      )}
    </aside>
  );
}
