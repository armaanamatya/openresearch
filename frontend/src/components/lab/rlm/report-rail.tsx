"use client";

import { useState } from "react";
import type { CSSProperties } from "react";
import type { RlmRunState, PrimitiveCallView } from "../../../hooks/use-rlm-run";
import type { DemoWorkerReport, DemoReportsSummary } from "../../../lib/demo/demo-run-types";
import styles from "./report-rail.module.css";

export interface ReportRailProps {
  status: RlmRunState["status"];
  elapsedMs: number;
  report: RlmRunState["report"];
  rubric: RlmRunState["rubric"];
  workerReports?: DemoWorkerReport[];
  primitiveCalls?: PrimitiveCallView[];
  reportsSummary?: DemoReportsSummary;
  style?: CSSProperties;
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

function compactList(items: string[] | undefined, fallback = "None"): string {
  const clean = (items ?? []).filter(Boolean);
  if (clean.length === 0) return fallback;
  return clean.slice(0, 2).join("; ");
}

function proceduresLabel(value: boolean | null | undefined): string {
  if (value === true) return "followed";
  if (value === false) return "not followed";
  return "unconfirmed";
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
  workerReports = [],
  primitiveCalls = [],
  reportsSummary,
  style,
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
    <aside className={styles.rail} style={style} aria-label="Report rail">
      {/* ── Verdict pill ─────────────────────────────────────── */}
      <div
        className={`${styles.verdict} ${verdictClass}`}
        title={`Verdict ${verdictLabel(status)}. This follows the live run status and final report verdict.`}
      >
        {verdictLabel(status)}
      </div>

      {/* ── Rubric score header + degraded note ──────────────── */}
      <div className={styles.scoreHeader}>
        <span
          className={styles.scoreLabel}
          title="Current rubric score. It is blank until verification or final report generation emits a score."
        >
          rubric score
        </span>
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
        <div
          className={styles.tile}
          title="Root REPL turns completed. Count is finalized when final_report.json is written."
        >
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
        <div
          className={styles.tile}
          title="Primitive tool calls observed in the run. Count is finalized in the report."
        >
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
        <div
          className={styles.tile}
          title="Improvement candidates proposed by the RLM loop. Usually appears after baseline verification."
        >
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
        <div
          className={styles.tile}
          title="Proposed candidates that improved or were accepted by the loop."
        >
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
          <div
            className={styles.tile}
            title="LLM cost reported by the run. OAuth subscription paths may report $0.00."
          >
            <span className={styles.tileValue}>
              <span>${report!.costUsd!.toFixed(2)}</span>
            </span>
            <span className={styles.tileLabel}>cost</span>
          </div>
        )}

        {/* Elapsed */}
        <div className={styles.tile} title="Wall-clock elapsed time since kickoff.">
          <span className={styles.tileValue}>
            <span>{fmtElapsed(elapsedMs)}</span>
          </span>
          <span className={styles.tileLabel}>elapsed</span>
        </div>
      </div>

      {report === null && (
        <p className={styles.emptyNote}>
          Final report counts appear when final_report.json is ready; live activity and phase state continue above while the run is active.
        </p>
      )}

      <section className={styles.workerReports} aria-label="Worker reports">
        <div className={styles.sectionHeader}>
          <span className={styles.breakdownHeading}>worker reports</span>
          <span className={styles.sectionCount}>{workerReports.length}</span>
        </div>

        {/* Summary card when reportsSummary is available */}
        {reportsSummary && reportsSummary.total_workers != null && reportsSummary.total_workers > 0 && (
          <div className={styles.workerCard}>
            <div className={styles.workerTitleRow}>
              <span className={styles.workerName}>summary</span>
              <span className={styles.workerStatus}>
                {reportsSummary.final_run_status ?? "unknown"}
              </span>
            </div>
            <dl className={styles.workerFacts}>
              <div>
                <dt>workers</dt>
                <dd>
                  {reportsSummary.total_workers} total
                  {reportsSummary.by_status && Object.entries(reportsSummary.by_status)
                    .filter(([, count]) => count > 0)
                    .map(([st, count]) => ` / ${count} ${st}`)
                    .join("")}
                </dd>
              </div>
              {(reportsSummary.critical_blockers?.length ?? 0) > 0 && (
                <div>
                  <dt>blockers</dt>
                  <dd className={styles.blockerText}>
                    {reportsSummary.critical_blockers!.slice(0, 2).map((b) => b.title).join("; ")}
                  </dd>
                </div>
              )}
              <div>
                <dt>commands</dt>
                <dd>{reportsSummary.commands_run ?? 0} run, {reportsSummary.failed_commands ?? 0} failed</dd>
              </div>
            </dl>
          </div>
        )}

        {workerReports.length === 0 ? (
          <p className={styles.emptyNote}>
            Worker summaries appear after each agent finishes.
          </p>
        ) : (
          <ul className={styles.workerList}>
            {workerReports.slice(-10).reverse().map((worker, index) => (
              <WorkerReportCard
                key={worker.report_id ?? `${worker.agent_id ?? "worker"}-${index}`}
                worker={worker}
              />
            ))}
          </ul>
        )}
      </section>

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

      {/* ── Experiment run registry (FIG § 4.3) ─────────────── */}
      <ExperimentRegistry primitiveCalls={primitiveCalls} />
    </aside>
  );
}

// ─── Experiment registry ──────────────────────────────────────────────────────

function runStatusLabel(status: PrimitiveCallView["status"]): string {
  return status === "start" ? "RUNNING" : status === "ok" ? "DONE" : "FAILED";
}

function ExperimentRegistry({ primitiveCalls }: { primitiveCalls: PrimitiveCallView[] }) {
  const runs = primitiveCalls.filter((c) => c.primitive === "run_experiment");
  if (runs.length === 0) return null;

  return (
    <section className={styles.breakdown} aria-label="Experiment runs">
      <div className={styles.sectionHeader}>
        <p className={styles.breakdownHeading}>experiments</p>
        <span className={styles.sectionCount}>{runs.length}</span>
      </div>
      <ul className={styles.runList}>
        {runs.map((call, i) => {
          const isRunning = call.status === "start";
          const isOk = call.status === "ok";
          const statusLabel = runStatusLabel(call.status);
          const result = call.result_summary
            ? call.result_summary.slice(0, 48)
            : null;
          return (
            <li key={`run-${i}`} className={styles.runRow}>
              <span className={styles.runIdx}>{String(i + 1).padStart(2, "0")}</span>
              <span className={styles.runTask} title={result ?? undefined}>
                {result ?? "run experiment"}
              </span>
              <span
                className={
                  isRunning
                    ? styles.runBadgeRunning
                    : isOk
                    ? styles.runBadgeMatch
                    : styles.runBadgeFailed
                }
              >
                {statusLabel}
              </span>
            </li>
          );
        })}
      </ul>
    </section>
  );
}

// ─── WorkerReportCard ──────────────────────────────────────────────────────

function WorkerReportCard({ worker }: { worker: DemoWorkerReport }) {
  const [showRaw, setShowRaw] = useState(false);
  const commands = worker.commands ?? [];
  const blockers = worker.blockers ?? [];
  const artifacts = worker.artifacts ?? [];
  const hasBlockers = blockers.length > 0;

  const statusClass =
    worker.status === "failed" || worker.status === "blocked"
      ? styles.workerFailed
      : worker.status === "running"
      ? styles.verdictRunning
      : styles.workerStatus;

  return (
    <li className={styles.workerCard} data-testid="worker-report-card">
      <div className={styles.workerTitleRow}>
        <span className={styles.workerName}>
          {worker.worker_type ? `[${worker.worker_type}] ` : ""}
          {worker.agent_id ?? "worker"}
        </span>
        <span className={statusClass} data-testid="worker-status-badge">
          {worker.status ?? "reported"}
        </span>
      </div>

      {/* Assignment summary */}
      {worker.assignment?.summary && (
        <div className={styles.assignmentSummary}>
          {worker.assignment.summary}
        </div>
      )}

      {/* Duration */}
      {worker.duration_ms != null && (
        <div className={styles.durationBadge}>
          {fmtElapsed(worker.duration_ms)}
        </div>
      )}

      <dl className={styles.workerFacts}>
        {/* Execution summary when available */}
        {worker.execution_summary?.concise_summary && (
          <div>
            <dt>summary</dt>
            <dd>{worker.execution_summary.concise_summary}</dd>
          </div>
        )}

        <div>
          <dt>implemented</dt>
          <dd>{compactList(worker.implemented ?? worker.execution_summary?.implemented)}</dd>
        </div>
        <div>
          <dt>undone</dt>
          <dd>{compactList(worker.left_undone ?? worker.execution_summary?.not_implemented)}</dd>
        </div>
        <div>
          <dt>commands</dt>
          <dd>
            {commands.length === 0
              ? "None"
              : commands.slice(0, 3).map((cmd, i) => (
                  <span key={`${cmd.command}-${i}`} className={styles.commandLine}>
                    {cmd.command.length > 60 ? cmd.command.slice(0, 57) + "..." : cmd.command}
                    <span className={styles.exitCode} data-testid="command-exit-code">
                      {cmd.exit_code == null ? "exit ?" : `exit ${cmd.exit_code}`}
                    </span>
                  </span>
                ))}
          </dd>
        </div>

        {/* Blockers */}
        {hasBlockers && (
          <div>
            <dt>blockers</dt>
            <dd>
              {blockers.map((b, i) => (
                <span key={`blocker-${i}`} className={styles.blockerText} data-testid="worker-blocker">
                  <strong>[{b.severity}]</strong> {b.title}
                  {b.source && <span className={styles.exitCode}> ({b.source})</span>}
                </span>
              ))}
            </dd>
          </div>
        )}

        {/* Artifacts */}
        {artifacts.length > 0 && (
          <div>
            <dt>artifacts</dt>
            <dd>{artifacts.slice(0, 3).map((a) => a.path).join(", ")}</dd>
          </div>
        )}

        <div>
          <dt>issues</dt>
          <dd>{compactList(worker.issues)}</dd>
        </div>
        <div>
          <dt>procedures</dt>
          <dd>{proceduresLabel(worker.procedures_followed)}</dd>
        </div>
      </dl>

      {/* Raw JSON disclosure */}
      <details
        className={styles.rawJsonDetails}
        data-testid="raw-json-disclosure"
        open={showRaw}
        onToggle={(e) => setShowRaw((e.target as HTMLDetailsElement).open)}
      >
        <summary className={styles.rawJsonSummary}>raw json</summary>
        <pre className={styles.rawJsonPre}>
          {JSON.stringify(worker, null, 2)}
        </pre>
      </details>
    </li>
  );
}
