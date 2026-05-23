"use client";

import { useMemo, useState } from "react";
import styles from "./leaderboard-table.module.css";

export interface LeaderboardRow {
  project_id: string;
  paper_id: string;
  paper_title: string | null;
  mode: "rlm" | "rdr";
  models: {
    planner: string | null;
    executor: string | null;
    verifier: string | null;
    grader: string | null;
  };
  overall_score: number | null;
  meets_target: boolean;
  degraded: boolean;
  cost_usd: number | null;
  iterations: number;
  wall_clock_s: number | null;
  sandbox: string | null;
  started_at: string | null;
  completed_at: string | null;
  verdict: string;
}

type SortKey =
  | "score"
  | "paper"
  | "mode"
  | "planner"
  | "executor"
  | "cost"
  | "iterations"
  | "time"
  | "verdict"
  | "completed";

type SortDir = "asc" | "desc";

const DEFAULT_DIR: Record<SortKey, SortDir> = {
  score: "desc",
  paper: "asc",
  mode: "asc",
  planner: "asc",
  executor: "asc",
  cost: "asc",
  iterations: "desc",
  time: "asc",
  verdict: "asc",
  completed: "desc",
};

function fmtElapsed(s: number | null): string | null {
  if (s === null) return null;
  if (s < 60) return `${Math.round(s)}s`;
  const m = Math.floor(s / 60);
  const sec = Math.round(s % 60);
  return `${m}m ${sec}s`;
}

function Dash() {
  return <span className={styles.dash} data-dash="true">—</span>;
}

interface LeaderboardTableProps {
  rows: LeaderboardRow[];
  error?: string | null;
}

export function LeaderboardTable({ rows, error = null }: LeaderboardTableProps) {
  const [sortKey, setSortKey] = useState<SortKey>("score");
  const [sortDir, setSortDir] = useState<SortDir>("desc");

  const sorted = useMemo(() => {
    const copy = [...rows];
    copy.sort((a, b) => {
      const mul = sortDir === "asc" ? 1 : -1;
      const ak = keyFor(a, sortKey);
      const bk = keyFor(b, sortKey);
      // Push nulls to the end regardless of direction.
      if (ak === null && bk === null) return 0;
      if (ak === null) return 1;
      if (bk === null) return -1;
      if (ak < bk) return -1 * mul;
      if (ak > bk) return 1 * mul;
      return 0;
    });
    return copy;
  }, [rows, sortKey, sortDir]);

  if (error) {
    return (
      <div className={styles.emptyState} role="status">
        {error}
      </div>
    );
  }

  if (rows.length === 0) {
    return (
      <div className={styles.emptyState}>
        No completed runs yet — start one from the lab.
      </div>
    );
  }

  const handleHeaderClick = (key: SortKey) => {
    if (key === sortKey) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir(DEFAULT_DIR[key]);
    }
  };

  const headers: Array<{ label: string; k: SortKey }> = [
    { label: "Paper", k: "paper" },
    { label: "Mode", k: "mode" },
    { label: "Planner", k: "planner" },
    { label: "Executor", k: "executor" },
    { label: "Score", k: "score" },
    { label: "Cost", k: "cost" },
    { label: "Iterations", k: "iterations" },
    { label: "Time", k: "time" },
    { label: "Verdict", k: "verdict" },
    { label: "Finished", k: "completed" },
  ];

  return (
    <div className={styles.tableWrap}>
      <table className={styles.table}>
        <thead>
          <tr>
            {headers.map(({ label, k }) => (
              <th
                key={k}
                className={styles.sortable}
                aria-sort={sortKey === k ? (sortDir === "asc" ? "ascending" : "descending") : "none"}
              >
                <button
                  type="button"
                  className={styles.sortButton}
                  onClick={() => handleHeaderClick(k)}
                >
                  {label}
                  {sortKey === k && (
                    <span aria-hidden="true" className={styles.sortIndicator}>
                      {sortDir === "asc" ? "▲" : "▼"}
                    </span>
                  )}
                </button>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((r) => {
            const scoreCls = r.meets_target
              ? styles.scoreCellMet
              : r.degraded
              ? styles.scoreCellDegraded
              : "";
            const elapsed = fmtElapsed(r.wall_clock_s);
            return (
              <tr key={r.project_id} data-project-id={r.project_id}>
                <td>
                  <a
                    className={styles.paperLink}
                    href={`/lab?projectId=${encodeURIComponent(r.project_id)}`}
                  >
                    {r.paper_title ?? r.paper_id}
                  </a>
                </td>
                <td><span className={styles.modeBadge}>{r.mode}</span></td>
                <td>{r.models.planner || <Dash />}</td>
                <td>{r.models.executor || <Dash />}</td>
                <td className={`${styles.numeric} ${scoreCls}`.trim()}>
                  {r.overall_score !== null ? r.overall_score.toFixed(2) : <Dash />}
                </td>
                <td className={styles.numeric}>
                  {r.cost_usd !== null ? `$${r.cost_usd.toFixed(2)}` : <Dash />}
                </td>
                <td className={styles.numeric}>{r.iterations}</td>
                <td className={styles.numeric}>{elapsed ?? <Dash />}</td>
                <td>{r.verdict}</td>
                <td>{r.completed_at ?? <Dash />}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function keyFor(r: LeaderboardRow, k: SortKey): string | number | null {
  switch (k) {
    case "score": return r.overall_score;
    case "paper": return r.paper_title ?? r.paper_id;
    case "mode": return r.mode;
    case "planner": return r.models.planner;
    case "executor": return r.models.executor;
    case "cost": return r.cost_usd;
    case "iterations": return r.iterations;
    case "time": return r.wall_clock_s;
    case "verdict": return r.verdict;
    case "completed": return r.completed_at;
  }
}
