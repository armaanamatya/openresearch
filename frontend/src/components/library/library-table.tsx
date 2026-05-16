"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import type { RunSummary } from "@/lib/runs/server-list";
import { LibraryFilters, type LibraryFiltersValue } from "./library-filters";
import styles from "./library-table.module.css";

export interface LibraryTableProps {
  initialRuns: RunSummary[];
  initialParams: { status?: string; q?: string; order_by?: string };
}

// Forest accent for completed runs only — every other status reads as
// neutral. This is what gives the table its "scan for green" rhythm:
// the eye picks out finished work without the rest of the rows
// competing for attention.
function statusClass(status: string): string {
  if (status === "completed") return `${styles.status} ${styles.statusCompleted}`;
  if (status === "running") return `${styles.status} ${styles.statusRunning}`;
  if (status === "failed") return `${styles.status} ${styles.statusFailed}`;
  if (status === "stopped") return `${styles.status} ${styles.statusStopped}`;
  return styles.status;
}

const RELATIVE_THRESHOLDS: { ms: number; unit: Intl.RelativeTimeFormatUnit; div: number }[] = [
  { ms: 60_000, unit: "second", div: 1_000 },
  { ms: 3_600_000, unit: "minute", div: 60_000 },
  { ms: 86_400_000, unit: "hour", div: 3_600_000 },
  { ms: 604_800_000, unit: "day", div: 86_400_000 },
  { ms: 2_592_000_000, unit: "week", div: 604_800_000 },
  { ms: 31_536_000_000, unit: "month", div: 2_592_000_000 }
];

function formatRelative(iso: string | undefined): string {
  if (!iso) return "—";
  const then = new Date(iso).getTime();
  if (!Number.isFinite(then)) return "—";
  const diff = Date.now() - then;
  if (diff < 5_000) return "just now";
  // date-fns-style: relative when recent, otherwise a brief locale stamp.
  for (const t of RELATIVE_THRESHOLDS) {
    if (diff < t.ms) {
      const value = -Math.round(diff / t.div);
      try {
        return new Intl.RelativeTimeFormat(undefined, { numeric: "auto" }).format(value, t.unit);
      } catch {
        return new Date(iso).toLocaleString();
      }
    }
  }
  return new Date(iso).toLocaleString();
}

function formatNumber(n: number | undefined, fractionDigits = 3): string {
  if (typeof n !== "number" || !Number.isFinite(n)) return "—";
  return n.toFixed(fractionDigits);
}

function formatDelta(n: number | undefined): string {
  if (typeof n !== "number" || !Number.isFinite(n)) return "—";
  // Show sign so "did it beat or miss the target?" reads at a glance.
  const sign = n > 0 ? "+" : n < 0 ? "" : "";
  return `${sign}${n.toFixed(3)}`;
}

export function LibraryTable({ initialRuns, initialParams }: LibraryTableProps) {
  const [filters, setFilters] = useState<LibraryFiltersValue>({
    status: initialParams.status,
    q: initialParams.q,
    order_by: initialParams.order_by ?? "updated_at"
  });
  const [runs, setRuns] = useState<RunSummary[]>(initialRuns);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Track the latest in-flight request so out-of-order responses can't
  // overwrite fresher state. Without this, a slow request for "all"
  // arriving after a fast request for "completed" would clobber the
  // narrower result.
  const requestSeqRef = useRef(0);
  const isFirstRef = useRef(true);

  const queryKey = useMemo(
    () =>
      JSON.stringify({
        status: filters.status ?? "",
        q: filters.q ?? "",
        order_by: filters.order_by
      }),
    [filters.status, filters.q, filters.order_by]
  );

  useEffect(() => {
    // Skip the very first run: the page handed us a server-rendered
    // dataset that already matches initialParams. Re-fetching on mount
    // just duplicates the request.
    if (isFirstRef.current) {
      isFirstRef.current = false;
      return;
    }
    const seq = ++requestSeqRef.current;
    setLoading(true);
    setError(null);
    const qs = new URLSearchParams();
    qs.set("limit", "100");
    if (filters.status) qs.set("status", filters.status);
    if (filters.q) qs.set("q", filters.q);
    qs.set("order_by", filters.order_by);
    fetch(`/api/runs/list?${qs.toString()}`, { cache: "no-store" })
      .then(async (res) => {
        if (seq !== requestSeqRef.current) return;
        if (!res.ok) {
          setError("Could not load runs.");
          setRuns([]);
          return;
        }
        const body = (await res.json()) as RunSummary[];
        setRuns(Array.isArray(body) ? body : []);
      })
      .catch(() => {
        if (seq !== requestSeqRef.current) return;
        setError("Could not load runs.");
        setRuns([]);
      })
      .finally(() => {
        if (seq === requestSeqRef.current) setLoading(false);
      });
  }, [queryKey, filters.status, filters.q, filters.order_by]);

  const handleFiltersChange = useCallback((next: LibraryFiltersValue) => {
    setFilters(next);
  }, []);

  return (
    <>
      <header className={styles.header}>
        <div className="eyebrow">library</div>
        <h1 className={styles.title}>Runs</h1>
        <p className={styles.subtitle}>
          {runs.length === 0 && !loading
            ? "No runs match the current filters."
            : `${runs.length} run${runs.length === 1 ? "" : "s"}`}
          {loading ? " · loading…" : null}
        </p>
      </header>
      <LibraryFilters value={filters} onChange={handleFiltersChange} />
      {error ? <div className={styles.error}>{error}</div> : null}
      <div className={styles.tableWrap}>
        <table className={styles.table}>
              <thead>
                {/* Cost column intentionally absent — backend has no USD
                    field on LiveDemoRunState. Plumb it through cost-ledger
                    rollup when that surface is built. */}
                <tr>
                  <th className={styles.thLeft}>Project</th>
                  <th className={styles.thLeft}>Source</th>
                  <th className={styles.thLeft}>Status</th>
                  <th className={styles.thLeft}>Updated</th>
                  <th className={styles.thRight}>Reward Δ</th>
                  <th className={styles.thRight}>Score</th>
                </tr>
              </thead>
              <tbody>
                {runs.length === 0 && !loading ? (
                  <tr>
                    <td className={styles.emptyCell} colSpan={6}>
                      No runs to show. Try clearing the filters or starting a new run from the lab.
                    </td>
                  </tr>
                ) : (
                  runs.map((run) => (
                    <tr key={run.projectId} className={styles.row}>
                      <td className={styles.tdLeft}>
                        <a
                          className={styles.projectLink}
                          href={`/lab?projectId=${encodeURIComponent(run.projectId)}`}
                        >
                          <span className={styles.projectId}>{run.projectId.slice(0, 14)}</span>
                        </a>
                      </td>
                      <td className={styles.tdLeft}>
                        <span className={styles.source} title={run.sourceLabel ?? ""}>
                          {run.sourceLabel ?? "—"}
                        </span>
                      </td>
                      <td className={styles.tdLeft}>
                        <span className={statusClass(run.status)}>{run.status}</span>
                      </td>
                      <td className={`${styles.tdLeft} ${styles.tdMuted}`}>
                        <time
                          dateTime={run.updatedAt ?? ""}
                          title={run.updatedAt ? new Date(run.updatedAt).toLocaleString() : ""}
                        >
                          {formatRelative(run.updatedAt)}
                        </time>
                      </td>
                      <td className={`${styles.tdRight} ${styles.tdMono}`}>
                        {formatDelta(run.benchmark?.deltaValue)}
                      </td>
                      <td className={`${styles.tdRight} ${styles.tdMono}`}>
                        {formatNumber(run.benchmark?.overallScore)}
                      </td>
                    </tr>
                  ))
                )}
          </tbody>
        </table>
      </div>
    </>
  );
}
