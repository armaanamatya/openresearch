"use client";

import { useEffect, useState } from "react";

import type { LiveDemoRunState } from "@/lib/demo/demo-run-types";
import {
  PIPELINE_STAGES,
  computeProgress,
  formatDuration,
  STALL_THRESHOLD_SECONDS
} from "@/lib/demo/progress";

interface ProgressStripProps {
  run: LiveDemoRunState | null;
  /** Override now() for tests; defaults to a 1 Hz live ticker. */
  nowMs?: number;
}

export function ProgressStrip({ run, nowMs }: ProgressStripProps) {
  // Tick at 1 Hz so the elapsed timers move while the run is in flight.
  const [tick, setTick] = useState(() => Date.now());
  useEffect(() => {
    if (nowMs !== undefined) return;
    if (!run || (run.status !== "running" && run.status !== "queued")) return;
    const id = setInterval(() => setTick(Date.now()), 1000);
    return () => clearInterval(id);
  }, [nowMs, run?.status]);

  if (!run) return null;

  const snapshot = computeProgress(run, nowMs ?? tick);
  const percent = Math.round(snapshot.percentComplete * 100);
  const stallSecsRemaining = snapshot.lastActivitySeconds === null
    ? null
    : Math.max(0, STALL_THRESHOLD_SECONDS - snapshot.lastActivitySeconds);

  return (
    <section
      className="rounded-2xl border border-zinc-200 bg-white p-4 shadow-sm"
      aria-label="Pipeline progress"
    >
      <header className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <StatusDot status={run.status} stalled={snapshot.isStalled} />
          <div>
            <p className="text-sm font-semibold text-zinc-900">
              {snapshot.currentStageLabel}
            </p>
            <p className="text-xs text-zinc-500" data-testid="last-activity">
              {snapshot.lastActivityText
                ? `${snapshot.lastActivityText}`
                : run.status === "running"
                  ? "Waiting for first agent activity…"
                  : "No activity yet"}
            </p>
          </div>
        </div>

        <div className="flex flex-col items-end gap-1 text-right">
          <span className="text-sm font-medium text-zinc-900" data-testid="elapsed">
            Total: {formatDuration(snapshot.elapsedSeconds)}
          </span>
          <span className="text-xs text-zinc-500" data-testid="last-activity-elapsed">
            Last tick: {formatDuration(snapshot.lastActivitySeconds)} ago
          </span>
        </div>
      </header>

      <div className="mt-3">
        <div className="h-1.5 w-full overflow-hidden rounded-full bg-zinc-100">
          <div
            className={`h-full rounded-full transition-all duration-500 ease-out ${
              snapshot.isStalled
                ? "bg-amber-500"
                : run.status === "failed"
                  ? "bg-rose-500"
                  : run.status === "completed"
                    ? "bg-emerald-500"
                    : "bg-sky-500"
            }`}
            style={{ width: `${percent}%` }}
            role="progressbar"
            aria-valuenow={percent}
            aria-valuemin={0}
            aria-valuemax={100}
          />
        </div>
        <div className="mt-2 grid grid-cols-6 gap-1 text-[10px] uppercase tracking-wide text-zinc-400">
          {PIPELINE_STAGES.map((stage, idx) => {
            const reached = idx < snapshot.completedStageCount;
            const current = stage.key === snapshot.currentStageKey && !reached;
            return (
              <span
                key={stage.key}
                className={
                  reached
                    ? "text-emerald-600"
                    : current
                      ? "font-semibold text-sky-600"
                      : ""
                }
                title={stage.label}
              >
                {idx + 1}. {stage.label}
              </span>
            );
          })}
        </div>
      </div>

      {snapshot.isStalled ? (
        <p
          className="mt-3 rounded-md bg-amber-50 px-3 py-2 text-xs text-amber-900"
          data-testid="stall-warning"
        >
          No agent activity for {formatDuration(snapshot.lastActivitySeconds)} —
          this stage may be stuck. Check the runner log below for the last tool call.
        </p>
      ) : run.status === "running" && stallSecsRemaining !== null && stallSecsRemaining < 30 ? (
        <p className="mt-3 text-xs text-zinc-500">
          Stall warning will trigger in {stallSecsRemaining}s if no new activity.
        </p>
      ) : null}

      {snapshot.failureText ? (
        <p
          className="mt-3 rounded-md bg-rose-50 px-3 py-2 text-xs text-rose-900"
          data-testid="failure"
        >
          {snapshot.failureText}
        </p>
      ) : null}
    </section>
  );
}

function StatusDot({
  status,
  stalled
}: {
  status: LiveDemoRunState["status"];
  stalled: boolean;
}) {
  let cls = "bg-zinc-300";
  if (stalled) cls = "bg-amber-500 animate-pulse";
  else if (status === "running" || status === "queued") cls = "bg-sky-500 animate-pulse";
  else if (status === "completed") cls = "bg-emerald-500";
  else if (status === "failed") cls = "bg-rose-500";
  else if (status === "stopped") cls = "bg-zinc-400";
  return <span className={`inline-block h-2.5 w-2.5 rounded-full ${cls}`} aria-hidden />;
}
