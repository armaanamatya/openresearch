"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import { DashboardShell } from "@/features/dashboard/dashboard-shell";
import type { LiveDemoRunState } from "@/lib/demo/demo-run-types";
import { createMockEventAdapter } from "@/lib/events/mock-event-adapter";

interface LiveDemoClientProps {
  initialRun: LiveDemoRunState | null;
}

const POLL_INTERVAL_MS = 3000;

function formatStatus(status: LiveDemoRunState["status"] | "idle") {
  switch (status) {
    case "queued":
      return "Queued";
    case "running":
      return "Running";
    case "completed":
      return "Completed";
    case "failed":
      return "Failed";
    default:
      return "Idle";
  }
}

function statusTone(status: LiveDemoRunState["status"] | "idle") {
  switch (status) {
    case "queued":
      return "border-amber-300/30 bg-amber-300/10 text-amber-100";
    case "running":
      return "border-sky-300/30 bg-sky-300/10 text-sky-100";
    case "completed":
      return "border-emerald-300/30 bg-emerald-300/10 text-emerald-100";
    case "failed":
      return "border-rose-300/30 bg-rose-300/10 text-rose-100";
    default:
      return "border-white/10 bg-white/5 text-stone-200";
  }
}

export function LiveDemoClient({ initialRun }: LiveDemoClientProps) {
  const [run, setRun] = useState(initialRun);
  const [runningMode, setRunningMode] = useState<"offline" | "sdk" | null>(
    initialRun && (initialRun.status === "queued" || initialRun.status === "running")
      ? initialRun.runMode
      : null
  );
  const [error, setError] = useState<string | null>(null);
  const pollTimer = useRef<number | null>(null);

  const adapter = useMemo(() => {
    if (!run?.payload) {
      return null;
    }

    return createMockEventAdapter({
      snapshot: run.payload.initialSnapshot,
      events: run.payload.events
    });
  }, [run]);

  useEffect(() => {
    if (run?.status !== "queued" && run?.status !== "running") {
      setRunningMode(null);
      if (pollTimer.current) {
        window.clearTimeout(pollTimer.current);
        pollTimer.current = null;
      }
      return;
    }

    const projectId = run.projectId;
    pollTimer.current = window.setTimeout(async () => {
      try {
        const response = await fetch(`/api/demo?projectId=${projectId}&mode=${run.runMode}`, {
          cache: "no-store"
        });
        if (!response.ok) {
          throw new Error(`Status check failed with ${response.status}`);
        }

        const next = (await response.json()) as LiveDemoRunState | null;
        if (next) {
          setRun(next);
          if (next.status === "failed") {
            setError(next.error ?? "Demo run failed");
          }
        }
      } catch (pollError) {
        setError(
          pollError instanceof Error ? pollError.message : "Unable to refresh demo run"
        );
      }
    }, POLL_INTERVAL_MS);

    return () => {
      if (pollTimer.current) {
        window.clearTimeout(pollTimer.current);
        pollTimer.current = null;
      }
    };
  }, [run]);

  async function handleRun(mode: "offline" | "sdk") {
    setRunningMode(mode);
    setError(null);

    try {
      const response = await fetch(`/api/demo?mode=${mode}`, { method: "POST" });
      if (!response.ok) {
        throw new Error(`Demo run failed with status ${response.status}`);
      }

      const next = (await response.json()) as LiveDemoRunState;
      setRun(next);
    } catch (runError) {
      setRunningMode(null);
      setError(runError instanceof Error ? runError.message : "Demo run failed");
    }
  }

  const currentStatus = run?.status ?? "idle";
  const currentPayload = run?.payload;
  const currentStage = currentPayload?.summary.stage ?? "not started";

  return (
    <main className="min-h-screen bg-stone-950 px-6 py-10 text-stone-100">
      <section className="mx-auto mb-8 max-w-7xl rounded-[28px] border border-emerald-400/20 bg-[radial-gradient(circle_at_top_left,_rgba(16,185,129,0.18),_transparent_38%),linear-gradient(135deg,_rgba(12,10,9,0.95),_rgba(28,25,23,0.92))] p-8 shadow-[0_20px_80px_rgba(0,0,0,0.35)]">
        <div className="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
          <div className="max-w-3xl">
            <p className="mb-3 text-xs uppercase tracking-[0.35em] text-emerald-300">
              ReproLab live demo
            </p>
            <h1 className="text-4xl font-semibold tracking-tight text-white md:text-5xl">
              Run the real pipeline and follow it from the UI
            </h1>
            <p className="mt-4 text-base leading-7 text-stone-300">
              This page launches the repo pipeline in the background, polls fresh
              checkpoints while it runs, and replays the latest state through the lab
              dashboard below.
            </p>
          </div>

          <div className="flex flex-wrap gap-3">
            <button
              className="inline-flex items-center justify-center rounded-full bg-emerald-400 px-6 py-3 text-sm font-semibold text-stone-950 transition hover:bg-emerald-300 disabled:cursor-not-allowed disabled:bg-stone-700 disabled:text-stone-300"
              disabled={runningMode !== null}
              onClick={() => void handleRun("offline")}
              type="button"
            >
              {runningMode === "offline" ? "Starting offline run..." : "Run offline demo"}
            </button>
            <button
              className="inline-flex items-center justify-center rounded-full border border-emerald-300/40 bg-transparent px-6 py-3 text-sm font-semibold text-emerald-100 transition hover:border-emerald-200 hover:bg-emerald-300/10 disabled:cursor-not-allowed disabled:border-stone-700 disabled:text-stone-500"
              disabled={runningMode !== null}
              onClick={() => void handleRun("sdk")}
              type="button"
            >
              {runningMode === "sdk" ? "Starting SDK run..." : "Run SDK demo"}
            </button>
          </div>
        </div>

        {error ? (
          <div className="mt-6 rounded-2xl border border-rose-400/30 bg-rose-400/10 px-4 py-3 text-sm text-rose-100">
            {error}
          </div>
        ) : null}

        {(currentStatus === "queued" || currentStatus === "running") && run ? (
          <div className="mt-6 rounded-2xl border border-sky-400/30 bg-sky-400/10 px-4 py-3 text-sm text-sky-50">
            {run.runMode === "sdk"
              ? "The Claude SDK pipeline is running in the background. This page refreshes checkpoints every few seconds."
              : "The offline demo is running in the background. This page refreshes checkpoints every few seconds."}
          </div>
        ) : null}

        <div className="mt-8 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
            <p className="text-xs uppercase tracking-[0.28em] text-stone-400">Run status</p>
            <div
              className={`mt-3 inline-flex rounded-full border px-3 py-1 text-xs font-semibold uppercase tracking-[0.24em] ${statusTone(currentStatus)}`}
            >
              {formatStatus(currentStatus)}
            </div>
            <p className="mt-3 text-sm leading-6 text-stone-400">Stage: {currentStage}</p>
          </div>

          <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
            <p className="text-xs uppercase tracking-[0.28em] text-stone-400">Mode</p>
            <p className="mt-2 text-lg font-medium text-white">
              {currentPayload?.summary.runModeLabel ??
                (run ? (run.runMode === "sdk" ? "SDK" : "Offline") : "No run yet")}
            </p>
            <p className="mt-2 text-sm leading-6 text-stone-400">
              {run?.runMode === "sdk"
                ? "Claude SDK path using your local Claude authentication."
                : "Deterministic offline path for fast UI checks."}
            </p>
          </div>

          <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
            <p className="text-xs uppercase tracking-[0.28em] text-stone-400">Source</p>
            <p className="mt-2 text-lg font-medium text-white">
              {currentPayload?.summary.sourceLabel ?? "No run yet"}
            </p>
            <p className="mt-2 text-sm leading-6 text-stone-400">
              {currentPayload?.sourceNote ??
                "Start a run to populate the dashboard from a real pipeline execution."}
            </p>
          </div>

          <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
            <p className="text-xs uppercase tracking-[0.28em] text-stone-400">Project</p>
            <p className="mt-2 text-lg font-medium text-white">{run?.projectId ?? "pending"}</p>
            <p className="mt-2 text-sm text-stone-400">
              Updated: {run?.updatedAt ? new Date(run.updatedAt).toLocaleTimeString() : "n/a"}
            </p>
          </div>

          <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
            <p className="text-xs uppercase tracking-[0.28em] text-stone-400">Baseline reward</p>
            <p className="mt-2 text-lg font-medium text-white">
              {currentPayload?.summary.meanReward ?? "n/a"}
            </p>
            <p className="mt-2 text-sm text-stone-400">
              Improvement paths: {currentPayload?.summary.improvementCount ?? 0}
            </p>
          </div>

          <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
            <p className="text-xs uppercase tracking-[0.28em] text-stone-400">Output directory</p>
            <p className="mt-2 break-all text-sm leading-6 text-stone-200">
              {run?.outputDir ?? "runs/ui*_demo_*"}
            </p>
          </div>
        </div>
      </section>

      {run ? (
        <>
          <section className="mx-auto mb-8 max-w-7xl overflow-hidden rounded-[28px] border border-white/10 bg-stone-900/80 shadow-[0_14px_60px_rgba(0,0,0,0.35)]">
            <div className="border-b border-white/10 px-6 py-4">
              <p className="text-xs uppercase tracking-[0.3em] text-stone-400">Runner log</p>
            </div>
            <pre className="max-h-[24rem] overflow-auto px-6 py-5 text-sm leading-6 text-emerald-100">
              {run.log || "No stderr log has been captured yet."}
            </pre>
          </section>

          {adapter ? <DashboardShell adapter={adapter} /> : null}
        </>
      ) : null}
    </main>
  );
}
