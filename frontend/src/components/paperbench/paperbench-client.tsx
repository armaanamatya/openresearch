"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import type {
  PaperBenchBundleListing,
  PaperBenchRunStatus,
} from "@/lib/paperbench/runner";

const POLL_INTERVAL_MS = 3000;

interface PaperBenchClientProps {
  initialBundles: PaperBenchBundleListing;
  initialRuns: PaperBenchRunStatus[];
}

interface ProviderChoice {
  value: "anthropic" | "openai";
  label: string;
}

const PROVIDERS: ProviderChoice[] = [
  { value: "anthropic", label: "Anthropic" },
  { value: "openai", label: "OpenAI" },
];

function formatPercent(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

function StatusPill({ status }: { status: PaperBenchRunStatus["status"] }) {
  const palette: Record<PaperBenchRunStatus["status"], string> = {
    pending: "bg-chip text-muted",
    running: "bg-warn-soft text-warn-ink",
    succeeded: "bg-accent-soft text-accent-ink",
    failed: "bg-err-soft text-err",
  };
  return (
    <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${palette[status]}`}>
      {status}
    </span>
  );
}

function BaselineRow({
  label,
  baseline,
  ours,
}: {
  label: string;
  baseline: { mean: number; se: number } | undefined;
  ours: number | null;
}) {
  const margin = baseline && ours !== null ? ours - baseline.mean : null;
  return (
    <div className="grid grid-cols-4 items-center gap-2 py-1 text-sm">
      <span className="font-medium text-zinc-700">{label}</span>
      <span className="text-zinc-600">
        {baseline ? `${formatPercent(baseline.mean)} ± ${formatPercent(baseline.se)}` : "—"}
      </span>
      <span className="text-zinc-900">{formatPercent(ours)}</span>
      <span
        className={
          margin === null
            ? "text-muted-2"
            : margin >= 0
              ? "text-accent-ink"
              : "text-err"
        }
      >
        {margin === null ? "—" : `${margin >= 0 ? "+" : ""}${formatPercent(margin)}`}
      </span>
    </div>
  );
}

export function PaperBenchClient({ initialBundles, initialRuns }: PaperBenchClientProps) {
  const validBundles = useMemo(
    () => initialBundles.bundles.filter((b) => b.paper_id && !b.error),
    [initialBundles]
  );

  const [paperId, setPaperId] = useState<string>(validBundles[0]?.paper_id ?? "");
  const [seedsText, setSeedsText] = useState("0");
  // Default ON — runs the real agent pipeline. Toggle off for a dry validation
  // pass that does not call any LLM.
  const [withPipeline, setWithPipeline] = useState(true);
  const [provider, setProvider] = useState<"anthropic" | "openai">("anthropic");
  const [model, setModel] = useState<string>("");
  const [maxParallel, setMaxParallel] = useState<number>(1);

  const [runs, setRuns] = useState<PaperBenchRunStatus[]>(initialRuns);
  const [activeRunId, setActiveRunId] = useState<string | null>(initialRuns[0]?.run_group_id ?? null);
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);

  const refreshRuns = useCallback(async () => {
    try {
      const response = await fetch("/api/paperbench", { cache: "no-store" });
      if (!response.ok) return;
      const json = (await response.json()) as { runs: PaperBenchRunStatus[] };
      setRuns(json.runs);
    } catch {
      // swallow polling errors
    }
  }, []);

  useEffect(() => {
    const interval = setInterval(refreshRuns, POLL_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [refreshRuns]);

  const activeRun = useMemo(
    () => runs.find((run) => run.run_group_id === activeRunId) ?? runs[0] ?? null,
    [runs, activeRunId]
  );

  const handleStart = useCallback(async () => {
    setError(null);
    setStarting(true);
    const seeds = seedsText
      .split(/[,\s]+/)
      .map((part) => part.trim())
      .filter(Boolean)
      .map((part) => Number(part))
      .filter((n) => Number.isInteger(n));
    if (seeds.length === 0) {
      setError("Provide at least one integer seed.");
      setStarting(false);
      return;
    }
    try {
      const response = await fetch("/api/paperbench", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          paperId,
          seeds,
          withPipeline,
          provider: withPipeline ? provider : undefined,
          model: withPipeline && model ? model : undefined,
          maxParallel,
        }),
      });
      if (!response.ok) {
        const errorBody = (await response.json().catch(() => ({}))) as { error?: string };
        throw new Error(errorBody.error ?? `Run failed: ${response.status}`);
      }
      const status = (await response.json()) as PaperBenchRunStatus;
      setRuns((prev) => [status, ...prev.filter((r) => r.run_group_id !== status.run_group_id)]);
      setActiveRunId(status.run_group_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Run failed");
    } finally {
      setStarting(false);
    }
  }, [paperId, seedsText, withPipeline, provider, model, maxParallel]);

  return (
    <main className="mx-auto min-h-screen max-w-6xl bg-white px-6 py-10">
      <header className="mb-8">
        <h1 className="text-3xl font-semibold tracking-tight text-zinc-900">PaperBench head-to-head</h1>
        <p className="mt-2 text-sm text-zinc-600">
          Run our agent pipeline on a vendored PaperBench paper bundle and compare the
          replication score to the published BasicAgent baselines from the PaperBench paper
          (OpenAI, April 2025).
        </p>
      </header>

      <section className="mb-10 rounded-2xl border border-zinc-200 bg-zinc-50 p-6">
        <h2 className="text-lg font-semibold text-zinc-900">Start a run</h2>
        <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          <label className="flex flex-col gap-1 text-sm">
            <span className="font-medium text-zinc-700">Paper</span>
            <select
              className="rounded-md border border-zinc-300 bg-white px-3 py-2"
              value={paperId}
              onChange={(event) => setPaperId(event.target.value)}
              disabled={validBundles.length === 0}
            >
              {validBundles.length === 0 ? (
                <option value="">no bundles vendored</option>
              ) : (
                validBundles.map((bundle) => (
                  <option key={bundle.paper_id} value={bundle.paper_id ?? ""}>
                    {(bundle.metadata?.title as string | undefined) ?? bundle.paper_id}
                  </option>
                ))
              )}
            </select>
          </label>

          <label className="flex flex-col gap-1 text-sm">
            <span className="font-medium text-zinc-700">Seeds (comma-separated)</span>
            <input
              type="text"
              className="rounded-md border border-zinc-300 bg-white px-3 py-2"
              value={seedsText}
              onChange={(event) => setSeedsText(event.target.value)}
              placeholder="0, 1, 2"
            />
          </label>

          <label className="flex flex-col gap-1 text-sm">
            <span className="font-medium text-zinc-700">Max parallel attempts</span>
            <input
              type="number"
              min={1}
              max={8}
              className="rounded-md border border-zinc-300 bg-white px-3 py-2"
              value={maxParallel}
              onChange={(event) => setMaxParallel(Number(event.target.value))}
            />
          </label>

          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={withPipeline}
              onChange={(event) => setWithPipeline(event.target.checked)}
            />
            <span className="font-medium text-zinc-700">
              Run real agent pipeline (default — needs LLM key). Uncheck for dry validation only.
            </span>
          </label>

          <label className="flex flex-col gap-1 text-sm">
            <span className="font-medium text-zinc-700">Provider</span>
            <select
              className="rounded-md border border-zinc-300 bg-white px-3 py-2"
              value={provider}
              onChange={(event) => setProvider(event.target.value as "anthropic" | "openai")}
              disabled={!withPipeline}
            >
              {PROVIDERS.map((choice) => (
                <option key={choice.value} value={choice.value}>{choice.label}</option>
              ))}
            </select>
          </label>

          <label className="flex flex-col gap-1 text-sm">
            <span className="font-medium text-zinc-700">Model override</span>
            <input
              type="text"
              className="rounded-md border border-zinc-300 bg-white px-3 py-2"
              value={model}
              onChange={(event) => setModel(event.target.value)}
              placeholder="(default)"
              disabled={!withPipeline}
            />
          </label>
        </div>

        {error ? (
          <p className="mt-4 rounded-md bg-rose-50 px-3 py-2 text-sm text-rose-700">{error}</p>
        ) : null}

        <div className="mt-6 flex items-center gap-3">
          <button
            type="button"
            onClick={handleStart}
            disabled={starting || !paperId}
            className="rounded-md bg-zinc-900 px-4 py-2 text-sm font-medium text-white hover:bg-zinc-800 disabled:bg-zinc-400"
          >
            {starting ? "Starting…" : withPipeline ? "Start pipeline run" : "Start dry validation"}
          </button>
          <span className="text-xs text-zinc-500">
            {withPipeline
              ? "Pipeline mode runs the real agent stack across the chosen seeds. Keep an LLM key in .env."
              : "Dry mode loads the bundle, computes the rubric ceiling, and validates a placeholder submission. No LLM key required."}
          </span>
        </div>
      </section>

      <section className="mb-10">
        <h2 className="text-lg font-semibold text-zinc-900">Runs</h2>
        {runs.length === 0 ? (
          <p className="mt-2 text-sm text-zinc-500">No runs yet.</p>
        ) : (
          <ul className="mt-3 divide-y divide-zinc-200 rounded-xl border border-zinc-200">
            {runs.map((run) => (
              <li
                key={run.run_group_id}
                className={`flex cursor-pointer items-center justify-between px-4 py-3 text-sm hover:bg-zinc-50 ${
                  activeRun?.run_group_id === run.run_group_id ? "bg-zinc-50" : ""
                }`}
                onClick={() => setActiveRunId(run.run_group_id)}
              >
                <span className="font-mono text-xs text-zinc-700">{run.run_group_id}</span>
                <span className="text-zinc-600">{run.paper_id}</span>
                <span className="text-zinc-500">{run.mode}</span>
                <span className="text-zinc-500">{run.n_attempts} attempt{run.n_attempts === 1 ? "" : "s"}</span>
                <StatusPill status={run.status} />
              </li>
            ))}
          </ul>
        )}
      </section>

      {activeRun ? (
        <section className="rounded-2xl border border-zinc-200 bg-white p-6">
          <header className="flex items-center justify-between">
            <div>
              <h2 className="text-lg font-semibold text-zinc-900">
                {activeRun.paper_id}
              </h2>
              <p className="text-xs text-zinc-500 font-mono">{activeRun.run_group_id}</p>
            </div>
            <StatusPill status={activeRun.status} />
          </header>

          <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-2">
            <div>
              <h3 className="text-sm font-semibold text-zinc-700">Score vs PaperBench baselines</h3>
              <div className="mt-2">
                <div className="grid grid-cols-4 gap-2 border-b border-zinc-200 py-1 text-xs uppercase tracking-wide text-zinc-500">
                  <span>Model</span>
                  <span>Their score (mean ± SE)</span>
                  <span>Ours</span>
                  <span>Margin</span>
                </div>
                {Object.entries(activeRun.published_baselines).map(([model, baseline]) => (
                  <BaselineRow
                    key={model}
                    label={model}
                    baseline={baseline}
                    ours={activeRun.mean_score}
                  />
                ))}
              </div>
              <p className="mt-3 text-xs text-zinc-500">
                Code-only ceiling for this rubric: <span className="font-medium">{formatPercent(activeRun.code_development_ceiling)}</span>.
                {activeRun.standard_error !== null
                  ? ` Our SE across ${activeRun.n_attempts} seed${activeRun.n_attempts === 1 ? "" : "s"}: ${formatPercent(activeRun.standard_error, 2)}.`
                  : ""}
              </p>
            </div>

            <div>
              <h3 className="text-sm font-semibold text-zinc-700">Rubric breakdown</h3>
              <ul className="mt-2 space-y-1 text-sm">
                {Object.entries(activeRun.rubric_summary.task_category_weights).map(([category, weight]) => (
                  <li key={category} className="flex items-center justify-between">
                    <span className="text-zinc-700">{category}</span>
                    <span className="text-zinc-500">
                      {formatPercent(weight.weight)} • {weight.leaf_count} leaves
                    </span>
                  </li>
                ))}
              </ul>
              <p className="mt-3 text-xs text-zinc-500">
                {activeRun.rubric_summary.node_count} nodes • {activeRun.rubric_summary.leaf_count} leaves • depth {activeRun.rubric_summary.max_depth}
              </p>
            </div>
          </div>

          {activeRun.attempts.length > 0 ? (
            <div className="mt-6">
              <h3 className="text-sm font-semibold text-zinc-700">Attempts</h3>
              <table className="mt-2 w-full text-sm">
                <thead>
                  <tr className="border-b border-zinc-200 text-left text-xs uppercase tracking-wide text-zinc-500">
                    <th className="py-1">Attempt</th>
                    <th className="py-1">Seed</th>
                    <th className="py-1">Status</th>
                    <th className="py-1">Score</th>
                    <th className="py-1">Submission OK</th>
                  </tr>
                </thead>
                <tbody>
                  {activeRun.attempts.map((attempt) => (
                    <tr key={attempt.attempt_id} className="border-b border-zinc-100">
                      <td className="py-1 font-mono text-xs text-zinc-700">{attempt.attempt_id}</td>
                      <td className="py-1 text-zinc-600">{attempt.seed ?? "—"}</td>
                      <td className="py-1 text-zinc-600">{attempt.status}</td>
                      <td className="py-1 text-zinc-900">{formatPercent(attempt.score ?? null)}</td>
                      <td className="py-1 text-zinc-600">
                        {attempt.submission_validation
                          ? attempt.submission_validation.ok
                            ? "✓"
                            : `✗ (${attempt.submission_validation.errors.length} errors)`
                          : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : null}

          {activeRun.error ? (
            <p className="mt-4 rounded-md bg-rose-50 px-3 py-2 text-sm text-rose-700">{activeRun.error}</p>
          ) : null}
        </section>
      ) : null}
    </main>
  );
}
