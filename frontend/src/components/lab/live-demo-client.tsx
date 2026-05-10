"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { ChevronDown } from "lucide-react";

import { DashboardShell } from "@/features/dashboard/dashboard-shell";
import { ProgressStrip } from "@/components/lab/progress-strip";
import { TimelinePanel } from "@/components/lab/timeline-panel";
import type {
  DemoExecutionMode,
  DemoGpuMode,
  DemoProvider,
  DemoSandboxMode,
  LiveDemoRunState
} from "@/lib/demo/demo-run-types";
import { createMockEventAdapter } from "@/lib/events/mock-event-adapter";

interface LiveDemoClientProps {
  initialRun: LiveDemoRunState | null;
}

const POLL_INTERVAL_MS = 3000;
const PROVIDER_OPTIONS: Array<{ value: DemoProvider; label: string; helper: string }> = [
  {
    value: "anthropic",
    label: "Anthropic",
    helper: "Claude Agent SDK"
  },
  {
    value: "openai",
    label: "OpenAI",
    helper: "OpenAI Agents SDK"
  }
];
type ReviewProviderOption = DemoProvider | "same";
const REVIEW_PROVIDER_OPTIONS: Array<{
  value: ReviewProviderOption;
  label: string;
  helper: string;
}> = [
  {
    value: "same",
    label: "Same",
    helper: "Reuse builder SDK"
  },
  {
    value: "openai",
    label: "OpenAI",
    helper: "Cross-check with Codex"
  },
  {
    value: "anthropic",
    label: "Anthropic",
    helper: "Cross-check with Claude"
  }
];
const EXECUTION_OPTIONS: Array<{
  value: DemoExecutionMode;
  label: string;
  helper: string;
}> = [
  {
    value: "efficient",
    label: "Efficient",
    helper: "Bounded budgets"
  },
  {
    value: "max",
    label: "Max",
    helper: "Higher confidence"
  }
];
const GPU_OPTIONS: Array<{ value: DemoGpuMode; label: string; helper: string }> = [
  {
    value: "auto",
    label: "Auto GPU",
    helper: "Safe default"
  },
  {
    value: "off",
    label: "CPU safe",
    helper: "Hide CUDA"
  },
  {
    value: "prefer",
    label: "Prefer GPU",
    helper: "Use if present"
  },
  {
    value: "max",
    label: "Max GPU",
    helper: "Higher budget"
  }
];
const SANDBOX_OPTIONS: Array<{ value: DemoSandboxMode; label: string; helper: string }> = [
  {
    value: "auto",
    label: "Auto Docker",
    helper: "Docker first"
  },
  {
    value: "docker",
    label: "Docker",
    helper: "Container sandbox"
  },
  {
    value: "runpod",
    label: "Runpod GPU",
    helper: "Remote GPU Pod"
  },
  {
    value: "local",
    label: "Local",
    helper: "Explicit host run"
  }
];

function formatStatus(status: LiveDemoRunState["status"] | "idle") {
  switch (status) {
    case "queued":
      return "Queued";
    case "running":
      return "Running";
    case "stopped":
      return "Stopped";
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
    case "stopped":
      return "border-stone-300/30 bg-stone-300/10 text-stone-100";
    case "completed":
      return "border-emerald-300/30 bg-emerald-300/10 text-emerald-100";
    case "failed":
      return "border-rose-300/30 bg-rose-300/10 text-rose-100";
    default:
      return "border-white/10 bg-white/5 text-stone-200";
  }
}

async function responseError(response: Response, fallback: string): Promise<Error> {
  const payload = (await response.json().catch(() => null)) as {
    error?: string;
  } | null;
  return new Error(payload?.error ?? fallback);
}

function SelectControl({
  disabled,
  id,
  label,
  onChange,
  options,
  value
}: {
  disabled: boolean;
  id: string;
  label: string;
  onChange: (value: string) => void;
  options: Array<{ value: string; label: string; helper: string }>;
  value: string;
}) {
  return (
    <label className="flex flex-col gap-2" htmlFor={id}>
      <span className="text-xs font-semibold uppercase tracking-[0.18em] text-stone-400">
        {label}
      </span>
      <span className="relative">
        <select
          className="w-full appearance-none rounded-xl border border-white/10 bg-stone-950 px-3 py-3 pr-9 text-sm font-semibold text-white outline-none transition focus:border-emerald-300/70 disabled:cursor-not-allowed disabled:text-stone-500"
          disabled={disabled}
          id={id}
          onChange={(event) => onChange(event.target.value)}
          value={value}
        >
          {options.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label} - {option.helper}
            </option>
          ))}
        </select>
        <ChevronDown
          aria-hidden="true"
          className="pointer-events-none absolute right-3 top-1/2 h-4 w-4 -translate-y-1/2 text-stone-400"
        />
      </span>
    </label>
  );
}

export function LiveDemoClient({ initialRun }: LiveDemoClientProps) {
  const [run, setRun] = useState(initialRun);
  const [sdkProvider, setSdkProvider] = useState<DemoProvider>(
    initialRun?.llmProvider ?? "anthropic"
  );
  const [reviewProvider, setReviewProvider] = useState<ReviewProviderOption>(
    initialRun?.verificationProvider ?? "same"
  );
  const [executionMode, setExecutionMode] = useState<DemoExecutionMode>(
    initialRun?.executionMode ?? "efficient"
  );
  const [sandboxMode, setSandboxMode] = useState<DemoSandboxMode>(
    initialRun?.sandboxMode ?? "auto"
  );
  const [gpuMode, setGpuMode] = useState<DemoGpuMode>(initialRun?.gpuMode ?? "auto");
  const [selectedPaper, setSelectedPaper] = useState<File | null>(null);
  const [runningMode, setRunningMode] = useState<"offline" | "sdk" | null>(
    initialRun && (initialRun.status === "queued" || initialRun.status === "running")
      ? initialRun.runMode
      : null
  );
  const [error, setError] = useState<string | null>(null);
  const pollTimer = useRef<number | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

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
    if (run?.llmProvider) {
      setSdkProvider(run.llmProvider);
    }
    if (run?.executionMode) {
      setExecutionMode(run.executionMode);
    }
    if (run?.sandboxMode) {
      setSandboxMode(run.sandboxMode);
    }
    if (run) {
      setReviewProvider(run.verificationProvider ?? "same");
    }
    if (run?.gpuMode) {
      setGpuMode(run.gpuMode);
    }
  }, [
    run?.executionMode,
    run?.gpuMode,
    run?.llmProvider,
    run?.sandboxMode,
    run?.verificationProvider
  ]);

  useEffect(() => {
    eventSourceRef.current?.close();
    eventSourceRef.current = null;

    if (run?.status !== "queued" && run?.status !== "running") {
      setRunningMode(null);
      if (pollTimer.current) {
        window.clearTimeout(pollTimer.current);
        pollTimer.current = null;
      }
      return;
    }

    const projectId = run.projectId;
    if (typeof EventSource !== "undefined") {
      const source = new EventSource(
        `/api/demo/events?projectId=${encodeURIComponent(projectId)}`
      );
      eventSourceRef.current = source;
      source.addEventListener("run_state", (event) => {
        try {
          const next = JSON.parse((event as MessageEvent).data) as LiveDemoRunState;
          setRun(next);
          if (next.status === "failed") {
            setError(next.error ?? "Demo run failed");
          }
        } catch {
          setError("Unable to parse live run update");
        }
      });
      source.addEventListener("agent_log", (event) => {
        try {
          const update = JSON.parse((event as MessageEvent).data) as {
            log?: string;
            text?: string;
          };
          setRun((current) =>
            current && current.projectId === projectId
              ? {
                  ...current,
                  log:
                    typeof update.log === "string"
                      ? update.log
                      : `${current.log}${update.text ?? ""}`
                }
              : current
          );
        } catch {
          setError("Unable to parse live log update");
        }
      });
      source.onerror = () => {
        source.close();
        if (eventSourceRef.current === source) {
          eventSourceRef.current = null;
        }
      };
      return () => {
        source.close();
        if (eventSourceRef.current === source) {
          eventSourceRef.current = null;
        }
      };
    }

    const providerParam =
      run.runMode === "sdk" && run.llmProvider ? `&provider=${run.llmProvider}` : "";
    const verificationParam =
      run.runMode === "sdk" && run.verificationProvider
        ? `&verificationProvider=${run.verificationProvider}`
        : "";
    const executionParam = run.executionMode ? `&executionMode=${run.executionMode}` : "";
    const sandboxParam = run.sandboxMode ? `&sandbox=${run.sandboxMode}` : "";
    const gpuParam = run.gpuMode ? `&gpuMode=${run.gpuMode}` : "";
    pollTimer.current = window.setTimeout(async () => {
      try {
        const response = await fetch(
          `/api/demo?projectId=${projectId}&mode=${run.runMode}${providerParam}${verificationParam}${executionParam}${sandboxParam}${gpuParam}`,
          { cache: "no-store" }
        );
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
  }, [
    run?.executionMode,
    run?.gpuMode,
    run?.llmProvider,
    run?.projectId,
    run?.runMode,
    run?.sandboxMode,
    run?.status,
    run?.verificationProvider
  ]);

  async function handleRun(mode: "offline" | "sdk") {
    setRunningMode(mode);
    setError(null);

    try {
      const providerParam = mode === "sdk" ? `&provider=${sdkProvider}` : "";
      const verificationParam =
        mode === "sdk" && reviewProvider !== "same"
          ? `&verificationProvider=${reviewProvider}`
          : "";
      const response = await fetch(
        `/api/demo?mode=${mode}${providerParam}${verificationParam}&executionMode=${executionMode}&sandbox=${sandboxMode}&gpuMode=${gpuMode}`,
        {
          method: "POST"
        }
      );
      if (!response.ok) {
        throw await responseError(response, `Demo run failed with status ${response.status}`);
      }

      const next = (await response.json()) as LiveDemoRunState;
      setRun(next);
    } catch (runError) {
      setRunningMode(null);
      setError(runError instanceof Error ? runError.message : "Demo run failed");
    }
  }

  async function handleUploadedRun(mode: "offline" | "sdk") {
    if (!selectedPaper) {
      setError("Choose a PDF before starting an uploaded-paper run.");
      return;
    }

    setRunningMode(mode);
    setError(null);

    try {
      const formData = new FormData();
      formData.set("mode", mode);
      formData.set("paper", selectedPaper);
      formData.set("executionMode", executionMode);
      formData.set("sandbox", sandboxMode);
      formData.set("gpuMode", gpuMode);
      if (mode === "sdk") {
        formData.set("provider", sdkProvider);
        formData.set("verificationProvider", reviewProvider);
      }

      const response = await fetch("/api/demo", {
        method: "POST",
        body: formData
      });
      if (!response.ok) {
        throw await responseError(response, `Demo run failed with status ${response.status}`);
      }

      const next = (await response.json()) as LiveDemoRunState;
      setRun(next);
    } catch (runError) {
      setRunningMode(null);
      setError(runError instanceof Error ? runError.message : "Demo run failed");
    }
  }

  async function handleStop() {
    if (!run || (run.status !== "queued" && run.status !== "running")) {
      return;
    }

    setError(null);

    try {
      const providerParam =
        run.runMode === "sdk" && run.llmProvider ? `&provider=${run.llmProvider}` : "";
      const verificationParam =
        run.runMode === "sdk" && run.verificationProvider
          ? `&verificationProvider=${run.verificationProvider}`
          : "";
      const executionParam = run.executionMode ? `&executionMode=${run.executionMode}` : "";
      const sandboxParam = run.sandboxMode ? `&sandbox=${run.sandboxMode}` : "";
      const gpuParam = run.gpuMode ? `&gpuMode=${run.gpuMode}` : "";
      const response = await fetch(
        `/api/demo?projectId=${run.projectId}&mode=${run.runMode}${providerParam}${verificationParam}${executionParam}${sandboxParam}${gpuParam}`,
        { method: "DELETE" }
      );
      if (!response.ok) {
        throw new Error(`Stop request failed with status ${response.status}`);
      }

      const next = (await response.json()) as LiveDemoRunState | null;
      setRun(next);
      setRunningMode(null);
    } catch (stopError) {
      setError(stopError instanceof Error ? stopError.message : "Unable to stop demo run");
    }
  }

  const currentStatus = run?.status ?? "idle";
  const currentPayload = run?.payload;
  const currentStage = currentPayload?.summary.stage ?? "not started";
  const activeProvider = run?.llmProvider ?? sdkProvider;
  const activeReviewProvider = run?.verificationProvider ?? reviewProvider;
  const activeProviderLabel =
    PROVIDER_OPTIONS.find((option) => option.value === activeProvider)?.label ?? "Anthropic";
  const activeReviewProviderLabel =
    REVIEW_PROVIDER_OPTIONS.find((option) => option.value === activeReviewProvider)?.label ??
    "Same";
  const activeExecutionMode = run?.executionMode ?? executionMode;
  const activeSandboxMode = run?.sandboxMode ?? sandboxMode;
  const activeGpuMode = run?.gpuMode ?? gpuMode;
  const activeExecutionLabel =
    EXECUTION_OPTIONS.find((option) => option.value === activeExecutionMode)?.label ??
    "Efficient";
  const activeSandboxLabel =
    SANDBOX_OPTIONS.find((option) => option.value === activeSandboxMode)?.label ??
    "Auto Docker";
  const activeGpuLabel =
    GPU_OPTIONS.find((option) => option.value === activeGpuMode)?.label ?? "Auto GPU";

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
              This page launches the backend pipeline, streams live agent events
              while it runs, and replays the latest state through the lab
              dashboard below.
            </p>
          </div>

          <div className="flex w-full flex-col gap-4 rounded-2xl border border-white/10 bg-white/5 p-4 sm:w-auto sm:min-w-[24rem]">
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
              <SelectControl
                disabled={runningMode !== null}
                id="sdk-provider"
                label="SDK provider"
                onChange={(value) => setSdkProvider(value as DemoProvider)}
                options={PROVIDER_OPTIONS}
                value={sdkProvider}
              />
              <SelectControl
                disabled={runningMode !== null}
                id="review-provider"
                label="Review SDK"
                onChange={(value) => setReviewProvider(value as ReviewProviderOption)}
                options={REVIEW_PROVIDER_OPTIONS}
                value={reviewProvider}
              />
              <SelectControl
                disabled={runningMode !== null}
                id="execution-mode"
                label="Profile"
                onChange={(value) => setExecutionMode(value as DemoExecutionMode)}
                options={EXECUTION_OPTIONS}
                value={executionMode}
              />
              <SelectControl
                disabled={runningMode !== null}
                id="sandbox-mode"
                label="Sandbox"
                onChange={(value) => setSandboxMode(value as DemoSandboxMode)}
                options={SANDBOX_OPTIONS}
                value={sandboxMode}
              />
              <SelectControl
                disabled={runningMode !== null}
                id="gpu-mode"
                label="Compute"
                onChange={(value) => setGpuMode(value as DemoGpuMode)}
                options={GPU_OPTIONS}
                value={gpuMode}
              />
            </div>
            <div className="flex flex-wrap gap-3">
              <button
                className="inline-flex flex-1 items-center justify-center rounded-full bg-emerald-400 px-5 py-3 text-sm font-semibold text-stone-950 transition hover:bg-emerald-300 disabled:cursor-not-allowed disabled:bg-stone-700 disabled:text-stone-300"
                disabled={runningMode !== null}
                onClick={() => void handleRun("offline")}
                type="button"
              >
                {runningMode === "offline" ? "Starting offline run..." : "Run offline demo"}
              </button>
              <button
                className="inline-flex flex-1 items-center justify-center rounded-full border border-emerald-300/40 bg-transparent px-5 py-3 text-sm font-semibold text-emerald-100 transition hover:border-emerald-200 hover:bg-emerald-300/10 disabled:cursor-not-allowed disabled:border-stone-700 disabled:text-stone-500"
                disabled={runningMode !== null}
                onClick={() => void handleRun("sdk")}
                type="button"
              >
                {runningMode === "sdk" ? "Starting SDK run..." : "Run SDK"}
              </button>
              <button
                className="inline-flex flex-1 items-center justify-center rounded-full border border-rose-300/35 bg-rose-400/10 px-5 py-3 text-sm font-semibold text-rose-100 transition hover:border-rose-200 hover:bg-rose-400/20 disabled:cursor-not-allowed disabled:border-stone-700 disabled:text-stone-500"
                disabled={!run || (run.status !== "queued" && run.status !== "running")}
                onClick={() => void handleStop()}
                type="button"
              >
                Stop current run
              </button>
            </div>
          </div>
        </div>

        <div className="mt-6 rounded-[24px] border border-white/10 bg-white/5 p-5">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
            <div className="max-w-2xl">
              <p className="text-xs uppercase tracking-[0.28em] text-stone-400">
                Upload paper
              </p>
              <h2 className="mt-2 text-xl font-semibold text-white">
                Start a lab run from your own PDF
              </h2>
              <p className="mt-2 text-sm leading-6 text-stone-300">
                Upload a paper here to run the same lab flow from a real PDF source
                instead of the built-in PPO fixture.
              </p>
            </div>

            <div className="flex flex-wrap gap-3">
              <button
                className="inline-flex items-center justify-center rounded-full bg-emerald-400 px-5 py-2.5 text-sm font-semibold text-stone-950 transition hover:bg-emerald-300 disabled:cursor-not-allowed disabled:bg-stone-700 disabled:text-stone-300"
                disabled={runningMode !== null || !selectedPaper}
                onClick={() => void handleUploadedRun("offline")}
                type="button"
              >
                {runningMode === "offline" && selectedPaper
                  ? "Starting uploaded offline run..."
                  : "Run uploaded paper (offline)"}
              </button>
              <button
                className="inline-flex items-center justify-center rounded-full border border-emerald-300/40 bg-transparent px-5 py-2.5 text-sm font-semibold text-emerald-100 transition hover:border-emerald-200 hover:bg-emerald-300/10 disabled:cursor-not-allowed disabled:border-stone-700 disabled:text-stone-500"
                disabled={runningMode !== null || !selectedPaper}
                onClick={() => void handleUploadedRun("sdk")}
                type="button"
              >
                {runningMode === "sdk" && selectedPaper
                  ? "Starting uploaded SDK run..."
                  : "Run uploaded paper (SDK)"}
              </button>
            </div>
          </div>

          <div className="mt-4 flex flex-col gap-3 rounded-2xl border border-dashed border-emerald-300/25 bg-stone-950/40 p-4 md:flex-row md:items-center md:justify-between">
            <div>
              <label
                className="text-sm font-medium text-white"
                htmlFor="lab-paper-upload"
              >
                Upload paper PDF
              </label>
              <p className="mt-1 text-sm text-stone-400">
                One PDF only. The lab will route it through the repo&apos;s paper
                ingestion pipeline.
              </p>
            </div>

            <div className="flex flex-wrap items-center gap-3">
              <input
                accept="application/pdf,.pdf"
                className="block text-sm text-stone-200 file:mr-4 file:rounded-full file:border-0 file:bg-emerald-400 file:px-4 file:py-2 file:font-semibold file:text-stone-950 hover:file:bg-emerald-300"
                id="lab-paper-upload"
                onChange={(event) =>
                  setSelectedPaper(event.target.files?.[0] ?? null)
                }
                type="file"
              />
              {selectedPaper ? (
                <>
                  <span className="rounded-full border border-emerald-300/20 bg-emerald-300/10 px-3 py-1 text-xs font-medium text-emerald-100">
                    {selectedPaper.name}
                  </span>
                  <button
                    className="text-sm text-stone-300 underline decoration-stone-500 underline-offset-4 hover:text-white"
                    onClick={() => setSelectedPaper(null)}
                    type="button"
                  >
                    Clear
                  </button>
                </>
              ) : (
                <span className="text-sm text-stone-400">No PDF selected yet.</span>
              )}
            </div>
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
              ? `The ${activeProviderLabel} SDK pipeline is running with ${activeReviewProviderLabel.toLowerCase()} review, ${activeExecutionLabel.toLowerCase()} profile, ${activeSandboxLabel.toLowerCase()} execution, and ${activeGpuLabel.toLowerCase()} compute. This page streams live backend events as they arrive.`
              : `The offline demo is running with ${activeExecutionLabel.toLowerCase()} profile, ${activeSandboxLabel.toLowerCase()} execution, and ${activeGpuLabel.toLowerCase()} compute. This page streams live backend events as they arrive.`}
          </div>
        ) : null}

        {currentStatus === "stopped" && run ? (
          <div className="mt-6 rounded-2xl border border-stone-300/20 bg-stone-300/10 px-4 py-3 text-sm text-stone-100">
            This run was stopped from the lab page. You can start a fresh SDK or offline run whenever you want.
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
                (run ? (run.runMode === "sdk" ? `SDK: ${activeProviderLabel}` : "Offline") : "No run yet")}
            </p>
            <p className="mt-2 text-sm leading-6 text-stone-400">
              {run?.runMode === "sdk"
                ? `${activeProviderLabel} agents, ${activeReviewProviderLabel.toLowerCase()} review, ${activeExecutionLabel.toLowerCase()} profile, ${activeSandboxLabel.toLowerCase()} execution, ${activeGpuLabel.toLowerCase()} compute.`
                : `Deterministic offline path, ${activeExecutionLabel.toLowerCase()} profile, ${activeSandboxLabel.toLowerCase()} execution, ${activeGpuLabel.toLowerCase()} compute.`}
            </p>
          </div>

          <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
            <p className="text-xs uppercase tracking-[0.28em] text-stone-400">Source</p>
            <p className="mt-2 text-lg font-medium text-white">
              {currentPayload?.summary.sourceLabel ?? run?.sourceLabel ?? "No run yet"}
            </p>
            <p className="mt-2 text-sm leading-6 text-stone-400">
              {currentPayload?.sourceNote ??
                run?.sourceNote ??
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
          <div className="mx-auto mb-6 max-w-7xl">
            <ProgressStrip run={run} />
          </div>
          <section className="mx-auto mb-8 max-w-7xl overflow-hidden rounded-[28px] border border-white/10 bg-stone-900/80 shadow-[0_14px_60px_rgba(0,0,0,0.35)]">
            <div className="flex items-center justify-between border-b border-white/10 px-6 py-4">
              <p className="text-xs uppercase tracking-[0.3em] text-stone-400">Runner log</p>
              <div className="flex items-center gap-2">
                <CopyLogButton log={run.log ?? ""} />
                <CopyDebugBundleButton projectId={run.projectId} />
              </div>
            </div>
            <pre className="max-h-[24rem] overflow-auto px-6 py-5 text-sm leading-6 text-emerald-100">
              {run.log || "No stderr log has been captured yet."}
            </pre>
          </section>

          <div className="mx-auto mb-8 max-w-7xl">
            <TimelinePanel telemetry={run.telemetry} />
          </div>

          {adapter ? <DashboardShell adapter={adapter} /> : null}
        </>
      ) : null}
    </main>
  );
}

function CopyDebugBundleButton({ projectId }: { projectId: string }) {
  const [state, setState] = useState<"idle" | "fetching" | "copied" | "error">(
    "idle"
  );

  const handleClick = async () => {
    setState("fetching");
    try {
      const res = await fetch(
        `/api/lab/debug-bundle?projectId=${encodeURIComponent(projectId)}`,
        { cache: "no-store" }
      );
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      const text = JSON.stringify(json, null, 2);
      await copyText(text);
      setState("copied");
      setTimeout(() => setState("idle"), 1800);
    } catch {
      setState("error");
      setTimeout(() => setState("idle"), 1800);
    }
  };

  const label =
    state === "fetching"
      ? "Bundling…"
      : state === "copied"
        ? "Copied"
        : state === "error"
          ? "Bundle failed"
          : "Copy debug bundle";

  return (
    <button
      type="button"
      onClick={handleClick}
      disabled={state === "fetching"}
      className={`rounded-md border px-3 py-1 text-xs font-medium transition-colors ${
        state === "copied"
          ? "border-emerald-400/40 bg-emerald-500/10 text-emerald-200"
          : state === "error"
            ? "border-rose-400/40 bg-rose-500/10 text-rose-200"
            : "border-white/15 bg-white/5 text-stone-200 hover:bg-white/10"
      }`}
      aria-live="polite"
      title="Copy a structured JSON bundle (status, last log lines, telemetry tail, pipeline state preview) to share with Claude Code or paste into a bug report."
    >
      {label}
    </button>
  );
}

async function copyText(text: string): Promise<void> {
  if (typeof navigator !== "undefined" && navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }
  // Fallback for environments without the async clipboard API.
  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  document.body.appendChild(textarea);
  textarea.select();
  const ok = document.execCommand?.("copy") ?? false;
  document.body.removeChild(textarea);
  if (!ok) throw new Error("execCommand copy failed");
}

function CopyLogButton({ log }: { log: string }) {
  const [state, setState] = useState<"idle" | "copied" | "error">("idle");
  const disabled = !log;

  const handleCopy = async () => {
    if (!log) return;
    try {
      await copyText(log);
      setState("copied");
      setTimeout(() => setState("idle"), 1800);
    } catch {
      setState("error");
      setTimeout(() => setState("idle"), 1800);
    }
  };

  const label =
    state === "copied" ? "Copied" : state === "error" ? "Copy failed" : "Copy log";

  return (
    <button
      type="button"
      onClick={handleCopy}
      disabled={disabled}
      className={`rounded-md border px-3 py-1 text-xs font-medium transition-colors ${
        disabled
          ? "cursor-not-allowed border-white/10 bg-white/5 text-stone-500"
          : state === "copied"
            ? "border-emerald-400/40 bg-emerald-500/10 text-emerald-200"
            : state === "error"
              ? "border-rose-400/40 bg-rose-500/10 text-rose-200"
              : "border-white/15 bg-white/5 text-stone-200 hover:bg-white/10"
      }`}
      aria-live="polite"
    >
      {label}
    </button>
  );
}
