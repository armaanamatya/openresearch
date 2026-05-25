"use client";

import { useState, Suspense, type ReactNode } from "react";
import { useSearchParams } from "next/navigation";

import type { AuthStatus, DemoModelChoice, DemoSandboxMode, LiveDemoRunState, RootProvider, SubagentAuth } from "@/lib/demo/demo-run-types";
import type { RecentRunSummary } from "@/lib/runs/server-list";
import type { ModelChoice } from "@/lib/models/server-fetch";
import { type DashboardLiveEvent } from "@/lib/events/dashboard-live-event";
import { UploadView } from "./upload-view";
import { LabSidebar } from "./lab-sidebar";
import { CommandPalette } from "./command-palette";
import { ShortcutOverlay } from "./shortcut-overlay";
import { useRun } from "@/hooks/use-run";
import { useCommandPalette } from "@/hooks/use-command-palette";
import { useShortcutOverlay } from "@/hooks/use-shortcut-overlay";
import { PresentationModeProvider, type PresentationMode } from "@/lib/presentation-mode";
import { readUserPrefs, writeUserPref, readProviderPrefs, writeProviderPrefs } from "@/lib/user-prefs";
import { RlmLab } from "./rlm/rlm-lab";
import { isRlmEvent } from "@/lib/events/rlm-events";
import { replayFixture } from "./rlm/replay";

import "./lab-shell.css";

type LabShellProps = {
  initialRun?: LiveDemoRunState | null;
  initialRecents?: RecentRunSummary[];
  initialRecentsError?: string | null;
  initialModels?: ModelChoice[];
  initialAuthStatus?: AuthStatus | null;
  serverDefaultSandbox?: DemoSandboxMode;
  presentationMode?: PresentationMode;
};

function WorkflowView({
  dashboardEvents,
  run
}: {
  dashboardEvents: DashboardLiveEvent[];
  run: LiveDemoRunState;
}) {
  const rlmEvents = dashboardEvents.filter(isRlmEvent);
  const paperTitle = run.sourceLabel ?? "Untitled paper";
  const paperMeta = run.sourceNote ?? "";
  const isActive = run.status === "queued" || run.status === "running";
  return (
    <RlmLab
      events={rlmEvents}
      runMeta={{
        projectId: run.projectId,
        paperTitle,
        paperMeta,
        startedAt: run.startedAt,
        // completedAt drives the elapsed-clock freeze on terminal runs.
        // Only forward when the run is in a terminal state — otherwise a
        // backend-stamped completedAt from a previous run would freeze the
        // counter mid-flight on a re-launch.
        completedAt: (run.status === "completed" || run.status === "failed" || run.status === "stopped")
          ? (run.completedAt ?? null)
          : null,
      }}
      runMode={run.runMode}
      isActive={isActive}
      runError={run.error ?? null}
      sandboxMode={run.sandboxMode ?? null}
      workerReports={run.payload?.workerReports ?? []}
    />
  );
}


// ── Dev/test-only: ?rlmFixture=1 path ────────────────────────────────────────
// When the URL has ?rlmFixture=1 the lab renders <RlmLab> against the instant-
// replayed fixture instead of any live run. This path is ONLY triggered by the
// explicit query param and never affects a real run.
const FIXTURE_RUN_META = {
  projectId: "prj_fixture",
  paperTitle: "Attention is all you need",
  paperMeta: "Vaswani et al. · fixture replay",
};

/** Inner fixture check — uses useSearchParams which requires a Suspense boundary. */
function RlmFixtureContent({ children }: { children: ReactNode }) {
  const searchParams = useSearchParams();
  const isFixtureMode = searchParams?.get("rlmFixture") === "1";
  if (isFixtureMode) {
    const events = replayFixture("instant");
    return <RlmLab events={events} runMeta={FIXTURE_RUN_META} />;
  }
  return <>{children}</>;
}

function resolveInitialModel(preferred: string, models: ModelChoice[]): string {
  if (models.length === 0) return preferred;
  const selected = models.find((candidate) => candidate.id === preferred);
  if (selected && selected.available !== false) return preferred;
  const legacyClaudeTarget = preferred === "sonnet" || preferred === "opus"
    ? models.find((candidate) => candidate.id === "claude-oauth" && candidate.available !== false)
    : undefined;
  return (
    legacyClaudeTarget ??
    models.find((candidate) => candidate.available !== false) ??
    models[0]
  ).id;
}

export function LabShell({
  initialRun = null,
  initialRecents = [],
  initialRecentsError = null,
  initialModels = [],
  initialAuthStatus = null,
  serverDefaultSandbox,
  presentationMode = "internal"
}: LabShellProps) {
  const [arxiv, setArxiv] = useState("");
  const [over, setOver] = useState(false);
  // Model state. resolveInitialModel reconciles the saved pref against the
  // backend-supplied availability list — if the persisted model is missing
  // credentials or absent from the registry, fall back to the first
  // available model (or, for legacy sonnet/opus, claude-oauth). Keeps the
  // select from rendering a stale unselectable option.
  const [model, setModel] = useState<DemoModelChoice>(() =>
    resolveInitialModel(readUserPrefs().model ?? "sonnet", initialModels)
  );

  // Provider selection state (D3 — persisted to localStorage).
  // If the persisted choice is unavailable per initialAuthStatus, fall back
  // to the server-reported default (D3 fall-back rule).
  const [rootProvider, setRootProvider] = useState<RootProvider>(() => {
    const saved = readProviderPrefs().root_provider as RootProvider | undefined;
    if (saved && initialAuthStatus) {
      const providerStatus = initialAuthStatus.providers[saved];
      if (!providerStatus?.available) {
        return initialAuthStatus.defaults.root_provider;
      }
    }
    return saved ?? initialAuthStatus?.defaults.root_provider ?? "anthropic_oauth";
  });
  const [subagentAuth, setSubagentAuth] = useState<SubagentAuth>(() => {
    const saved = readProviderPrefs().subagent_auth as SubagentAuth | undefined;
    if (saved && initialAuthStatus) {
      const available = initialAuthStatus.subagent_auth[saved];
      if (!available) {
        return initialAuthStatus.defaults.subagent_auth;
      }
    }
    return saved ?? initialAuthStatus?.defaults.subagent_auth ?? "anthropic_oauth";
  });
  const [dynamicGpu, setDynamicGpu] = useState<boolean>(() => readProviderPrefs().dynamic_gpu ?? false);
  const [forceSingleGpu, setForceSingleGpu] = useState<boolean>(() => readProviderPrefs().force_single_gpu ?? false);
  const [maxGpuUsdPerHour, setMaxGpuUsdPerHour] = useState<number>(() => readProviderPrefs().max_gpu_usd_per_hour ?? 0);
  const [vramGb, setVramGb] = useState<number>(() => readProviderPrefs().vram_gb ?? 0);
  // Lane Q — minimize-compute toggle. Persisted alongside the other run-config
  // prefs so the user's preferred reproduction style sticks across reloads.
  const [minimizeCompute, setMinimizeCompute] = useState<boolean>(() => readProviderPrefs().minimize_compute ?? false);
  // Sandbox default: user's saved pref → server-side REPROLAB_DEFAULT_SANDBOX → "docker".
  // serverDefaultSandbox is read from env at request time so Railway (runpod) overrides
  // the fallback without requiring a code change.
  const [sandbox, setSandbox] = useState<DemoSandboxMode>(
    () => readUserPrefs().sandbox ?? serverDefaultSandbox ?? "docker"
  );

  const {
    run,
    busy,
    error,
    dashboardEvents,
    runMode,
    setRunMode,
    startFixtureRun,
    startUploadedRun,
    startArxivRun
  } = useRun(initialRun, {
    rootProvider,
    subagentAuth,
    dynamicGpu: dynamicGpu || undefined,
    forceSingleGpu: forceSingleGpu || undefined,
    maxGpuUsdPerHour: maxGpuUsdPerHour > 0 ? maxGpuUsdPerHour : undefined,
    vramGb: vramGb > 0 ? vramGb : undefined,
    minimizeCompute: minimizeCompute || undefined,
  });

  const palette = useCommandPalette();
  const shortcuts = useShortcutOverlay();

  const main = (
    <main className="content">
      {/* Dev/test-only: ?rlmFixture=1 renders the fixture-driven RlmLab
          regardless of any live run. RlmFixtureContent uses useSearchParams
          which requires a Suspense boundary per Next.js App Router rules.
          When rlmFixture=1, the normal run/upload content is replaced. */}
      <Suspense fallback={null}>
        <RlmFixtureContent>
          {run ? (
            <WorkflowView
              run={run}
              dashboardEvents={dashboardEvents}
            />
          ) : (
            <UploadView
              arxiv={arxiv}
              authStatus={initialAuthStatus}
              busy={busy}
              error={error}
              model={model}
              models={initialModels}
              runMode={runMode}
              onArxivChange={setArxiv}
              onArxivSubmit={() =>
                arxiv.trim().length > 0
                  ? void startArxivRun(arxiv, model)
                  : void startFixtureRun(model)
              }
              onFileSelected={(file) => void startUploadedRun(file, model)}
              onModelChange={(value) => {
                setModel(value);
                writeUserPref("model", value);
              }}
              onRunModeChange={setRunMode}
              over={over}
              setOver={setOver}
              rootProvider={rootProvider}
              subagentAuth={subagentAuth}
              dynamicGpu={dynamicGpu}
              forceSingleGpu={forceSingleGpu}
              maxGpuUsdPerHour={maxGpuUsdPerHour}
              vramGb={vramGb}
              minimizeCompute={minimizeCompute}
              sandbox={sandbox}
              onSandboxChange={(value) => {
                setSandbox(value);
                writeUserPref("sandbox", value);
              }}
              onRootProviderChange={(value) => {
                setRootProvider(value);
                writeProviderPrefs({ ...readProviderPrefs(), root_provider: value });
              }}
              onSubagentAuthChange={(value) => {
                setSubagentAuth(value);
                writeProviderPrefs({ ...readProviderPrefs(), subagent_auth: value });
              }}
              onDynamicGpuChange={(value) => {
                setDynamicGpu(value);
                writeProviderPrefs({ ...readProviderPrefs(), dynamic_gpu: value });
              }}
              onForceSingleGpuChange={(value) => {
                setForceSingleGpu(value);
                writeProviderPrefs({ ...readProviderPrefs(), force_single_gpu: value });
              }}
              onMaxGpuUsdPerHourChange={(value) => {
                setMaxGpuUsdPerHour(value);
                writeProviderPrefs({ ...readProviderPrefs(), max_gpu_usd_per_hour: value });
              }}
              onVramGbChange={(value) => {
                setVramGb(value);
                writeProviderPrefs({ ...readProviderPrefs(), vram_gb: value });
              }}
              onMinimizeComputeChange={(value) => {
                setMinimizeCompute(value);
                writeProviderPrefs({ ...readProviderPrefs(), minimize_compute: value });
              }}
            />
          )}
        </RlmFixtureContent>
      </Suspense>
    </main>
  );

  return (
    <div className="reproLab">
      <PresentationModeProvider mode={presentationMode}>
        <div className="layout">
          <LabSidebar
            active={run ? "lab" : "upload"}
            recents={initialRecents}
            recentsError={initialRecentsError}
          />
          {main}
        </div>
        <CommandPalette
          open={palette.open}
          setOpen={palette.setOpen}
          recents={initialRecents}
          currentRun={run}
        />
        <ShortcutOverlay open={shortcuts.open} setOpen={shortcuts.setOpen} />
      </PresentationModeProvider>
    </div>
  );
}
