"use client";

import { useState, Suspense, type ReactNode } from "react";
import { useSearchParams } from "next/navigation";

import type { DemoModelChoice, LiveDemoRunState } from "@/lib/demo/demo-run-types";
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
import { readUserPrefs, writeUserPref } from "@/lib/user-prefs";
import { RlmLab } from "./rlm/rlm-lab";
import { isRlmEvent } from "@/lib/events/rlm-events";
import { replayFixture } from "./rlm/replay";

import "./lab-shell.css";

type LabShellProps = {
  initialRun?: LiveDemoRunState | null;
  initialRecents?: RecentRunSummary[];
  initialModels?: ModelChoice[];
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
      runMeta={{ projectId: run.projectId, paperTitle, paperMeta, startedAt: run.startedAt }}
      runMode={run.runMode}
      isActive={isActive}
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

export function LabShell({
  initialRun = null,
  initialRecents = [],
  initialModels = [],
  presentationMode = "internal"
}: LabShellProps) {
  const [arxiv, setArxiv] = useState("");
  const [over, setOver] = useState(false);
  const [model, setModel] = useState<DemoModelChoice>(() => readUserPrefs().model ?? "sonnet");
  const {
    run,
    busy,
    error,
    dashboardEvents,
    runMode,
    setRunMode,
    startFixtureRun,
    startUploadedRun,
    startArxivRun,
    resetToUpload: resetRun
  } = useRun(initialRun);

  const resetToUpload = () => {
    setArxiv("");
    setOver(false);
    resetRun();
  };

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
          <LabSidebar active="lab" onBrandClick={resetToUpload} recents={initialRecents} />
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
