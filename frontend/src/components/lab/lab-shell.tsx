"use client";

import { useState, Suspense, type ReactNode } from "react";

import type { DemoModelChoice, LiveDemoRunState } from "@/lib/demo/demo-run-types";
import type { RecentRunSummary } from "@/lib/runs/server-list";
import type { ModelChoice } from "@/lib/models/server-fetch";
import { useSearchParams } from "next/navigation";
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

// Dev/test-only fixture metadata for the ?rlmFixture=1 replay path. A real run
// gets its title/meta/projectId from the run object (sourceLabel / sourceNote /
// projectId); this constant is the fixture's own stand-in for that.
const FIXTURE_RUN_META = {
  projectId: "prj_fixture",
  paperTitle: "Attention is all you need",
  paperMeta: "Vaswani et al. · fixture replay",
};

// The lab is RLM-only. A live run renders <RlmLab>; an older run that predates
// the RLM orchestrator (runMode !== "rlm", e.g. opened from Recent) cannot be
// rendered by the RLM lab — show an honest notice rather than an empty shell.
function RunView({
  run,
  dashboardEvents,
}: {
  run: LiveDemoRunState;
  dashboardEvents: ReturnType<typeof useRun>["dashboardEvents"];
}) {
  if (run.runMode !== "rlm") {
    return (
      <div className="card" style={{ padding: 24 }} data-testid="non-rlm-notice">
        <div className="eyebrow">Run</div>
        <p style={{ marginTop: 8 }}>
          This run was produced by the retired 14-stage pipeline. The RLM lab
          renders <code>rlm</code>-mode runs only — start a new run to watch it live.
        </p>
      </div>
    );
  }
  const rlmEvents = dashboardEvents.filter(isRlmEvent);
  return (
    <RlmLab
      events={rlmEvents}
      runMeta={{
        projectId: run.projectId,
        paperTitle: run.sourceLabel ?? "Untitled paper",
        paperMeta: run.sourceNote ?? "",
      }}
    />
  );
}

// Dev/test-only: ?rlmFixture=1 renders the fixture-driven RlmLab regardless of
// any live run. useSearchParams requires a Suspense boundary (App Router rule).
function RlmFixtureContent({ children }: { children: ReactNode }) {
  const searchParams = useSearchParams();
  if (searchParams?.get("rlmFixture") === "1") {
    return <RlmLab events={replayFixture("instant")} runMeta={FIXTURE_RUN_META} />;
  }
  return <>{children}</>;
}

export function LabShell({
  initialRun = null,
  initialRecents = [],
  initialModels = [],
  presentationMode = "internal",
}: LabShellProps) {
  const [arxiv, setArxiv] = useState("");
  const [over, setOver] = useState(false);
  const [model, setModel] = useState<DemoModelChoice>(() => readUserPrefs().model ?? "sonnet");
  const {
    run,
    busy,
    error,
    dashboardEvents,
    startFixtureRun,
    startUploadedRun,
    startArxivRun,
    resetToUpload: resetRun,
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
      <Suspense fallback={null}>
        <RlmFixtureContent>
          {run ? (
            <RunView run={run} dashboardEvents={dashboardEvents} />
          ) : (
            <UploadView
              arxiv={arxiv}
              busy={busy}
              error={error}
              model={model}
              models={initialModels}
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
