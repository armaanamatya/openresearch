"use client";

import { useMemo, useState } from "react";
import { useRouter } from "next/navigation";

import type { DemoModelChoice, LiveDemoRunState } from "@/lib/demo/demo-run-types";
import type { RecentRunSummary } from "@/lib/runs/server-list";
import type { LaidOutNode } from "@/lib/pipeline/layout";
import type { PipelineTopology } from "@/lib/pipeline/topology";
import type { ModelChoice } from "@/lib/models/server-fetch";
import { AgentTimelineRail, type DashboardLiveEvent } from "./agent-timeline-rail";
import {
  deriveStage,
  stateMapForRun,
  type NodeState
} from "./node-config";
import { UploadView } from "./upload-view";
import { NODE_H, NODE_W, NodeCard } from "./node-card";
import { GateChips } from "./gate-chips";
import { PanWrap } from "./pan-wrap";
import { FloatingAgentWindow } from "./floating-agent-window";
import { AgentInfoPanel } from "./agent-info-panel";
import { ScriptPanel } from "./script-panel";
import { StatusPill, type Status } from "./status";
import { sourceTitle } from "./agent-info-helpers";
import { LabSidebar } from "./lab-sidebar";
import { TelemetryStrip } from "./telemetry-strip";
import { CommandPalette } from "./command-palette";
import { ShortcutOverlay } from "./shortcut-overlay";
import { ResizableSplit } from "./resizable-split";
import { useRun } from "@/hooks/use-run";
import { useCommandPalette } from "@/hooks/use-command-palette";
import { useCanvasKeyboardNav } from "@/hooks/use-canvas-keyboard-nav";
import { useShortcutOverlay } from "@/hooks/use-shortcut-overlay";
import { useTopology } from "@/hooks/use-topology";
import { TopologyProvider, useTopologyContext } from "@/lib/pipeline/topology-context";
import { PresentationModeProvider, usePresentationMode, type PresentationMode } from "@/lib/presentation-mode";
import { readUserPrefs, writeUserPref } from "@/lib/user-prefs";
import { issueText } from "./shared-helpers";

import "./lab-shell.css";

// Re-export so existing test imports keep working until Task 2.10's rename.
export { stateMapForRun };

type LabShellProps = {
  initialRun?: LiveDemoRunState | null;
  initialRecents?: RecentRunSummary[];
  initialTopology?: PipelineTopology | null;
  initialModels?: ModelChoice[];
  presentationMode?: PresentationMode;
};

function parseLogEntries(run: LiveDemoRunState | null) {
  if (!run?.log) {
    return [];
  }

  return run.log
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .slice(-80)
    .reverse()
    .map((line, index) => ({
      id: `${run.projectId}-${index}`,
      time: run.updatedAt ? new Date(run.updatedAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "--:--",
      // Log lines render verbatim — euphemising "failed" to "needs
      // attention" here hid real failures from anyone reading the log.
      msg: line
    }));
}

function telemetryForSelectedNode(
  run: LiveDemoRunState | null,
  selectedNode: LaidOutNode | null
) {
  if (!run?.telemetry?.length || !selectedNode) {
    return [];
  }
  const matches = selectedNode.agent_ids;
  return run.telemetry
    .filter((record) => matches.some((match) => record.agent_id?.includes(match)))
    .slice(-6)
    .reverse();
}

function failedNodeIdForRun(
  run: LiveDemoRunState,
  stateMap: Record<string, NodeState>,
  nodes: LaidOutNode[]
) {
  if (run.status !== "failed") {
    return null;
  }

  const runningNode = nodes.find((node) => stateMap[node.id] === "running");
  if (runningNode) {
    return runningNode.id;
  }

  const firstUpcoming = nodes.find((node) => stateMap[node.id] === "upcoming");
  if (firstUpcoming) {
    return firstUpcoming.id;
  }

  return "report";
}

function RunOverview({
  error,
  logEntries,
  run,
  stateMap
}: {
  error: string | null;
  logEntries: Array<{ id: string; msg: string; time: string }>;
  run: LiveDemoRunState;
  stateMap: Record<string, NodeState>;
}) {
  const { topology, layout } = useTopologyContext();
  const totals = layout.nodes.reduce(
    (acc, node) => {
      const state = stateMap[node.id];
      acc[state] += 1;
      return acc;
    },
    { done: 0, running: 0, upcoming: 0 }
  );

  const mode = usePresentationMode();
  // The "Improvement sub-agents" rollup uses the topology's declared
  // improvement_path_ids so adding/removing a path in topology.py is a
  // single edit.
  const nodeById = Object.fromEntries(layout.nodes.map((n) => [n.id, n]));
  const subagents = topology.improvement_path_ids
    .map((id) => nodeById[id])
    .filter((node): node is LaidOutNode => Boolean(node))
    .map((node) => ({
      id: node.id,
      node,
      state: stateMap[node.id],
      label: mode === "demo" ? node.demo_label : node.internal_label
    }));

  const title =
    run.status === "completed"
      ? "Run complete"
      : run.status === "failed"
        ? "Run needs attention"
        : "Reproducing live backend run";

  return (
    <div>
      <div className="eyebrow">Run</div>
      <div className="overview-title">{title}</div>
      <div className="overview-copy">
        {run.payload?.summary.stage
          ? `Current backend stage: ${run.payload.summary.stage}`
          : run.sourceNote ?? "Waiting for the first backend update."}
      </div>
      <div className="overview-grid">
        <Stat label="Done" value={totals.done} dot="var(--ink)" />
        <Stat label="Running" value={totals.running} dot="var(--accent)" pulse />
        <Stat label="Queued" value={totals.upcoming} dot="var(--line-2)" />
        <Stat label="Agents" value={layout.nodes.length} dot="var(--muted-2)" />
      </div>
      {error || run.error ? (
        <div className="agent-section">
          <div className="eyebrow">Issue</div>
          <div className="agent-detail">{issueText(error ?? run.error)}</div>
        </div>
      ) : null}
      {logEntries.length > 0 ? (
        <div className="agent-section">
          <div className="eyebrow">Latest backend log</div>
          <div className="agent-detail">{logEntries[0].msg}</div>
        </div>
      ) : null}
      <ScriptPanel run={run} />
      <div className="agent-section">
        <div className="eyebrow">Improvement sub-agents</div>
        <div className="subagent-list">
          {subagents.map((item) => (
            <div
              key={item.id}
              className="subagent-row"
              style={{
                background:
                  item.state === "running"
                    ? "var(--accent-soft)"
                    : item.state === "done"
                      ? "var(--bg)"
                      : "transparent"
              }}
            >
              <span
                className={item.state === "running" ? "pulse-dot subagent-dot" : "subagent-dot"}
                style={{
                  background:
                    item.state === "running"
                      ? "var(--accent)"
                      : item.state === "done"
                        ? "var(--ink)"
                        : "var(--line-2)"
                }}
              />
              <span className="subagent-name">{item.label}</span>
              <span className="subagent-step">{item.node.step}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function Stat({
  dot,
  label,
  pulse,
  value
}: {
  dot: string;
  label: string;
  pulse?: boolean;
  value: number;
}) {
  return (
    <div className="stat-card">
      <div className="stat-head">
        <span className={pulse ? "pulse-dot stat-dot" : "stat-dot"} style={{ background: dot }} />
        <span className="stat-label">{label}</span>
      </div>
      <div className="stat-value">{value}</div>
    </div>
  );
}

function RightPanel({
  error,
  run,
  selectedId,
  stateMap
}: {
  error: string | null;
  run: LiveDemoRunState;
  selectedId: string | null;
  stateMap: Record<string, NodeState>;
}) {
  const { layout } = useTopologyContext();
  const selected = selectedId
    ? layout.nodes.find((node) => node.id === selectedId) ?? null
    : null;
  const logEntries = parseLogEntries(run);
  const telemetry = telemetryForSelectedNode(run, selected);
  const failedNodeId = failedNodeIdForRun(run, stateMap, layout.nodes);
  const mode = usePresentationMode();
  const selectedLabel = selected
    ? mode === "demo"
      ? selected.demo_label
      : selected.internal_label
    : null;

  return (
    <aside className="card side-panel">
      <div className="side-panel-top">
        <div key={selectedId ?? "overview"} className="rp-pane side-panel-scroll">
          {selected ? (
            <AgentInfoPanel
              failedNodeId={failedNodeId}
              node={selected}
              state={stateMap[selected.id]}
              run={run}
              telemetry={telemetry}
              logEntries={logEntries}
            />
          ) : (
            <RunOverview run={run} stateMap={stateMap} logEntries={logEntries} error={error} />
          )}
        </div>
      </div>
      <div className="side-panel-bottom">
        <div className="side-panel-heading">
          <div className="side-panel-title">{selectedLabel ? `${selectedLabel} activity` : "Live activity"}</div>
          <span className="live-pill">
            <span className="pulse-dot live-pill-dot" />
            live
          </span>
        </div>
        <div className="side-panel-scroll">
          {logEntries.length > 0 ? (
            logEntries.map((entry, index) => (
              <div key={entry.id} className="event fadeup" style={{ animationDelay: `${index * 30}ms` }}>
                {/* Locale/timezone-formatted on both server and client now
                    that a run can be SSR'd — suppress the hydration diff. */}
                <span className="mono event-time" suppressHydrationWarning>
                  {entry.time}
                </span>
                <span className="event-dot" />
                <div className="mono event-message">{entry.msg}</div>
              </div>
            ))
          ) : (
            <div className="empty-activity">Waiting for backend activity...</div>
          )}
        </div>
      </div>
    </aside>
  );
}

// A draggable / resizable / scrollable live-agent feed that floats inside
// the canvas surface and anchors next to whichever node is currently
// running — it "follows" the pipeline as the active agent advances. Once
// the user drags it, it stays put; an "anchor" control snaps it back to
// following. Size is persisted; position resets to following each session.
function WorkflowView({
  busy,
  dashboardEvents,
  error,
  onClear,
  run
}: {
  busy: boolean;
  dashboardEvents: DashboardLiveEvent[];
  error: string | null;
  onClear: () => Promise<void>;
  run: LiveDemoRunState;
}) {
  const { topology, layout } = useTopologyContext();
  const [selectedId, setSelectedId] = useState<string | null>(null);
  // j/k traversal order: laid-out node order (left-to-right, then
  // top-to-bottom within a column). Derived so adding a node in
  // topology.py automatically extends the keyboard sequence.
  const order = useMemo(
    () =>
      [...layout.nodes]
        .sort((a, b) => (a.x === b.x ? a.y - b.y : a.x - b.x))
        .map((n) => n.id),
    [layout]
  );
  useCanvasKeyboardNav({ selectedId, onSelect: setSelectedId, enabled: true, order });

  const stateMap = useMemo(
    () => stateMapForRun(run, layout.nodes, topology.improvement_path_ids),
    [run, layout.nodes, topology.improvement_path_ids]
  );
  const doneCount = layout.nodes.filter((node) => stateMap[node.id] === "done").length;
  const liveStatus: Status =
    run.status === "failed"
      ? "attention"
      : run.status === "completed"
        ? "completed"
        : deriveStage(run) === "research_map_generated"
          ? "auditing"
          : run.status === "stopped"
            ? "stopped"
            : "running";
  const decisions = run.payload?.decisionLog ?? [];

  return (
    <>
      <div className="workflow-header">
        <div>
          <div className="eyebrow">workflow - {run.projectId}</div>
          <h1 className="h1 workflow-title">{sourceTitle(run)}</h1>
          <div className="workflow-meta">
            <StatusPill status={liveStatus} />
            <span className="workflow-meta-sep">.</span>
            <span className="mono">{doneCount}/{layout.nodes.length} agents complete</span>
          </div>
        </div>
        <div className="workflow-actions">
          <button className="btn btn-primary" onClick={() => void onClear()} type="button" disabled={busy}>
            {busy ? "Stopping…" : "Start New Run"}
          </button>
        </div>
      </div>
      <div className="workflow-stage">
        <ResizableSplit
          left={
            <div className="canvas-wrap canvas-wrap-full">
              <PanWrap
                run={run}
                stateMap={stateMap}
                selectedId={selectedId}
                onSelect={setSelectedId}
                dashboardEvents={dashboardEvents}
                decisions={decisions}
              />
            </div>
          }
          right={
            <RightPanel run={run} selectedId={selectedId} stateMap={stateMap} error={error} />
          }
        />
      </div>
      <TelemetryStrip run={run} />
    </>
  );
}


export function LabShell({
  initialRun = null,
  initialRecents = [],
  initialTopology = null,
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
    startFixtureRun,
    startUploadedRun,
    clearRun,
    resetToUpload: resetRun
  } = useRun(initialRun);
  // SSR-warmed topology — falls back to a client fetch via the hook.
  // The workflow view requires it; the upload view doesn't, so a null
  // topology is non-fatal until a run starts.
  const topology = useTopology(initialTopology);

  const resetToUpload = () => {
    setArxiv("");
    setOver(false);
    resetRun();
  };

  const palette = useCommandPalette();
  const shortcuts = useShortcutOverlay();

  const main = (
    <main className="content">
      {run ? (
        topology ? (
          <WorkflowView
            run={run}
            onClear={clearRun}
            busy={busy}
            error={error}
            dashboardEvents={dashboardEvents}
          />
        ) : (
          <div className="card" style={{ padding: 24 }}>
            <div className="eyebrow">Pipeline unavailable</div>
            <p style={{ marginTop: 8 }}>
              The pipeline topology could not be loaded. Reload the page or
              check that the backend is reachable.
            </p>
          </div>
        )
      ) : (
        <UploadView
          arxiv={arxiv}
          busy={busy}
          error={error}
          model={model}
          models={initialModels}
          onArxivChange={setArxiv}
          onArxivSubmit={() => void startFixtureRun(model)}
          onFileSelected={(file) => void startUploadedRun(file, model)}
          onModelChange={(value) => {
            setModel(value);
            writeUserPref("model", value);
          }}
          over={over}
          setOver={setOver}
        />
      )}
    </main>
  );

  // Wrap the layout in TopologyProvider only when topology is non-null;
  // child components that consume it (canvas, gate chips) only render
  // inside the WorkflowView, which is itself gated above.
  const layoutTree = (
    <div className="layout">
      <LabSidebar active="lab" onBrandClick={resetToUpload} recents={initialRecents} />
      {main}
    </div>
  );

  return (
    <div className="reproLab">
      <PresentationModeProvider mode={presentationMode}>
        {topology ? (
          <TopologyProvider topology={topology}>{layoutTree}</TopologyProvider>
        ) : (
          layoutTree
        )}
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
