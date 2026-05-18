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
import { summariseFailure } from "./failure-summary";

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

// Build agent_id → started_at(ms) map from telemetry so per-line timestamps
// can be computed as `agent_start + (Ns)`. The (Ns) marker that backend
// log lines carry is elapsed-since-agent-start, not since-run-start; using
// the run-level updatedAt stamped every line with one frozen time and made
// the live feed look paused. Regression of commit 33ddc51, which lived in
// the now-split repro-lab-client.tsx.
function agentStartIndex(run: LiveDemoRunState | null): Map<string, number> {
  const index = new Map<string, number>();
  for (const record of run?.telemetry ?? []) {
    const id = record.agent_id;
    const startedAt = record.started_at;
    if (!id || !startedAt) continue;
    const ms = new Date(startedAt).getTime();
    if (!Number.isFinite(ms)) continue;
    // Keep the earliest start per agent so re-invoked agents anchor to
    // their first appearance — matches the log's chronological ordering.
    if (!index.has(id) || ms < (index.get(id) ?? Infinity)) {
      index.set(id, ms);
    }
  }
  return index;
}

// Lines with the elapsed marker: `[agent] (Ns) message`
const LOG_LINE_RE = /^\[([^\]]+)\]\s+\((\d+)s\)/;
// Agent-completion lines: `[agent] completed in Ns (...)`
const COMPLETED_LINE_RE = /^\[([^\]]+)\]\s+completed\s+in\s+(\d+)s/;

function formatTime(ms: number): string {
  return new Date(ms).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function parseLogEntries(run: LiveDemoRunState | null) {
  if (!run?.log) {
    return [];
  }

  const agentStarts = agentStartIndex(run);
  const runStart = run.startedAt ? new Date(run.startedAt).getTime() : null;
  const runUpdated = run.updatedAt ? new Date(run.updatedAt).getTime() : null;

  const lines = run.log
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  // Two-pass: stamp in original order so unmatched lines (separators,
  // transition markers like `> Starting: ...`, multi-line tool args)
  // can carry the timestamp forward from the most recent matched line.
  // Without this, ~half the feed reverted to runUpdated which is
  // essentially frozen — making the live view look stuck.
  let lastKnownMs: number | null = runStart;
  const stamped = lines.map((line, index) => {
    const match = line.match(LOG_LINE_RE) ?? line.match(COMPLETED_LINE_RE);
    let ms: number | null = null;
    if (match) {
      const agentId = match[1];
      const elapsedMs = Number(match[2]) * 1000;
      const anchor = agentStarts.get(agentId) ?? runStart;
      if (anchor != null && Number.isFinite(elapsedMs)) {
        ms = anchor + elapsedMs;
      }
    }
    if (ms == null) ms = lastKnownMs;
    if (ms != null) lastKnownMs = ms;
    return { line, ms, index };
  });

  return stamped.slice(-80).reverse().map((entry) => ({
    id: `${run.projectId}-${entry.index}`,
    time:
      entry.ms != null
        ? formatTime(entry.ms)
        : runUpdated != null
          ? formatTime(runUpdated)
          : "--:--",
    // Log lines render verbatim — euphemising "failed" to "needs
    // attention" here hid real failures from anyone reading the log.
    msg: entry.line
  }));
}

// Humanise a snake_case stage id for display: `gate_2_passed` -> `Gate 2 Passed`.
function humanizeStage(stageId: string): string {
  return stageId
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
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
  busy,
  error,
  logEntries,
  onResume,
  run,
  stateMap
}: {
  busy: boolean;
  error: string | null;
  logEntries: Array<{ id: string; msg: string; time: string }>;
  onResume: (overrides: Record<string, string>) => Promise<void>;
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

  // Translate `paper_understood` -> `Stage 2 of 14: Paper Understood` so a
  // glance at the header tells you where the pipeline is, not just the raw
  // enum value the backend ships. Falls back to the raw id if the stage
  // isn't in topology yet (defensive — topology is the source of truth).
  const currentStageId = run.payload?.summary.stage;
  const stageEntry = currentStageId
    ? topology.stages.find((s) => s.id === currentStageId)
    : undefined;
  const stageCopy = currentStageId
    ? stageEntry != null
      ? `Stage ${stageEntry.order + 1} of ${topology.stages.length}: ${humanizeStage(currentStageId)}`
      : `Current backend stage: ${humanizeStage(currentStageId)}`
    : null;

  return (
    <div>
      <div className="eyebrow">Run</div>
      <div className="overview-title">{title}</div>
      <div className="overview-copy">
        {stageCopy ?? run.sourceNote ?? "Waiting for the first backend update."}
      </div>
      <div className="overview-grid">
        <Stat label="Done" value={totals.done} dot="var(--ink)" />
        <Stat label="Running" value={totals.running} dot="var(--accent)" pulse />
        <Stat label="Queued" value={totals.upcoming} dot="var(--line-2)" />
        <Stat label="Agents" value={layout.nodes.length} dot="var(--muted-2)" />
      </div>
      {error || run.error ? (
        <FailurePanel
          rawError={error ?? run.error ?? null}
          busy={busy}
          onResume={onResume}
        />
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
  busy,
  error,
  onResume,
  run,
  selectedId,
  stateMap
}: {
  busy: boolean;
  error: string | null;
  onResume: (overrides: Record<string, string>) => Promise<void>;
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
            <RunOverview
              run={run}
              stateMap={stateMap}
              logEntries={logEntries}
              error={error}
              busy={busy}
              onResume={onResume}
            />
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
/**
 * Renders a failed/halted run's error in a way the operator can act on:
 * plain-English headline + explanation + remedy + (when applicable) a
 * button that calls /api/demo/resume with the right config overrides.
 * Raw error text is preserved behind a "Show technical details"
 * disclosure so power users still have it.
 *
 * Replaces the prior bare `<div>{issueText(error)}</div>` which forced
 * the operator to read a Python stack trace and figure out what to do.
 */
function FailurePanel({
  rawError,
  busy,
  onResume
}: {
  rawError: string;
  busy: boolean;
  onResume: (overrides: Record<string, string>) => Promise<void>;
}) {
  const summary = summariseFailure(rawError);
  const [showRaw, setShowRaw] = useState(false);
  if (!summary) {
    return (
      <div className="agent-section">
        <div className="eyebrow">Issue</div>
        <div className="agent-detail">{issueText(rawError)}</div>
      </div>
    );
  }
  return (
    <div className="agent-section" data-testid="failure-panel">
      <div className="eyebrow">Issue · {summary.kind.replace(/_/g, " ")}</div>
      {/* issueText keeps the existing "failed -> needs attention" euphemism
          contract for user-visible copy. Raw error stays verbatim under
          the technical-details disclosure for power users. */}
      <div className="agent-task" style={{ marginTop: 4 }}>{issueText(summary.headline)}</div>
      <div className="agent-detail" style={{ marginTop: 6 }}>{issueText(summary.explanation)}</div>
      <div className="agent-detail" style={{ marginTop: 6, fontWeight: 500 }}>
        Suggested fix: {summary.remedy}
      </div>
      {summary.action ? (
        <div style={{ marginTop: 10 }}>
          <button
            type="button"
            className="btn btn-primary"
            disabled={busy}
            onClick={() => {
              if (summary.action) {
                void onResume(summary.action.overrides);
              }
            }}
            data-testid="failure-action"
          >
            {busy ? "Working…" : summary.action.label}
          </button>
        </div>
      ) : null}
      <div style={{ marginTop: 10 }}>
        <button
          type="button"
          className="btn btn-sm"
          onClick={() => setShowRaw((prev) => !prev)}
          data-testid="failure-toggle-raw"
        >
          {showRaw ? "Hide technical details" : "Show technical details"}
        </button>
        {showRaw ? (
          <pre
            className="mono"
            style={{
              marginTop: 8,
              padding: 10,
              background: "var(--bg-2, #f6f6f7)",
              border: "1px solid var(--line)",
              borderRadius: 6,
              whiteSpace: "pre-wrap",
              maxHeight: 240,
              overflowY: "auto",
              fontSize: 12
            }}
          >
            {rawError}
          </pre>
        ) : null}
      </div>
    </div>
  );
}

function WorkflowView({
  busy,
  dashboardEvents,
  error,
  onClear,
  onResume,
  run
}: {
  busy: boolean;
  dashboardEvents: DashboardLiveEvent[];
  error: string | null;
  onClear: () => Promise<void>;
  onResume: (overrides: Record<string, string>) => Promise<void>;
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
            <RightPanel
              run={run}
              selectedId={selectedId}
              stateMap={stateMap}
              error={error}
              busy={busy}
              onResume={onResume}
            />
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
    startArxivRun,
    resumeRun,
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
            onResume={(overrides) => resumeRun(run.projectId, overrides)}
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
