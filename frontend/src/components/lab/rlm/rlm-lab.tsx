"use client";

import { useState, useMemo, useCallback, useEffect, useRef } from "react";
import type { DemoRunMode, DemoSandboxMode, DemoWorkerReport } from "../../../lib/demo/demo-run-types";
import type { RlmDashboardEvent } from "../../../lib/events/rlm-events";
import type { PrimitiveCallView } from "../../../hooks/use-rlm-run";
import { useRlmRunBatched } from "../../../hooks/use-rlm-run";
import { useSteeringChat } from "../../../hooks/use-steering-chat";
import { useResizablePanels } from "../../../hooks/use-resizable-panels";
import { useRerun } from "../../../hooks/use-rerun";
import { useWorkerReports } from "../../../hooks/use-worker-reports";
import { RlmHeader } from "./rlm-header";
import { LiveActivityStrip } from "./live-activity-strip";
import { PipelinePhaseStrip } from "./pipeline-phase-strip";
import { RubricStrip } from "./rubric-strip";
import { ReplStateRail } from "./repl-state-rail";
import { ConstellationCanvas } from "./constellation-canvas";
import { ReportRail } from "./report-rail";
import { PrimitiveHistoryBar } from "./primitive-history-bar";
import { RubricBreakdown } from "./rubric-breakdown";
import { ScorecardPanel } from "./scorecard-panel";
import { NodeDetailSidebar } from "./node-detail-sidebar";
import { ResizeHandle } from "./resize-handle";
import { RunToasts } from "./run-toasts";
import styles from "./rlm-lab.module.css";

interface RlmLabProps {
  events: RlmDashboardEvent[];
  runMeta: {
    projectId: string;
    paperTitle: string;
    paperMeta: string;
    /** ISO timestamp from demo_status.json; drives the real-time elapsed clock. */
    startedAt?: string | null;
    /** ISO timestamp when the run reached a terminal state. When present, the
     *  elapsed clock FREEZES at (completedAt - startedAt) instead of ticking
     *  against the wall clock — avoids the misleading "elapsed: 5h" badge on
     *  a run that died 10 min in. */
    completedAt?: string | null;
  };
  /** Run mode — when "rlm" or "rdr" the RubricBreakdown panel is shown. */
  runMode?: DemoRunMode;
  /** Whether the run is still active (used to gate polling). */
  isActive?: boolean;
  /** Error string from the run state — surfaced in the failed-run banner. */
  runError?: string | null;
  sandboxMode?: DemoSandboxMode | null;
  workerReports?: DemoWorkerReport[];
}

function findInFlightPrimitive(calls: PrimitiveCallView[]): { name: string; startedAt: string } | null {
  for (let i = calls.length - 1; i >= 0; i--) {
    const c = calls[i];
    if (c.status !== "start") continue;
    let terminated = false;
    for (let j = i + 1; j < calls.length; j++) {
      if (calls[j].primitive === c.primitive && calls[j].status !== "start") {
        terminated = true;
        break;
      }
    }
    if (!terminated) return { name: c.primitive, startedAt: c.timestamp };
  }
  return null;
}

/**
 * RlmLab — 4-band shell composing the 6 RLM sub-components.
 *
 * Band 1: RlmHeader    (paper title, status, project id, cost)
 * Band 2: RubricStrip  (current/target score)
 * Band 3: workspace    ReplStateRail | ExplorationCanvas | NodeDetailSidebar
 * Band 4: PrimitiveHistoryBar (collapsible)
 *
 * Selection state is lifted here so both the canvas (highlight) and the
 * sidebar (detail content) subscribe to the same source of truth.
 *
 * Spec: docs/superpowers/specs/2026-05-21-rlm-phase4-frontend-design.md §7 / §9 / §14
 */
export function RlmLab({
  events,
  runMeta,
  runMode,
  isActive = false,
  runError = null,
  sandboxMode = null,
  workerReports = [],
}: RlmLabProps) {
  // Pass the events array to useRlmRunBatched so its lazy state initializer
  // folds them synchronously on the FIRST render. This preserves the behaviour
  // of the original useRlmRun(events) for tests and fixture-replay paths that
  // render with a full event array already in hand. Subsequent changes to the
  // `events` prop do NOT re-run the initializer (React lazy init runs once).
  const { state, addEvent, reset } = useRlmRunBatched(events);
  const { rerun, busy: rerunBusy } = useRerun(runMeta.projectId);

  // Feed new events into the batched hook. We track how many events have been
  // fed so far via a ref; on each render where `events` has grown, we push
  // only the new tail into addEvent (preserving order, no replays). When
  // `events` shrinks (new run started, array reset to []), we reset the hook.
  // fedCountRef starts at events.length since the initial array is already
  // folded synchronously by the hook's lazy initializer.
  const fedCountRef = useRef(events.length);
  useEffect(() => {
    const start = fedCountRef.current;
    const end = events.length;
    if (end < start) {
      // Array shrank — new run. Reset state and replay from the beginning.
      reset();
      fedCountRef.current = 0;
      for (let i = 0; i < end; i++) {
        addEvent(events[i]);
      }
      fedCountRef.current = end;
    } else {
      // Array grew — push only the new tail.
      for (let i = start; i < end; i++) {
        addEvent(events[i]);
      }
      fedCountRef.current = end;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [events]);

  const { sizes, dragHandle, collapsedByViewport } = useResizablePanels();

  // ReplStateRail collapse state is owned here (the rail itself is a pure
  // presenter — it receives collapsed/onToggle props).
  const [replRailCollapsed, setReplRailCollapsed] = useState(false);

  // NodeDetailSidebar internal collapsed state, lifted so the handle can be hidden.
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  // ── Lifted selection state ──────────────────────────────────────────────
  // The canvas notifies us via onSelectNode; we forward the id to both the
  // canvas (for highlight) and the sidebar (for detail).
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);

  // Derive the unique primitive names for the ReplStateRail primitives list.
  const primitiveNames = useMemo(
    () => [...new Set(state.primitiveCalls.map((c) => c.primitive))],
    [state.primitiveCalls]
  );
  const inFlightPrimitive = useMemo(
    () => findInFlightPrimitive(state.primitiveCalls),
    [state.primitiveCalls],
  );

  // Real-time elapsed clock. If startedAt is provided, tick every second against
  // wall-clock now so the display updates while the run is in-flight. Fall back to
  // the event-timestamp span (static) when startedAt is absent.
  //
  // SSR-safety: `nowMs` starts as `null`, so the server-rendered markup and the
  // client's first hydration pass both compute elapsed = 0 against the same
  // startedAtMs — no hydration mismatch. The useEffect below populates nowMs on
  // mount (client-only), starting the tick. This avoids the classic
  // "Date.now() during render differs server vs client" hydration error.
  const startedAtMs = useMemo(
    () => (runMeta.startedAt ? new Date(runMeta.startedAt).getTime() : null),
    [runMeta.startedAt]
  );
  // When the run finishes (terminal state), the backend stamps completedAt on
  // demo_status.json. We freeze elapsed at that timestamp so the badge stops
  // ticking on a dead/failed run — fixes the "elapsed: 5h" misleading display.
  const completedAtMs = useMemo(
    () => (runMeta.completedAt ? new Date(runMeta.completedAt).getTime() : null),
    [runMeta.completedAt]
  );
  const [nowMs, setNowMs] = useState<number | null>(null);
  const tickRef = useRef<ReturnType<typeof setInterval> | null>(null);
  useEffect(() => {
    if (startedAtMs === null) return;
    // If the run already has a completedAt, freeze immediately — no interval.
    if (completedAtMs !== null) {
      setNowMs(completedAtMs);
      return;
    }
    const update = () => setNowMs(Date.now());
    const initialTick = setTimeout(update, 0);
    tickRef.current = setInterval(update, 1000);
    return () => {
      clearTimeout(initialTick);
      if (tickRef.current !== null) clearInterval(tickRef.current);
    };
  }, [startedAtMs, completedAtMs]);
  const elapsedMs = useMemo(() => {
    if (startedAtMs !== null) {
      // Terminal run: freeze elapsed at (completedAt - startedAt). Otherwise
      // (running): tick against the wall clock via nowMs.
      const referenceMs = completedAtMs !== null ? completedAtMs : nowMs;
      if (referenceMs === null) return 0;
      return Math.max(0, referenceMs - startedAtMs);
    }
    // Fallback: derive from first/last event timestamps (static, SSR-safe).
    if (events.length < 2) return 0;
    const firstTimestamp = events[0].timestamp;
    const lastTimestamp = events[events.length - 1].timestamp;
    if (!firstTimestamp || !lastTimestamp) return 0;
    const first = new Date(firstTimestamp).getTime();
    const last = new Date(lastTimestamp).getTime();
    return Math.max(0, last - first);
  }, [startedAtMs, nowMs, events]);

  // ── Selected node + iteration resolution ──────────────────────────────
  const selectedNode = useMemo(
    () =>
      selectedNodeId == null
        ? null
        : state.tree.find((n) => n.id === selectedNodeId) ?? null,
    [selectedNodeId, state.tree]
  );

  // Derived aggregate counters for the sidebar strip.
  const candidatesProposed = useMemo(
    () => state.tree.filter((n) => n.kind === "candidate").length,
    [state.tree]
  );
  const candidatesPromoted = useMemo(
    () => state.tree.filter((n) => n.kind === "candidate" && n.outcome === "promoted").length,
    [state.tree]
  );

  const selectedIteration = useMemo(() => {
    if (!selectedNode) return null;
    const [lo, hi] = selectedNode.iterationRange;
    let best = null;
    for (const it of state.iterations) {
      if (it.iteration >= lo && it.iteration <= hi) {
        if (best === null || it.iteration > best.iteration) best = it;
      }
    }
    return best;
  }, [selectedNode, state.iterations]);

  // ── Chat ──────────────────────────────────────────────────────────────
  const { messages: chatMessages, send: sendChat, sending: chatSending } =
    useSteeringChat(runMeta.projectId, events);

  // ── Worker reports (live-refreshing via SSE events) ──────────────────
  const { workers: liveWorkerReports, summary: liveReportsSummary } =
    useWorkerReports(runMeta.projectId, events, isActive);
  // Merge: prefer live-fetched reports over prop-passed ones
  const effectiveWorkerReports = liveWorkerReports.length > 0
    ? liveWorkerReports
    : workerReports;

  // ── Stable callbacks for memoized children ─────────────────────────────
  // useCallback ensures function identity is stable across clock-tick renders
  // so React.memo on ConstellationCanvas / NodeDetailSidebar skips them.
  const handleToggleReplRail = useCallback(
    () => setReplRailCollapsed((c) => !c),
    []
  );
  const handleSidebarCollapsedChange = useCallback(
    (c: boolean) => setSidebarCollapsed(c),
    []
  );

  // Memoized style objects — new object identity every render breaks React.memo.
  const sidebarStyle = useMemo(
    () => (sidebarCollapsed ? undefined : { width: sizes.detailSidebar }),
    [sidebarCollapsed, sizes.detailSidebar]
  );
  const replRailStyle = useMemo(
    () => (replRailCollapsed ? undefined : { width: sizes.replRail }),
    [replRailCollapsed, sizes.replRail]
  );
  const reportRailStyle = useMemo(
    () => ({ width: sizes.reportRail }),
    [sizes.reportRail]
  );

  return (
    <div className={styles.shell} data-testid="rlm-lab">
      {/* Band 1 */}
      <RlmHeader
        paperTitle={runMeta.paperTitle}
        paperMeta={runMeta.paperMeta}
        projectId={runMeta.projectId}
        status={state.status}
        iterationCount={state.iterationCount}
        costUsd={state.report?.costUsd ?? null}
        warnings={state.warnings}
        lastHeartbeatAt={state.lastHeartbeatAt}
        heartbeatNowMs={nowMs}
        error={runError}
        onRerun={rerun}
        rerunBusy={rerunBusy}
        inFlightPrimitive={inFlightPrimitive}
        sandboxMode={sandboxMode}
        primitiveCalls={state.primitiveCalls}
      />

      {/* Band 1.5 — always-visible live activity narration.
       *  Built 2026-05-23 after the user reported "just 7 mins doing nothing"
       *  with a near-blank canvas while the agent was actively running a
       *  long primitive. The strip never goes blank: it derives the current
       *  activity from primitiveCalls + subRlms + iterationCount and ticks
       *  a seconds counter so the UI is visibly alive. */}
      <LiveActivityStrip
        status={state.status}
        iterationCount={state.iterationCount}
        primitiveCalls={state.primitiveCalls}
        subRlms={state.subRlms}
        lastHeartbeatAt={state.lastHeartbeatAt}
        nowMs={nowMs}
        startedAt={runMeta.startedAt}
      />

      <PipelinePhaseStrip
        status={state.status}
        primitiveCalls={state.primitiveCalls}
      />

      {/* Band 2 */}
      <RubricStrip rubric={state.rubric} />

      {/* Band 2.5 — Scorecard table (FIG § 5.1). Shows when rubric areas land. */}
      {state.rubric.areas.length > 0 && (
        <ScorecardPanel
          rubric={state.rubric}
          projectId={runMeta.projectId}
          paperTitle={runMeta.paperTitle}
        />
      )}

      {/* RDR/RLM artifact panel — cluster grid, leaf scores, repair history */}
      {(runMode === "rlm" || runMode === "rdr" || runMode === "rlm-pure") && (
        <RubricBreakdown
          projectId={runMeta.projectId}
          isActive={isActive}
          perModelMetrics={state.perModelMetrics}
        />
      )}

      {/* Band 3: workspace */}
      <div className={styles.workspace}>
        {!collapsedByViewport.replRail && (
          <>
            <ReplStateRail
              variables={state.variables}
              primitives={primitiveNames}
              collapsed={replRailCollapsed}
              onToggle={handleToggleReplRail}
              style={replRailStyle}
            />
            <ResizeHandle
              {...dragHandle("replRail", "right")}
              aria-valuenow={sizes.replRail}
              disabled={replRailCollapsed}
            />
          </>
        )}
        <div className={styles.canvas}>
          <ConstellationCanvas
            tree={state.tree}
            iterations={state.iterations}
            selectedNodeId={selectedNodeId}
            onSelectNode={setSelectedNodeId}
          />
        </div>
        {!collapsedByViewport.reportRail && (
          <>
            <ResizeHandle
              {...dragHandle("reportRail", "left")}
              aria-valuenow={sizes.reportRail}
            />
            <ReportRail
              status={state.status}
              elapsedMs={elapsedMs}
              report={state.report}
              rubric={state.rubric}
              workerReports={effectiveWorkerReports}
              primitiveCalls={state.primitiveCalls}
              reportsSummary={liveReportsSummary}
              style={reportRailStyle}
            />
          </>
        )}
        <ResizeHandle
          {...dragHandle("detailSidebar", "left")}
          aria-valuenow={sizes.detailSidebar}
          disabled={sidebarCollapsed}
        />
        <NodeDetailSidebar
          node={selectedNode}
          iteration={selectedIteration}
          primitiveCalls={state.primitiveCalls}
          paperMeta={runMeta.paperMeta}
          projectId={runMeta.projectId}
          chatMessages={chatMessages}
          onSendChat={sendChat}
          chatSending={chatSending}
          subRlms={state.subRlms}
          iterationCount={state.iterationCount}
          candidatesProposed={candidatesProposed}
          candidatesPromoted={candidatesPromoted}
          gpuPlan={state.gpuPlan}
          perModelMetrics={state.perModelMetrics}
          collapsed={sidebarCollapsed}
          onCollapsedChange={handleSidebarCollapsedChange}
          style={sidebarStyle}
        />
      </div>

      {/* Band 4 */}
      <PrimitiveHistoryBar calls={state.primitiveCalls} />
      <RunToasts
        status={state.status}
        iterationCount={state.iterationCount}
        primitiveCalls={state.primitiveCalls}
        report={state.report}
      />
    </div>
  );
}
