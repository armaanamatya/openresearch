"use client";

import { useState, useMemo, useEffect, useRef } from "react";
import type { DemoRunMode } from "../../../lib/demo/demo-run-types";
import type { RlmDashboardEvent } from "../../../lib/events/rlm-events";
import { useRlmRun } from "../../../hooks/use-rlm-run";
import { useSteeringChat } from "../../../hooks/use-steering-chat";
import { RlmHeader } from "./rlm-header";
import { RubricStrip } from "./rubric-strip";
import { ReplStateRail } from "./repl-state-rail";
import { ExplorationCanvas } from "./exploration-canvas";
import { ReportRail } from "./report-rail";
import { PrimitiveHistoryBar } from "./primitive-history-bar";
import { RubricBreakdown } from "./rubric-breakdown";
import { NodeDetailSidebar } from "./node-detail-sidebar";
import styles from "./rlm-lab.module.css";

interface RlmLabProps {
  events: RlmDashboardEvent[];
  runMeta: {
    projectId: string;
    paperTitle: string;
    paperMeta: string;
    /** ISO timestamp from demo_status.json; drives the real-time elapsed clock. */
    startedAt?: string | null;
  };
  /** Run mode — when "rlm" or "rdr" the RubricBreakdown panel is shown. */
  runMode?: DemoRunMode;
  /** Whether the run is still active (used to gate polling). */
  isActive?: boolean;
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
export function RlmLab({ events, runMeta, runMode, isActive = false }: RlmLabProps) {
  const state = useRlmRun(events);

  // ReplStateRail collapse state is owned here (the rail itself is a pure
  // presenter — it receives collapsed/onToggle props).
  const [replRailCollapsed, setReplRailCollapsed] = useState(false);

  // ── Lifted selection state ──────────────────────────────────────────────
  // The canvas notifies us via onSelectNode; we forward the id to both the
  // canvas (for highlight) and the sidebar (for detail).
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);

  // Derive the unique primitive names for the ReplStateRail primitives list.
  const primitiveNames = useMemo(
    () => [...new Set(state.primitiveCalls.map((c) => c.primitive))],
    [state.primitiveCalls]
  );

  // Real-time elapsed clock. If startedAt is provided, tick every second against
  // wall-clock now so the display updates while the run is in-flight. Fall back to
  // the event-timestamp span (static) when startedAt is absent.
  const startedAtMs = useMemo(
    () => (runMeta.startedAt ? new Date(runMeta.startedAt).getTime() : null),
    [runMeta.startedAt]
  );
  const [nowMs, setNowMs] = useState(() => Date.now());
  const tickRef = useRef<ReturnType<typeof setInterval> | null>(null);
  useEffect(() => {
    if (startedAtMs === null) return;
    setNowMs(Date.now());
    tickRef.current = setInterval(() => setNowMs(Date.now()), 1000);
    return () => {
      if (tickRef.current !== null) clearInterval(tickRef.current);
    };
  }, [startedAtMs]);
  const elapsedMs = useMemo(() => {
    if (startedAtMs !== null) return Math.max(0, nowMs - startedAtMs);
    // Fallback: derive from first/last event timestamps.
    if (events.length < 2) return 0;
    const first = new Date(events[0].timestamp).getTime();
    const last = new Date(events[events.length - 1].timestamp).getTime();
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
      />

      {/* Band 2 */}
      <RubricStrip rubric={state.rubric} />

      {/* RDR/RLM artifact panel — cluster grid, leaf scores, repair history */}
      {(runMode === "rlm" || runMode === "rdr") && (
        <RubricBreakdown projectId={runMeta.projectId} isActive={isActive} />
      )}

      {/* Band 3: workspace */}
      <div className={styles.workspace}>
        <ReplStateRail
          variables={state.variables}
          primitives={primitiveNames}
          collapsed={replRailCollapsed}
          onToggle={() => setReplRailCollapsed((c) => !c)}
        />
        <div className={styles.canvas}>
          <ExplorationCanvas
            tree={state.tree}
            iterations={state.iterations}
            selectedNodeId={selectedNodeId}
            onSelectNode={setSelectedNodeId}
          />
        </div>
        <ReportRail
          status={state.status}
          elapsedMs={elapsedMs}
          report={state.report}
          rubric={state.rubric}
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
        />
      </div>

      {/* Band 4 */}
      <PrimitiveHistoryBar calls={state.primitiveCalls} />
    </div>
  );
}
