"use client";

import { useState, useMemo } from "react";
import type { RlmDashboardEvent } from "../../../lib/events/rlm-events";
import { useRlmRun } from "../../../hooks/use-rlm-run";
import { RlmHeader } from "./rlm-header";
import { RubricStrip } from "./rubric-strip";
import { ReplStateRail } from "./repl-state-rail";
import { ExplorationCanvas } from "./exploration-canvas";
import { ReportRail } from "./report-rail";
import { PrimitiveHistoryBar } from "./primitive-history-bar";
import styles from "./rlm-lab.module.css";

interface RlmLabProps {
  events: RlmDashboardEvent[];
  runMeta: {
    projectId: string;
    paperTitle: string;
    paperMeta: string;
  };
}

/**
 * RlmLab — 4-band shell composing the 6 RLM sub-components.
 *
 * Band 1: RlmHeader    (paper title, status, project id, cost)
 * Band 2: RubricStrip  (current/target score)
 * Band 3: workspace    ReplStateRail | ExplorationCanvas | ReportRail
 * Band 4: PrimitiveHistoryBar (collapsible)
 *
 * Spec: docs/superpowers/specs/2026-05-21-rlm-phase4-frontend-design.md §7 / §9 / §14
 */
export function RlmLab({ events, runMeta }: RlmLabProps) {
  const state = useRlmRun(events);

  // ReplStateRail collapse state is owned here (the rail itself is a pure
  // presenter — it receives collapsed/onToggle props).
  const [replRailCollapsed, setReplRailCollapsed] = useState(false);

  // Derive the unique primitive names for the ReplStateRail primitives list.
  const primitiveNames = useMemo(
    () => [...new Set(state.primitiveCalls.map((c) => c.primitive))],
    [state.primitiveCalls]
  );

  // Derive elapsed time from the first and last event timestamps.
  const elapsedMs = useMemo(() => {
    if (events.length < 2) return 0;
    const first = new Date(events[0].timestamp).getTime();
    const last = new Date(events[events.length - 1].timestamp).getTime();
    return Math.max(0, last - first);
  }, [events]);

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

      {/* Band 3: workspace */}
      <div className={styles.workspace}>
        <ReplStateRail
          variables={state.variables}
          primitives={primitiveNames}
          collapsed={replRailCollapsed}
          onToggle={() => setReplRailCollapsed((c) => !c)}
        />
        <div className={styles.canvas}>
          <ExplorationCanvas tree={state.tree} iterations={state.iterations} />
        </div>
        <ReportRail
          status={state.status}
          elapsedMs={elapsedMs}
          report={state.report}
          rubric={state.rubric}
        />
      </div>

      {/* Band 4 */}
      <PrimitiveHistoryBar calls={state.primitiveCalls} />
    </div>
  );
}
