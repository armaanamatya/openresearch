"use client";

import type { LiveDemoRunState } from "@/lib/demo/demo-run-types";
import { usePan } from "@/hooks/use-pan";
import type { DashboardLiveEvent } from "./agent-timeline-rail";
import { LabCanvas } from "./lab-canvas";
import type { NodeState } from "./node-config";

export function PanWrap({
  onSelect,
  run,
  selectedId,
  stateMap,
  dashboardEvents,
  decisions
}: {
  onSelect: (id: string | null) => void;
  run: LiveDemoRunState;
  selectedId: string | null;
  stateMap: Record<string, NodeState>;
  dashboardEvents: DashboardLiveEvent[];
  decisions: string[];
}) {
  const { wrapRef, dragRef, onMouseDown } = usePan();

  return (
    <div ref={wrapRef} className="pan-wrap" onMouseDown={onMouseDown}>
      <LabCanvas
        run={run}
        stateMap={stateMap}
        selectedId={selectedId}
        dashboardEvents={dashboardEvents}
        decisions={decisions}
        onSelect={(id) => {
          if (!dragRef.current.moved) {
            onSelect(id);
          }
        }}
      />
    </div>
  );
}
