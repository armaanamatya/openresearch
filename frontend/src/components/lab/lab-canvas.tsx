"use client";

import type { LiveDemoRunState } from "@/lib/demo/demo-run-types";
import type { LaidOutNode } from "@/lib/pipeline/layout";
import { useTopologyContext } from "@/lib/pipeline/topology-context";
import { stageProgressFromTopology } from "@/lib/pipeline/topology-helpers";
import type { DashboardLiveEvent } from "./agent-timeline-rail";
import { FloatingAgentWindow } from "./floating-agent-window";
import { GateChips } from "./gate-chips";
import type { NodeState } from "./node-config";
import { NODE_H, NODE_W, NodeCard } from "./node-card";

import "./lab-canvas.css";

function buildEdgePath(from: LaidOutNode, to: LaidOutNode) {
  const x1 = from.x + NODE_W;
  const y1 = from.y + NODE_H / 2;
  const x2 = to.x;
  const y2 = to.y + NODE_H / 2;
  const cx1 = x1 + Math.max(40, (x2 - x1) * 0.45);
  const cx2 = x2 - Math.max(40, (x2 - x1) * 0.45);
  return `M ${x1} ${y1} C ${cx1} ${y1}, ${cx2} ${y2}, ${x2} ${y2}`;
}

export function LabCanvas({
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
  const { topology, layout } = useTopologyContext();
  const nodesById: Record<string, LaidOutNode> = Object.fromEntries(
    layout.nodes.map((n) => [n.id, n])
  );

  function edgeState(from: string, to: string) {
    const source = stateMap[from];
    const target = stateMap[to];
    if (source === "done" && target === "done") {
      return "done" as const;
    }
    if (source === "done" && target === "running") {
      return "active" as const;
    }
    return "upcoming" as const;
  }

  const progress = stageProgressFromTopology(topology, run.payload?.summary?.stage);

  return (
    <div
      className="canvas-surface"
      style={{ width: layout.width, height: layout.height }}
    >
      <svg
        width={layout.width}
        height={layout.height}
        className="canvas-edges"
        aria-hidden="true"
      >
        {topology.edges.map((edge) => {
          const from = nodesById[edge.source];
          const to = nodesById[edge.target];
          if (!from || !to) return null;
          const state = edgeState(edge.source, edge.target);
          const path = buildEdgePath(from, to);
          let color = "var(--line-2)";
          let strokeWidth = 1.5;
          let opacity = 1;

          if (state === "upcoming") {
            opacity = 0.5;
          } else if (state === "done") {
            color = "var(--ink-2)";
            strokeWidth = 1.6;
          } else {
            color = "var(--accent)";
            strokeWidth = 2;
          }

          return (
            <g key={`${edge.source}-${edge.target}`} style={{ opacity }}>
              <path d={path} fill="none" stroke={color} strokeWidth={strokeWidth} strokeLinecap="round" />
              {state === "active" ? (
                <path
                  d={path}
                  fill="none"
                  stroke="var(--accent)"
                  strokeWidth="3"
                  strokeLinecap="round"
                  strokeDasharray="4 8"
                  className="wf-flow"
                  style={{ opacity: 0.7 }}
                />
              ) : null}
            </g>
          );
        })}
      </svg>
      {layout.nodes.map((node) => (
        <NodeCard
          key={node.id}
          node={node}
          state={stateMap[node.id]}
          selected={selectedId === node.id}
          progress={progress}
          onClick={() =>
            stateMap[node.id] === "upcoming" ? undefined : onSelect(node.id === selectedId ? null : node.id)
          }
        />
      ))}
      <GateChips run={run} />
      <FloatingAgentWindow events={dashboardEvents} decisions={decisions} stateMap={stateMap} />
    </div>
  );
}
