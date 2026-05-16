"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { AgentTimelineRail, type DashboardLiveEvent } from "./agent-timeline-rail";
import type { NodeState } from "./node-config";
import { NODE_W } from "./node-card";
import { useTopologyContext } from "@/lib/pipeline/topology-context";
import { usePresentationMode } from "@/lib/presentation-mode";

import "./floating-agent-window.css";

const AGENT_WINDOW_KEY = "reprolab:agentWindow";
const AGENT_WINDOW_MIN = { w: 264, h: 208 };
const AGENT_WINDOW_DEFAULT = { w: 350, h: 320 };

export function FloatingAgentWindow({
  events,
  decisions,
  stateMap
}: {
  events: DashboardLiveEvent[];
  decisions: string[];
  stateMap: Record<string, NodeState>;
}) {
  const { layout } = useTopologyContext();
  const activeNode = useMemo(() => {
    const running = layout.nodes.find((node) => stateMap[node.id] === "running");
    if (running) return running;
    const lastDone = [...layout.nodes].reverse().find((node) => stateMap[node.id] === "done");
    return lastDone ?? layout.nodes[0];
  }, [layout.nodes, stateMap]);

  const mode = usePresentationMode();
  const activeLabel = mode === "demo" ? activeNode.demo_label : activeNode.internal_label;

  const [size, setSize] = useState(AGENT_WINDOW_DEFAULT);
  const [manualPos, setManualPos] = useState<{ x: number; y: number } | null>(null);
  const dragRef = useRef<
    { mode: "drag" | "resize"; px: number; py: number; ox: number; oy: number; ow: number; oh: number } | null
  >(null);

  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(AGENT_WINDOW_KEY);
      if (!raw) return;
      const saved = JSON.parse(raw) as { w?: number; h?: number };
      if (typeof saved.w === "number" && typeof saved.h === "number") {
        // eslint-disable-next-line react-hooks/set-state-in-effect
        setSize({
          w: Math.max(AGENT_WINDOW_MIN.w, saved.w),
          h: Math.max(AGENT_WINDOW_MIN.h, saved.h)
        });
      }
    } catch {
      // ignore — fall back to the default size
    }
  }, []);

  useEffect(() => {
    try {
      window.localStorage.setItem(AGENT_WINDOW_KEY, JSON.stringify(size));
    } catch {
      // non-fatal
    }
  }, [size]);

  // Anchor to the right of the active node; flip left when it would spill
  // off the canvas surface, and clamp within it. Width/height come from
  // the layout function so the clamp follows any topology change.
  const anchorRight = activeNode.x + NODE_W + 28;
  const anchorX =
    anchorRight + size.w > layout.width ? Math.max(8, activeNode.x - size.w - 28) : anchorRight;
  const anchorY = Math.min(Math.max(8, activeNode.y - 14), layout.height - size.h - 8);
  const x = manualPos?.x ?? anchorX;
  const y = manualPos?.y ?? anchorY;
  const following = manualPos === null;

  const beginPointer = useCallback(
    (mode: "drag" | "resize") => (event: React.MouseEvent) => {
      event.preventDefault();
      event.stopPropagation();
      dragRef.current = {
        mode,
        px: event.clientX,
        py: event.clientY,
        ox: x,
        oy: y,
        ow: size.w,
        oh: size.h
      };
      const move = (moveEvent: MouseEvent) => {
        const drag = dragRef.current;
        if (!drag) return;
        const dx = moveEvent.clientX - drag.px;
        const dy = moveEvent.clientY - drag.py;
        if (drag.mode === "drag") {
          setManualPos({ x: drag.ox + dx, y: drag.oy + dy });
        } else {
          setSize({
            w: Math.max(AGENT_WINDOW_MIN.w, drag.ow + dx),
            h: Math.max(AGENT_WINDOW_MIN.h, drag.oh + dy)
          });
        }
      };
      const end = () => {
        dragRef.current = null;
        window.removeEventListener("mousemove", move);
        window.removeEventListener("mouseup", end);
      };
      window.addEventListener("mousemove", move);
      window.addEventListener("mouseup", end);
    },
    [x, y, size.w, size.h]
  );

  return (
    <aside
      className={`agent-window${following ? " following" : ""}`}
      style={{ left: x, top: y, width: size.w, height: size.h }}
    >
      <header className="agent-window-head" onMouseDown={beginPointer("drag")}>
        <span className="agent-window-dot" aria-hidden="true" />
        <span className="agent-window-title">Live agents</span>
        <span className="agent-window-active" title={`Active agent: ${activeLabel}`}>
          {activeLabel}
        </span>
        {!following && (
          <button
            type="button"
            className="agent-window-anchor"
            onClick={() => setManualPos(null)}
            onMouseDown={(event) => event.stopPropagation()}
            title="Re-anchor to the active agent"
          >
            anchor
          </button>
        )}
      </header>
      <div className="agent-window-body">
        <AgentTimelineRail events={events} decisions={decisions} />
      </div>
      <span
        className="agent-window-resize"
        onMouseDown={beginPointer("resize")}
        aria-hidden="true"
      />
    </aside>
  );
}
