"use client";

import { useCallback, useMemo, useState } from "react";

import { NODE_W } from "@/components/lab/node-card";
import { layoutTopology } from "@/lib/pipeline/layout";
import type { PipelineTopology } from "@/lib/pipeline/topology";

import "./demo-overlay.css";

// The canvas is panned/scrolled, so the overlay is positioned with
// approximate fixed pixel coordinates relative to the viewport, biased
// toward the centre. For Phase B this is acceptable per the plan.
function tooltipPosition(node: { x: number; y: number }) {
  // Map the canvas-local coordinates (1740 × 720) into a screen-friendly
  // band so the tooltip stays visible even when the canvas is panned.
  // We bias roughly toward the top-center of the viewport and offset to
  // the right of the node by NODE_W + a small gap.
  const left = Math.min(Math.max(node.x + NODE_W + 24, 320), 1240);
  const top = Math.min(Math.max(node.y + 120, 180), 540);
  return { left, top };
}

// `topology` is the server-fetched pipeline topology — when present the
// overlay positions its tooltip using the laid-out node coordinates so
// adding/removing a node in topology.py reflows the tour automatically.
// Tour steps are derived from `topology.nodes.filter(n => n.tour_caption)`
// — adding a tour_caption on a node in backend/agents/topology.py adds a
// step here automatically, with no frontend edit.
export function DemoOverlay({
  topology
}: {
  topology?: PipelineTopology | null;
} = {}) {
  const [step, setStep] = useState(0);

  // Memoise the layout — the topology reference is stable across renders
  // since it comes from a server prop.
  const layout = useMemo(
    () => (topology ? layoutTopology(topology) : null),
    [topology]
  );

  const steps = useMemo(() => {
    if (!topology || !layout) return [];
    return topology.nodes
      .filter((n) => n.tour_caption)
      .map((n) => {
        const laidOut = layout.nodes.find((l) => l.id === n.id);
        return laidOut
          ? { nodeId: n.id, caption: n.tour_caption!, x: laidOut.x, y: laidOut.y }
          : null;
      })
      .filter((s): s is NonNullable<typeof s> => s !== null);
  }, [topology, layout]);

  const dismiss = useCallback(() => setStep(steps.length), [steps.length]);
  const next = useCallback(() => setStep((s) => Math.min(steps.length - 1, s + 1)), [steps.length]);
  const back = useCallback(() => setStep((s) => Math.max(0, s - 1)), []);

  if (steps.length === 0) {
    return null;
  }
  if (step >= steps.length) {
    return null;
  }
  if (!layout) {
    return null;
  }

  const current = steps[step];
  const node = layout.nodes.find((n) => n.id === current.nodeId);
  if (!node) {
    return null;
  }
  const pos = tooltipPosition(node);

  return (
    <div
      role="dialog"
      aria-label={`Demo tour, step ${step + 1} of ${steps.length}`}
      className="demo-overlay-card"
      style={{ left: pos.left, top: pos.top }}
    >
      <button
        type="button"
        className="demo-overlay-close"
        onClick={dismiss}
        aria-label="Dismiss tour"
      >
        ×
      </button>
      <div className="demo-overlay-step mono">
        Step {step + 1} of {steps.length}
      </div>
      <div className="demo-overlay-caption">{current.caption}</div>
      <div className="demo-overlay-actions">
        <button
          type="button"
          className="demo-overlay-btn demo-overlay-btn-secondary"
          onClick={back}
          disabled={step === 0}
        >
          Back
        </button>
        <button
          type="button"
          className="demo-overlay-btn demo-overlay-btn-secondary"
          onClick={dismiss}
        >
          Skip
        </button>
        {step === steps.length - 1 ? (
          <button
            type="button"
            className="demo-overlay-btn demo-overlay-btn-primary"
            onClick={dismiss}
          >
            Done
          </button>
        ) : (
          <button
            type="button"
            className="demo-overlay-btn demo-overlay-btn-primary"
            onClick={next}
          >
            Next
          </button>
        )}
      </div>
    </div>
  );
}

