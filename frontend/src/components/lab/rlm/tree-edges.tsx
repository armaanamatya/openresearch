"use client";

import { memo, useMemo } from "react";
import type { PositionedNode, Edge } from "./layout-tree";
import type { TreeNode } from "../../../hooks/use-rlm-run";

// ─── Outcome → stroke color (tokens.css vars) ─────────────────────────────────

const OUTCOME_STROKE: Record<NonNullable<TreeNode["outcome"]>, string> = {
  promoted: "var(--accent)",
  marginal:  "var(--warn)",
  failed:    "var(--err)",
  running:   "var(--hermes)",
  skipped:   "var(--muted-2)",
  declined:  "var(--muted-2)",
};

/** Outcomes that render as a dashed edge. */
const DASHED_OUTCOMES = new Set<NonNullable<TreeNode["outcome"]>>(["failed", "declined"]);

const STROKE_WIDTH = 1.5;
const STROKE_DASHARRAY = "4 3";
const FALLBACK_STROKE = "var(--line)";

// ─── Per-edge component ───────────────────────────────────────────────────────

interface EdgePathProps {
  from: PositionedNode;
  to: PositionedNode;
  outcome: Edge["outcome"];
}

const EdgePath = memo(function EdgePath({ from, to, outcome }: EdgePathProps) {
  const stroke = outcome ? (OUTCOME_STROKE[outcome] ?? FALLBACK_STROKE) : FALLBACK_STROKE;
  const dashed = outcome && DASHED_OUTCOMES.has(outcome);

  // Cubic bezier: depart horizontally from the parent, arrive horizontally
  // at the child — midpoint x is the inflection column.
  const midX = (from.x + to.x) / 2;
  const d = `M ${from.x},${from.y} C ${midX},${from.y} ${midX},${to.y} ${to.x},${to.y}`;

  return (
    <path
      d={d}
      stroke={stroke}
      strokeWidth={STROKE_WIDTH}
      strokeDasharray={dashed ? STROKE_DASHARRAY : undefined}
      fill="none"
    />
  );
});

// ─── TreeEdges ────────────────────────────────────────────────────────────────

export interface TreeEdgesProps {
  positioned: PositionedNode[];
  edges: Edge[];
}

/**
 * TreeEdges — SVG edge layer for the exploration tree (spec §8).
 *
 * Renders one <path> per edge, routed as a horizontal-first cubic bezier
 * (control points at the mid-x between parent and child). Stroke color comes
 * from the child's outcome palette (§9 → tokens.css). Failed/declined edges
 * are dashed. Edges whose from/to ids don't resolve to a positioned node are
 * skipped silently.
 *
 * The <svg> uses overflow="visible" so it never clips paths that extend
 * beyond the nominal bounding box. It is purely decorative (aria-hidden).
 *
 * D3: per-edge rendering is memoized via EdgePath (React.memo) — unchanged
 * edges skip reconciliation even when the positioned array grows (new nodes
 * appended). posById is memoized via useMemo to avoid rebuilding the Map on
 * every render.
 */
export function TreeEdges({ positioned, edges }: TreeEdgesProps) {
  // D3: memoize the id → position lookup so it is only rebuilt when positioned changes.
  const posById = useMemo(() => {
    const m = new Map<string, PositionedNode>();
    for (const n of positioned) m.set(n.id, n);
    return m;
  }, [positioned]);

  return (
    <svg
      aria-hidden="true"
      overflow="visible"
      style={{ position: "absolute", top: 0, left: 0, pointerEvents: "none" }}
    >
      {edges.map((edge) => {
        const from = posById.get(edge.from);
        const to = posById.get(edge.to);
        if (!from || !to) return null; // skip unresolved edges
        return (
          <EdgePath
            key={`${edge.from}-${edge.to}`}
            from={from}
            to={to}
            outcome={edge.outcome}
          />
        );
      })}
    </svg>
  );
}
