/**
 * layoutConstellation — force-directed layout for the RLM constellation view.
 *
 * Uses d3-force to position all node kinds (paper, work, baseline, candidate,
 * subrlm, declined-group, primitive, llm_primitive) with no overlap.
 *
 * Spec: live constellation — feature spec 2026-05-23.
 *
 * Design choices:
 *   - Candidates pulled left (toward x=0) so they remain prominent.
 *   - Primitives pushed toward horizontal bands by iteration number.
 *   - forceCollide prevents any two nodes touching.
 *   - Simulation is ticked synchronously (300 steps) — pure function, same input → same output.
 */

import {
  forceSimulation,
  forceLink,
  forceManyBody,
  forceX,
  forceY,
  forceCollide,
} from "d3-force";
import type { TreeNode } from "../../../hooks/use-rlm-run";
import type { Edge } from "./layout-tree";

// ─── Exported types ─────────────────────────────────────────────────────────

export type { Edge };

export type ConstellationNode = TreeNode & {
  x: number;
  y: number;
  /** Collision radius used by forceCollide — depends on node kind. */
  radius: number;
};

export interface ConstellationResult {
  positioned: ConstellationNode[];
  edges: Edge[];
}

// ─── Node radii by kind ──────────────────────────────────────────────────────

/** Half-diagonal of a rect node (used as collision radius for rect nodes). */
const CANDIDATE_HALF_W = 70; // half of 140px width
const CANDIDATE_HALF_H = 30; // half of 60px height
const CANDIDATE_RADIUS = Math.hypot(CANDIDATE_HALF_W, CANDIDATE_HALF_H);

export function nodeRadius(kind: TreeNode["kind"]): number {
  switch (kind) {
    case "candidate":
      return CANDIDATE_RADIUS;
    case "paper":
    case "work":
    case "baseline":
    case "declined-group":
      return 48;
    case "subrlm":
      return 24;
    case "llm_primitive":
      return 20;
    case "primitive":
      return 14;
    default:
      return 16;
  }
}

// ─── layoutConstellation ────────────────────────────────────────────────────

/**
 * layoutConstellation(nodes) → { positioned, edges }
 *
 * Pure — never mutates input. Same input → same output (deterministic ticking).
 * Safe to call on an empty array.
 */
export function layoutConstellation(nodes: TreeNode[]): ConstellationResult {
  if (nodes.length === 0) return { positioned: [], edges: [] };

  const byId = new Map<string, TreeNode>();
  for (const n of nodes) byId.set(n.id, n);

  // ── Build edges from parentId relationships ──────────────────────────────
  // Dedupe by `from-to` key so duplicate tree nodes (which shouldn't exist
  // after the foldCandidateProposed in-place update, but might if a fixture
  // or backend bug slips through) don't produce duplicate React keys at the
  // SVG layer. Later occurrence wins so the most recent outcome is preserved.
  const edgeByKey = new Map<string, Edge>();
  for (const n of nodes) {
    if (n.parentId == null || !byId.has(n.parentId)) continue;
    const key = `${n.parentId}-${n.id}`;
    edgeByKey.set(key, { from: n.parentId, to: n.id, outcome: n.outcome });
  }
  const edges: Edge[] = Array.from(edgeByKey.values());

  // ── Initial positions — seeded deterministically ─────────────────────────
  // Place nodes in a rough grid by kind to give the force simulation a good
  // starting configuration. This avoids degenerate all-at-origin explosions.
  const CANDIDATE_X_BAND = 600;
  const PRIMITIVE_X_BAND = 200;
  const ROW_STEP = 60;

  let candidateRow = 0;
  let primitiveRow = 0;
  let otherRow = 0;

  interface SimNode {
    id: string;
    x: number;
    y: number;
    fx?: number | null;
    fy?: number | null;
    radius: number;
    kind: TreeNode["kind"];
    iterationRange: [number, number];
  }

  const simNodes: SimNode[] = nodes.map((n) => {
    const r = nodeRadius(n.kind);
    let x: number;
    let y: number;
    switch (n.kind) {
      case "candidate":
        x = CANDIDATE_X_BAND;
        y = candidateRow * ROW_STEP;
        candidateRow++;
        break;
      case "primitive":
      case "llm_primitive": {
        // Spread primitives in the left-center area, grouped by iteration.
        const iter = n.iterationRange[0] || 0;
        x = PRIMITIVE_X_BAND + (iter % 3) * 80;
        y = primitiveRow * (ROW_STEP * 0.6);
        primitiveRow++;
        break;
      }
      default:
        x = 0;
        y = otherRow * ROW_STEP;
        otherRow++;
        break;
    }
    return { id: n.id, x, y, radius: r, kind: n.kind, iterationRange: n.iterationRange };
  });

  const simById = new Map<string, SimNode>();
  for (const s of simNodes) simById.set(s.id, s);

  // d3-force link objects — source/target are SimNode refs after initialization.
  // d3 mutates these in-place so source/target become SimNode objects post-init.
  interface LinkDatum {
    source: string | SimNode;
    target: string | SimNode;
  }
  const linkData: LinkDatum[] = edges
    .map((e) => ({ source: e.from as string | SimNode, target: e.to as string | SimNode }))
    .filter((l) => simById.has(l.source as string) && simById.has(l.target as string));

  // ── Force simulation — synchronous ticking ───────────────────────────────
  const sim = forceSimulation<SimNode>(simNodes)
    .force(
      "link",
      forceLink<SimNode, LinkDatum>(linkData)
        .id((d) => d.id)
        .distance((l) => {
          // Larger distance between candidates and their parents so the graph
          // is readable. Primitive→work links can be shorter.
          const target = l.target as SimNode;
          if (target.kind === "candidate") return 160;
          if (target.kind === "primitive" || target.kind === "llm_primitive") return 60;
          return 110;
        })
        .strength(0.5)
    )
    .force(
      "charge",
      forceManyBody<SimNode>().strength((d) => {
        // Candidates push harder so they spread out.
        if (d.kind === "candidate") return -300;
        if (d.kind === "primitive" || d.kind === "llm_primitive") return -60;
        return -150;
      })
    )
    .force(
      "collide",
      forceCollide<SimNode>().radius((d) => d.radius + 14).strength(1.0).iterations(3)
    )
    // Candidates are pulled toward the right (higher x); structural nodes toward left.
    .force(
      "x",
      forceX<SimNode>().x((d) => {
        if (d.kind === "candidate") return CANDIDATE_X_BAND;
        if (d.kind === "primitive" || d.kind === "llm_primitive") return PRIMITIVE_X_BAND;
        return 0;
      }).strength((d) => {
        if (d.kind === "candidate") return 0.15;
        if (d.kind === "primitive" || d.kind === "llm_primitive") return 0.1;
        return 0.08;
      })
    )
    // Primitives are pulled toward iteration bands (vertical grouping).
    .force(
      "y",
      forceY<SimNode>().y((d) => {
        if (d.kind === "primitive" || d.kind === "llm_primitive") {
          const iter = d.iterationRange[0] || 0;
          return iter * ROW_STEP * 1.2;
        }
        return 0;
      }).strength((d) => {
        if (d.kind === "primitive" || d.kind === "llm_primitive") return 0.15;
        return 0.02;
      })
    )
    .stop();

  // Tick synchronously — 300 steps is sufficient for convergence at ~50 nodes.
  sim.tick(300);

  // ── Build output — shift all positions to be non-negative ──────────────
  let minX = Infinity;
  let minY = Infinity;
  for (const s of simNodes) {
    if (s.x < minX) minX = s.x;
    if (s.y < minY) minY = s.y;
  }
  const offsetX = -minX + 80; // 80px left margin
  const offsetY = -minY + 80; // 80px top margin

  const positioned: ConstellationNode[] = nodes.map((n) => {
    const s = simById.get(n.id)!;
    return {
      ...n,
      x: s.x + offsetX,
      y: s.y + offsetY,
      radius: s.radius,
    };
  });

  return { positioned, edges };
}
