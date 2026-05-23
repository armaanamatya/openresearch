/**
 * layoutTree — pure tree-layout function for the RLM exploration canvas.
 *
 * Spec: docs/superpowers/specs/2026-05-21-rlm-phase4-frontend-design.md §8
 *
 * Algorithm (Reingold-Tilford-lite):
 *   1. Build a parentId → children[] adjacency map (preserving chronological order).
 *   2. Roots = nodes with parentId === null OR parentId not present in the id set.
 *   3. Post-order DFS from all roots, sharing a single `nextRow` counter:
 *      - Leaf: x = depth * COLUMN_WIDTH, y = nextRow++ * ROW_HEIGHT
 *      - Internal: x = depth * COLUMN_WIDTH, y = midpoint of children's y span
 *   4. Emit one Edge per node whose parentId resolves to a known id.
 *
 * Pure — never mutates the input nodes.
 */

import type { TreeNode } from "../../../hooks/use-rlm-run";

// ─── Exported types ────────────────────────────────────────────────────────────

export type PositionedNode = TreeNode & { x: number; y: number };

export interface Edge {
  from: string;
  to: string;
  /** Child's outcome — drives edge color/dash in ExplorationCanvas. */
  outcome?: TreeNode["outcome"];
}

// ─── Layout constants ──────────────────────────────────────────────────────────

export const COLUMN_WIDTH = 220;
// 120px gives 30-40px breathing room below the tallest node cards
// (multi-line title + subtitle + outcome pill ≈ 90-100px). The prior
// 80px caused adjacent vertical rows to visually touch.
export const ROW_HEIGHT = 120;

// ─── layoutTree ───────────────────────────────────────────────────────────────

export interface LayoutResult {
  positioned: PositionedNode[];
  edges: Edge[];
}

/**
 * layoutTree(nodes) → { positioned, edges }
 *
 * Safe to call on an empty array or on a forest (multiple roots).
 * Does not mutate `nodes`.
 */
export function layoutTree(nodes: TreeNode[]): LayoutResult {
  if (nodes.length === 0) return { positioned: [], edges: [] };

  // Build fast id-lookup and adjacency map (in insertion / chronological order).
  const byId = new Map<string, TreeNode>();
  for (const n of nodes) byId.set(n.id, n);

  // Two-pass build: first guarantee every id has an entry, then wire parent→child.
  // This makes insertion order irrelevant — a child that appears before its parent
  // in the array still resolves correctly.
  const children = new Map<string, string[]>();
  for (const n of nodes) children.set(n.id, []);
  for (const n of nodes) {
    if (n.parentId != null && byId.has(n.parentId)) {
      children.get(n.parentId)!.push(n.id);
    }
  }

  // Roots: no parentId, or parentId references an id not in the set.
  const roots = nodes.filter(
    (n) => n.parentId === null || !byId.has(n.parentId)
  );

  // Post-order DFS — `nextRow` shared across all roots so subtrees don't overlap.
  let nextRow = 0;
  const posMap = new Map<string, { x: number; y: number }>();
  const visited = new Set<string>();

  function dfs(id: string, depth: number): void {
    if (visited.has(id)) return; // cycle guard
    visited.add(id);

    const kids = children.get(id) ?? [];
    for (const kid of kids) dfs(kid, depth + 1);

    const x = depth * COLUMN_WIDTH;
    let y: number;

    if (kids.length === 0) {
      // Leaf — claim the next row.
      y = nextRow * ROW_HEIGHT;
      nextRow++;
    } else {
      // Internal — center vertically on children's y span.
      // Filter to only kids that were positioned; a cycle back-edge may have
      // skipped a kid (visited guard), so posMap.get(kid) could be undefined.
      const positionedKids = kids.filter((kid) => posMap.has(kid));
      if (positionedKids.length === 0) {
        // All children skipped (degenerate cycle) — treat as a leaf.
        y = nextRow * ROW_HEIGHT;
        nextRow++;
      } else {
        const kidYs = positionedKids.map((kid) => posMap.get(kid)!.y);
        const minY = Math.min(...kidYs);
        const maxY = Math.max(...kidYs);
        y = (minY + maxY) / 2;
      }
    }

    posMap.set(id, { x, y });
  }

  for (const root of roots) dfs(root.id, 0);

  // Any node not yet visited (disconnected via cycle or other anomaly) — place
  // them as additional roots so nothing is silently dropped.
  for (const n of nodes) {
    if (!visited.has(n.id)) dfs(n.id, 0);
  }

  // Build output arrays, preserving input order.
  const positioned: PositionedNode[] = nodes.map((n) => {
    const pos = posMap.get(n.id);
    if (pos === undefined) {
      // Every reachable node is visited by the DFS loops above; an absent entry
      // indicates a logic error — fail loud rather than silently misplace the node.
      throw new Error(
        `layoutTree: node "${n.id}" was not positioned — this is a bug in the DFS`
      );
    }
    return { ...n, x: pos.x, y: pos.y };
  });

  // Emit one edge per node that has a valid parent.
  const edges: Edge[] = nodes
    .filter((n) => n.parentId != null && byId.has(n.parentId))
    .map((n) => ({ from: n.parentId!, to: n.id, outcome: n.outcome }));

  return { positioned, edges };
}
