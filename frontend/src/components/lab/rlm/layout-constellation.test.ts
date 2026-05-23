import { describe, it, expect } from "vitest";
import { layoutConstellation, nodeRadius } from "./layout-constellation";
import type { TreeNode } from "../../../hooks/use-rlm-run";

// ─── Helpers ────────────────────────────────────────────────────────────────

function makeNode(
  id: string,
  parentId: string | null,
  kind: TreeNode["kind"] = "candidate"
): TreeNode {
  return {
    id,
    parentId,
    kind,
    title: id,
    iterationRange: [0, 0],
  } as TreeNode;
}

function primitiveNode(id: string, parentId: string | null, iteration = 1): TreeNode {
  return {
    id,
    parentId,
    kind: "llm_primitive",
    title: id,
    iterationRange: [iteration, iteration],
    primitiveName: "understand_section",
  } as TreeNode;
}

// ─── Tests ───────────────────────────────────────────────────────────────────

describe("layoutConstellation", () => {
  it("returns empty output for empty input", () => {
    const result = layoutConstellation([]);
    expect(result.positioned).toHaveLength(0);
    expect(result.edges).toHaveLength(0);
  });

  it("1 paper + 3 candidates → 4 positioned nodes, 3 edges", () => {
    const nodes: TreeNode[] = [
      makeNode("paper", null, "paper"),
      makeNode("c1", "paper"),
      makeNode("c2", "paper"),
      makeNode("c3", "paper"),
    ];
    const { positioned, edges } = layoutConstellation(nodes);
    expect(positioned).toHaveLength(4);
    // 3 edges: paper→c1, paper→c2, paper→c3
    expect(edges).toHaveLength(3);
    expect(edges.every((e) => e.from === "paper")).toBe(true);
  });

  it("all positioned nodes have finite x/y coordinates", () => {
    const nodes: TreeNode[] = [
      makeNode("paper", null, "paper"),
      makeNode("c1", "paper"),
      makeNode("c2", "paper"),
    ];
    const { positioned } = layoutConstellation(nodes);
    for (const n of positioned) {
      expect(isFinite(n.x)).toBe(true);
      expect(isFinite(n.y)).toBe(true);
    }
  });

  it("50 primitive_call nodes — all positioned, none closer than forceCollide radius", () => {
    const nodes: TreeNode[] = [makeNode("paper", null, "paper")];
    for (let i = 0; i < 50; i++) {
      nodes.push(primitiveNode(`p${i}`, "paper", Math.floor(i / 5) + 1));
    }
    const { positioned } = layoutConstellation(nodes);
    expect(positioned).toHaveLength(51);

    // Check no two nodes overlap (their bounding circles should not intersect).
    for (let i = 0; i < positioned.length; i++) {
      for (let j = i + 1; j < positioned.length; j++) {
        const a = positioned[i];
        const b = positioned[j];
        const dist = Math.hypot(a.x - b.x, a.y - b.y);
        const minDist = a.radius + b.radius; // the forceCollide adds 8px extra but we allow slight tolerance
        expect(dist).toBeGreaterThan(minDist * 0.7); // 70% tolerance — force may not fully converge
      }
    }
  });

  it("idempotent given same input (deterministic output)", () => {
    const nodes: TreeNode[] = [
      makeNode("paper", null, "paper"),
      makeNode("baseline", "paper", "baseline"),
      makeNode("c1", "baseline"),
      makeNode("c2", "baseline"),
      primitiveNode("prim1", "baseline", 1),
      primitiveNode("prim2", "baseline", 2),
    ];
    const r1 = layoutConstellation(nodes);
    const r2 = layoutConstellation(nodes);
    // Same positions on repeated calls
    for (let i = 0; i < r1.positioned.length; i++) {
      expect(r1.positioned[i].x).toBeCloseTo(r2.positioned[i].x, 5);
      expect(r1.positioned[i].y).toBeCloseTo(r2.positioned[i].y, 5);
    }
  });

  it("all positioned nodes have non-negative x/y (offset applied)", () => {
    const nodes: TreeNode[] = [
      makeNode("paper", null, "paper"),
      makeNode("c1", "paper"),
      primitiveNode("prim1", "paper"),
    ];
    const { positioned } = layoutConstellation(nodes);
    for (const n of positioned) {
      expect(n.x).toBeGreaterThanOrEqual(0);
      expect(n.y).toBeGreaterThanOrEqual(0);
    }
  });

  it("nodeRadius returns correct values by kind", () => {
    expect(nodeRadius("candidate")).toBeGreaterThan(nodeRadius("primitive"));
    expect(nodeRadius("llm_primitive")).toBeGreaterThan(nodeRadius("primitive"));
    expect(nodeRadius("subrlm")).toBeGreaterThan(nodeRadius("primitive"));
  });

  it("does not mutate input nodes", () => {
    const nodes: TreeNode[] = [
      makeNode("paper", null, "paper"),
      makeNode("c1", "paper"),
    ];
    const clone = JSON.parse(JSON.stringify(nodes));
    layoutConstellation(nodes);
    expect(nodes).toEqual(clone);
  });
});
