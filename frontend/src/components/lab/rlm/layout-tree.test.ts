import { describe, it, expect } from "vitest";
import { layoutTree, ROW_HEIGHT } from "./layout-tree";
import type { TreeNode } from "../../../hooks/use-rlm-run";

const node = (id: string, parentId: string | null, kind: TreeNode["kind"] = "candidate"): TreeNode =>
  ({ id, parentId, kind, title: id } as unknown as TreeNode);

describe("layoutTree", () => {
  it("places the root at depth 0 and children to its right", () => {
    const { positioned } = layoutTree([node("p", null, "paper"), node("a", "p"), node("b", "p")]);
    const p = positioned.find((n) => n.id === "p")!;
    const a = positioned.find((n) => n.id === "a")!;
    expect(a.x).toBeGreaterThan(p.x);
  });
  it("stacks siblings vertically without overlap", () => {
    const { positioned } = layoutTree([node("p", null, "paper"), node("a", "p"), node("b", "p")]);
    const a = positioned.find((n) => n.id === "a")!;
    const b = positioned.find((n) => n.id === "b")!;
    expect(Math.abs(a.y - b.y)).toBe(ROW_HEIGHT);
  });
  it("emits one edge per parent->child relation", () => {
    const { edges } = layoutTree([node("p", null, "paper"), node("a", "p"), node("b", "a")]);
    expect(edges).toHaveLength(2);
    expect(edges.some((e) => e.from === "p" && e.to === "a")).toBe(true);
  });
  it("x increases with depth", () => {
    const { positioned } = layoutTree([
      node("p", null, "paper"), node("a", "p"), node("a2", "a"), node("a3", "a2")]);
    const xs = ["p", "a", "a2", "a3"].map((id) => positioned.find((n) => n.id === id)!.x);
    expect(xs).toEqual([...xs].sort((m, n) => m - n));
  });
  it("handles an empty tree", () => {
    expect(layoutTree([])).toEqual({ positioned: [], edges: [] });
  });
  it("does not throw when a child appears before its parent in the array", () => {
    // out-of-order: child first, then parent
    const { positioned, edges } = layoutTree([node("a", "p"), node("p", null, "paper")]);
    expect(positioned).toHaveLength(2);
    const p = positioned.find((n) => n.id === "p")!;
    const a = positioned.find((n) => n.id === "a")!;
    expect(a.x).toBeGreaterThan(p.x);
    expect(edges.some((e) => e.from === "p" && e.to === "a")).toBe(true);
  });
});
