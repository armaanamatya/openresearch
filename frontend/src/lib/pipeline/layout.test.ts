import { describe, it, expect } from "vitest";
import { layoutTopology } from "./layout";
import type { PipelineTopology } from "./topology";

describe("layoutTopology", () => {
  const t: PipelineTopology = {
    nodes: [
      { id: "a", kind: "source", internal_label: "a", demo_label: "A", step: "", role: "", detail: "", icon: "doc", tone: "neutral", agent_ids: [] },
      { id: "b", kind: "agent",  internal_label: "b", demo_label: "B", step: "", role: "", detail: "", icon: "doc", tone: "info", agent_ids: [] },
      { id: "c", kind: "agent",  internal_label: "c", demo_label: "C", step: "", role: "", detail: "", icon: "doc", tone: "info", agent_ids: [] }
    ],
    edges: [
      { source: "a", target: "b" },
      { source: "b", target: "c" }
    ],
    gates: [
      { id: "g1", before_node: "a", after_node: "b", label: "Gate 1" }
    ],
    stages: [],
    improvement_path_ids: []
  };

  it("places nodes in topological columns", () => {
    const layout = layoutTopology(t);
    const [a, b, c] = layout.nodes;
    expect(a.x).toBeLessThan(b.x);
    expect(b.x).toBeLessThan(c.x);
  });

  it("places gates at edge midpoints", () => {
    const layout = layoutTopology(t);
    const g = layout.gates[0];
    expect(g.x).toBeGreaterThan(0);
    expect(g.y).toBeGreaterThan(0);
  });

  it("computes total width/height to enclose all nodes", () => {
    const layout = layoutTopology(t);
    expect(layout.width).toBeGreaterThan(0);
    expect(layout.height).toBeGreaterThan(0);
  });
});
