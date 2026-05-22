import { describe, it, expect } from "vitest";
import { render } from "@testing-library/react";
import { TreeEdges } from "./tree-edges";
import type { PositionedNode } from "./layout-tree";

describe("TreeEdges", () => {
  const positioned = [
    { id: "p", x: 0, y: 0 }, { id: "a", x: 100, y: 0 }, { id: "b", x: 100, y: 60 },
  ] as Pick<PositionedNode, "id" | "x" | "y">[] as PositionedNode[];
  it("renders one SVG path per edge", () => {
    const { container } = render(<TreeEdges positioned={positioned}
      edges={[{ from: "p", to: "a", outcome: "promoted" },
              { from: "p", to: "b", outcome: "failed" }]} />);
    expect(container.querySelectorAll("path")).toHaveLength(2);
  });
  it("dashes a failed/declined edge", () => {
    const { container } = render(<TreeEdges positioned={positioned}
      edges={[{ from: "p", to: "b", outcome: "failed" }]} />);
    const path = container.querySelector("path")!;
    expect(path.getAttribute("stroke-dasharray")).toBeTruthy();
  });
});
