import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { TreeNode } from "./tree-node";
import type { TreeNode as TreeNodeData } from "../../../hooks/use-rlm-run";

const candidateBase = {
  id: "n1", parentId: "p", kind: "candidate", title: "learning-rate warmup",
  iterationRange: [10, 11], outcome: "promoted", rubricDelta: 0.14, round: 1,
  candidate: { id: "c1", title: "learning-rate warmup", category: "optimizer",
    description: "warm up the LR", reasoning: "stabilises early training" },
} as const;
const mk = (over: Partial<TreeNodeData> = {}): TreeNodeData =>
  ({ ...candidateBase, ...over } as unknown as TreeNodeData);

describe("TreeNode", () => {
  it("renders a candidate's title and outcome", () => {
    render(<TreeNode node={mk()} selected={false} onSelect={() => {}} />);
    expect(screen.getByText("learning-rate warmup")).toBeInTheDocument();
    expect(screen.getByText(/promoted/)).toBeInTheDocument();
  });
  it("carries a data-outcome attribute (outcome not by color alone)", () => {
    const { container } = render(<TreeNode node={mk()} selected={false} onSelect={() => {}} />);
    expect(container.firstChild).toHaveAttribute("data-outcome", "promoted");
  });
  it("renders a declined-group node with its count", () => {
    render(<TreeNode node={mk({ kind: "declined-group", declinedCount: 3, title: "3 declined" })}
      selected={false} onSelect={() => {}} />);
    expect(screen.getByText(/3 declined/)).toBeInTheDocument();
  });
  it("fires onSelect with the node id when clicked", () => {
    let picked = "";
    render(<TreeNode node={mk()} selected={false} onSelect={(id) => { picked = id; }} />);
    fireEvent.click(screen.getByRole("button"));
    expect(picked).toBe("n1");
  });
});
