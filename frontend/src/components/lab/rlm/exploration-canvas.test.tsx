import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ExplorationCanvas } from "./exploration-canvas";
import { fold, INITIAL_RLM_STATE } from "../../../hooks/use-rlm-run";
import { rlmRunFixture } from "./__fixtures__/rlm-run.fixture";

const state = rlmRunFixture.reduce(fold, INITIAL_RLM_STATE);

describe("ExplorationCanvas", () => {
  it("renders a node for every tree node", () => {
    render(<ExplorationCanvas tree={state.tree} iterations={state.iterations} />);
    expect(screen.getAllByRole("button").length).toBeGreaterThanOrEqual(state.tree.length);
  });
  it("opens the detail popup when a node is clicked", () => {
    render(<ExplorationCanvas tree={state.tree} iterations={state.iterations} />);
    fireEvent.click(screen.getAllByRole("button")[1]);
    expect(screen.getByTestId("node-detail-popup")).toBeInTheDocument();
  });
  it("collapses a fan beyond the 8-node soft cap into a +N node", () => {
    const big = [
      { id: "b", parentId: null, kind: "baseline", title: "baseline", iterationRange: [1, 1] },
      ...Array.from({ length: 12 }, (_, i) => ({
        id: `c${i}`, parentId: "b", kind: "candidate", title: `c${i}`,
        round: 1, iterationRange: [2, 2] as [number, number],
      })),
    ] as unknown as typeof state.tree;
    render(<ExplorationCanvas tree={big} iterations={[]} />);
    expect(screen.getByText(/\+\d+ more/)).toBeInTheDocument();
  });
});
