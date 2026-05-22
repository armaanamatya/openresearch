/* eslint-disable @typescript-eslint/no-explicit-any */
import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { NodeDetailPopup } from "./node-detail-popup";

const node = {
  id: "n1", parentId: "p", kind: "candidate" as const, title: "longer warmup",
  iterationRange: [11, 13], outcome: "running" as const, round: 2,
  candidate: { id: "c5", title: "longer warmup", category: "optimizer",
    description: "extend warmup to 8k steps", reasoning: "stabilises late training" },
};
const iteration = { iteration: 11, response: "Extending warmup to 8k steps.",
  code_blocks: [{ code: "res = run_experiment(code, env)",
    stdout_meta: { length: 1400, prefix: "ok", has_traceback: false } }] };

describe("NodeDetailPopup", () => {
  it("shows the node title, iteration code, and reasoning", () => {
    render(<NodeDetailPopup node={node as any} iteration={iteration as any} onClose={() => {}} />);
    expect(screen.getByText(/longer warmup/)).toBeInTheDocument();
    expect(screen.getByText(/run_experiment/)).toBeInTheDocument();
    expect(screen.getByText(/Extending warmup/)).toBeInTheDocument();
  });
  it("calls onClose on the dismiss control and on Escape", () => {
    let closed = 0;
    render(<NodeDetailPopup node={node as any} iteration={iteration as any}
      onClose={() => { closed += 1; }} />);
    fireEvent.click(screen.getByRole("button", { name: /close|dismiss/i }));
    fireEvent.keyDown(document, { key: "Escape" });
    expect(closed).toBe(2);
  });
});
