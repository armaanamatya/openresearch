import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ReplStateRail } from "./repl-state-rail";

describe("ReplStateRail", () => {
  const props = {
    variables: {
      paper_text: { type: "str", size: 88000, firstSeenIteration: 1 },
      final_report: { type: "NoneType", size: 0, firstSeenIteration: 0 },
    },
    primitives: ["understand_section", "run_experiment"],
    collapsed: false,
    onToggle: () => {},
  };
  it("lists variables with their type", () => {
    render(<ReplStateRail {...props} />);
    expect(screen.getByText("paper_text")).toBeInTheDocument();
    expect(screen.getByText(/str/)).toBeInTheDocument();
  });
  it("lists the available primitives", () => {
    render(<ReplStateRail {...props} />);
    expect(screen.getByText(/understand_section/)).toBeInTheDocument();
  });
  it("calls onToggle when the collapse control is clicked", () => {
    let toggled = false;
    render(<ReplStateRail {...props} onToggle={() => { toggled = true; }} />);
    fireEvent.click(screen.getByRole("button", { name: /collapse|expand/i }));
    expect(toggled).toBe(true);
  });
});
