import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { Stepper } from "./Stepper";
import styles from "./Stepper.module.css";

const STEPS = [
  { label: "Ask the agent what to try", status: "done" as const },
  { label: "Agent iterates on the codebase", status: "active" as const },
  { label: "Inspect runs, steer the agent", status: "pending" as const },
];

describe("Stepper", () => {
  it("renders every step's label, in order", () => {
    render(<Stepper steps={STEPS} />);
    const items = screen.getAllByRole("listitem");
    expect(items).toHaveLength(3);
    expect(items[0]).toHaveTextContent("Ask the agent what to try");
    expect(items[1]).toHaveTextContent("Agent iterates on the codebase");
    expect(items[2]).toHaveTextContent("Inspect runs, steer the agent");
  });

  it("renders a real ordered list", () => {
    const { container } = render(<Stepper steps={STEPS} />);
    expect(container.querySelector("ol")).toBeInTheDocument();
  });

  it("marks the done step with data-status and the done class", () => {
    render(<Stepper steps={STEPS} />);
    const item = screen.getByText("Ask the agent what to try").closest("li");
    expect(item).toHaveAttribute("data-status", "done");
    expect(item).toHaveClass(styles.done);
  });

  it("marks the active step with data-status, the active class, and aria-current", () => {
    render(<Stepper steps={STEPS} />);
    const item = screen.getByText("Agent iterates on the codebase").closest("li");
    expect(item).toHaveAttribute("data-status", "active");
    expect(item).toHaveClass(styles.active);
    expect(item).toHaveAttribute("aria-current", "step");
  });

  it("marks the pending step with data-status and the pending class, no aria-current", () => {
    render(<Stepper steps={STEPS} />);
    const item = screen.getByText("Inspect runs, steer the agent").closest("li");
    expect(item).toHaveAttribute("data-status", "pending");
    expect(item).toHaveClass(styles.pending);
    expect(item).not.toHaveAttribute("aria-current");
  });

  it("renders a live dot only on the active step", () => {
    render(<Stepper steps={STEPS} />);
    const activeItem = screen.getByText("Agent iterates on the codebase").closest("li");
    const doneItem = screen.getByText("Ask the agent what to try").closest("li");
    const pendingItem = screen.getByText("Inspect runs, steer the agent").closest("li");
    expect(activeItem?.querySelector(`.${styles.liveDot}`)).toBeInTheDocument();
    expect(doneItem?.querySelector(`.${styles.liveDot}`)).not.toBeInTheDocument();
    expect(pendingItem?.querySelector(`.${styles.liveDot}`)).not.toBeInTheDocument();
  });

  it("merges a caller-provided className onto the root", () => {
    const { container } = render(<Stepper steps={STEPS} className="custom-x" />);
    expect(container.firstElementChild).toHaveClass("custom-x");
  });
});
