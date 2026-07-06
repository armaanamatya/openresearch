import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { ReasoningChip } from "./ReasoningChip";
import styles from "./ReasoningChip.module.css";

describe("ReasoningChip", () => {
  it("renders the label text", () => {
    render(<ReasoningChip label="Loaded skill: orx" />);
    expect(screen.getByText("Loaded skill: orx")).toBeInTheDocument();
  });

  it("renders a status dot", () => {
    const { container } = render(<ReasoningChip label="List project runs" />);
    expect(container.querySelector(`.${styles.dot}`)).toBeInTheDocument();
  });

  it("defaults to done status", () => {
    const { container } = render(<ReasoningChip label="Fetch paper report" />);
    expect(container.firstElementChild).toHaveAttribute("data-status", "done");
    expect(container.firstElementChild).toHaveClass(styles.done);
  });

  it("honors an explicit active status (pulsing dot)", () => {
    const { container } = render(
      <ReasoningChip label="Check baseline run command" status="active" />
    );
    expect(container.firstElementChild).toHaveAttribute("data-status", "active");
    expect(container.firstElementChild).toHaveClass(styles.active);
  });

  it("honors an explicit pending status", () => {
    const { container } = render(<ReasoningChip label="Queued" status="pending" />);
    expect(container.firstElementChild).toHaveAttribute("data-status", "pending");
    expect(container.firstElementChild).toHaveClass(styles.pending);
  });

  it("merges a caller-provided className", () => {
    const { container } = render(
      <ReasoningChip label="Project overview" className="custom-x" />
    );
    expect(container.firstElementChild).toHaveClass("custom-x");
  });
});
