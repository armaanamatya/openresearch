import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { Card } from "./Card";
import styles from "./Card.module.css";

describe("Card", () => {
  it("renders its children", () => {
    render(
      <Card>
        <p>Card body content</p>
      </Card>
    );
    expect(screen.getByText("Card body content")).toBeInTheDocument();
  });

  it("carries the neo-brutalist card class by default", () => {
    const { container } = render(<Card>content</Card>);
    const root = container.firstElementChild;
    expect(root).toHaveClass(styles.card);
    expect(root).toHaveClass(styles.default);
    expect(root).toHaveAttribute("data-variant", "default");
  });

  it("honors variant=\"panel\" with the softer panel class", () => {
    const { container } = render(<Card variant="panel">content</Card>);
    const root = container.firstElementChild;
    expect(root).toHaveClass(styles.card);
    expect(root).toHaveClass(styles.panel);
    expect(root).not.toHaveClass(styles.default);
    expect(root).toHaveAttribute("data-variant", "panel");
  });

  it("merges a caller-provided className", () => {
    const { container } = render(<Card className="custom-x">content</Card>);
    expect(container.firstElementChild).toHaveClass("custom-x");
  });
});
