import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { Eyebrow } from "./Eyebrow";
import styles from "./Eyebrow.module.css";

describe("Eyebrow", () => {
  it("renders its children text", () => {
    render(<Eyebrow>welcome to openresearch</Eyebrow>);
    expect(screen.getByText("welcome to openresearch")).toBeInTheDocument();
  });

  it("applies uppercase styling via the eyebrow class (CSS text-transform, not text mutation)", () => {
    render(<Eyebrow>arxiv 2605.15155</Eyebrow>);
    expect(screen.getByText("arxiv 2605.15155")).toHaveClass(styles.eyebrow);
  });

  it("merges a caller-provided className", () => {
    render(<Eyebrow className="custom-x">label</Eyebrow>);
    expect(screen.getByText("label")).toHaveClass("custom-x");
  });
});
