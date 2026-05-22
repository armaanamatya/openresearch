import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { RubricStrip } from "./rubric-strip";

describe("RubricStrip", () => {
  const rubric = { current: 0.53, baseline: 0.22, target: 0.4,
    series: [{ iteration: 9, score: 0.22 }, { iteration: 13, score: 0.53 }], areas: [] };
  it("shows the current score and the target", () => {
    render(<RubricStrip rubric={rubric} />);
    expect(screen.getByText("0.53")).toBeInTheDocument();
    expect(screen.getByText(/0\.40/)).toBeInTheDocument();
  });
  it("shows the climb from baseline", () => {
    render(<RubricStrip rubric={rubric} />);
    expect(screen.getByText(/0\.22/)).toBeInTheDocument();
  });
  it("renders one sparkline bar per series point", () => {
    const { container } = render(<RubricStrip rubric={rubric} />);
    expect(container.querySelectorAll("[data-spark-bar]")).toHaveLength(2);
  });
  it("handles a not-yet-scored run", () => {
    render(<RubricStrip rubric={{ current: null, baseline: null, target: 0.4, series: [], areas: [] }} />);
    expect(screen.getByText(/—|not scored/i)).toBeInTheDocument();
  });
});
