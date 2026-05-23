import { render } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { Sparkline } from "./sparkline";

describe("Sparkline", () => {
  it("renders nothing when there are no points", () => {
    const { container } = render(<Sparkline series={[]} />);
    expect(container.querySelector("svg")).toBeNull();
  });

  it("renders a single circle when there is one point", () => {
    const { container } = render(<Sparkline series={[{ iteration: 1, score: 0.5 }]} />);
    expect(container.querySelector("svg")).not.toBeNull();
    expect(container.querySelectorAll("circle").length).toBe(1);
  });

  it("renders a polyline with N points for N>=2 points", () => {
    const series = [
      { iteration: 1, score: 0.1 },
      { iteration: 2, score: 0.4 },
      { iteration: 3, score: 0.7 },
    ];
    const { container } = render(<Sparkline series={series} />);
    const polyline = container.querySelector("polyline");
    expect(polyline).not.toBeNull();
    const points = polyline!.getAttribute("points")!.trim().split(/\s+/);
    expect(points.length).toBe(3);
  });

  it("clamps y-axis to [0,1]", () => {
    const series = [
      { iteration: 1, score: -0.2 },
      { iteration: 2, score: 1.5 },
    ];
    const { container } = render(<Sparkline series={series} width={100} height={20} />);
    const polyline = container.querySelector("polyline")!;
    const points = polyline.getAttribute("points")!.trim().split(/\s+/);
    const [, y1] = points[0].split(",").map(Number);
    const [, y2] = points[1].split(",").map(Number);
    // -0.2 clamps to 0 -> y=20 (bottom); 1.5 clamps to 1 -> y=0 (top).
    expect(y1).toBeCloseTo(20, 1);
    expect(y2).toBeCloseTo(0, 1);
  });
});
