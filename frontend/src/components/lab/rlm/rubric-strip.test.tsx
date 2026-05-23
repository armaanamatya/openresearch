import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import type { RlmRunState } from "../../../hooks/use-rlm-run";
import { RubricStrip } from "./rubric-strip";

type Rubric = RlmRunState["rubric"];

const EMPTY_RUBRIC: Rubric = {
  current: null,
  baseline: null,
  target: 0.4,
  series: [],
  areas: [],
  previousAreas: [],
  attributableCandidate: null,
};

const CLIMB_RUBRIC: Rubric = {
  current: 0.53,
  baseline: 0.22,
  target: 0.4,
  series: [
    { iteration: 9, score: 0.22 },
    { iteration: 13, score: 0.53 },
  ],
  areas: [],
  previousAreas: [],
  attributableCandidate: null,
};

describe("RubricStrip — existing assertions (preserved)", () => {
  it("shows the current score and the target", () => {
    render(<RubricStrip rubric={CLIMB_RUBRIC} />);
    expect(screen.getByText("0.53")).toBeInTheDocument();
    expect(screen.getByText(/0\.40/)).toBeInTheDocument();
  });
  it("shows the climb from baseline", () => {
    render(<RubricStrip rubric={CLIMB_RUBRIC} />);
    expect(screen.getByText(/0\.22/)).toBeInTheDocument();
  });
  it("handles a not-yet-scored run", () => {
    render(<RubricStrip rubric={EMPTY_RUBRIC} />);
    expect(screen.getByText(/—|not scored/i)).toBeInTheDocument();
  });
});

describe("RubricStrip — Sparkline (spec §4.1)", () => {
  it("renders an SVG polyline when there are >= 2 series points", () => {
    const { container } = render(<RubricStrip rubric={CLIMB_RUBRIC} />);
    expect(container.querySelector("polyline")).not.toBeNull();
  });

  it("does not render an SVG when there are zero series points", () => {
    const { container } = render(<RubricStrip rubric={EMPTY_RUBRIC} />);
    expect(container.querySelector("polyline")).toBeNull();
  });
});

describe("RubricStrip — per-area chip row (spec §4.1, §4.2)", () => {
  it("renders one chip per area with a status data attribute", () => {
    const { container } = render(
      <RubricStrip
        rubric={{
          ...CLIMB_RUBRIC,
          areas: [
            { area: "Setup", score: 0.9, weight: 1, status: "pass" },
            { area: "Train", score: 0.45, weight: 1, status: "partial" },
          ],
        }}
      />,
    );
    const chips = container.querySelectorAll("[data-area-chip]");
    expect(chips.length).toBe(2);
  });

  it("applies data-just-flipped=true to areas that transitioned to a higher status", () => {
    const { container } = render(
      <RubricStrip
        rubric={{
          ...CLIMB_RUBRIC,
          areas: [
            { area: "Setup", score: 0.9, weight: 1, status: "pass" },
            { area: "Train", score: 0.1, weight: 1, status: "fail" },
          ],
          previousAreas: [
            { area: "Setup", score: 0.1, weight: 1, status: "fail" },
            { area: "Train", score: 0.1, weight: 1, status: "fail" },
          ],
        }}
      />,
    );
    const flipped = container.querySelectorAll('[data-area-chip][data-just-flipped="true"]');
    expect(flipped.length).toBe(1);
    expect(flipped[0].getAttribute("data-area")).toBe("Setup");
  });

  it("does not apply just-flipped on the first rubric_score (no previousAreas)", () => {
    const { container } = render(
      <RubricStrip
        rubric={{
          ...CLIMB_RUBRIC,
          areas: [{ area: "Setup", score: 0.9, weight: 1, status: "pass" }],
          previousAreas: [],
        }}
      />,
    );
    const flipped = container.querySelectorAll('[data-just-flipped="true"]');
    expect(flipped.length).toBe(0);
  });
});

describe("RubricStrip — candidate attribution (spec §4.1)", () => {
  it("renders attribution when last delta >= 0.05 and a candidate is live", () => {
    const { getByText } = render(
      <RubricStrip
        rubric={{
          ...CLIMB_RUBRIC,
          series: [
            { iteration: 1, score: 0.22 },
            { iteration: 5, score: 0.40 },
            { iteration: 7, score: 0.55 },
          ],
          attributableCandidate: { id: "c1", title: "Bigger batch", outcome: "promoted" },
        }}
      />,
    );
    expect(getByText(/from candidate Bigger batch/i)).toBeTruthy();
  });

  it("omits attribution when last delta < 0.05", () => {
    const { queryByText } = render(
      <RubricStrip
        rubric={{
          ...CLIMB_RUBRIC,
          series: [
            { iteration: 5, score: 0.53 },
            { iteration: 7, score: 0.55 },
          ],
          attributableCandidate: { id: "c1", title: "Bigger batch", outcome: "promoted" },
        }}
      />,
    );
    expect(queryByText(/from candidate/i)).toBeNull();
  });

  it("omits attribution when no candidate is live", () => {
    const { queryByText } = render(
      <RubricStrip
        rubric={{
          ...CLIMB_RUBRIC,
          attributableCandidate: null,
        }}
      />,
    );
    expect(queryByText(/from candidate/i)).toBeNull();
  });

  it("threshold boundary: 0.04 delta omits attribution, 0.05 includes it", () => {
    // Just below threshold — must NOT render.
    const below = render(
      <RubricStrip
        rubric={{
          ...CLIMB_RUBRIC,
          series: [
            { iteration: 5, score: 0.5 },
            { iteration: 7, score: 0.54 }, // delta = +0.04
          ],
          attributableCandidate: { id: "c1", title: "Marginal step", outcome: "marginal" },
        }}
      />,
    );
    expect(below.queryByText(/from candidate/i)).toBeNull();

    // At threshold — must render.
    const at = render(
      <RubricStrip
        rubric={{
          ...CLIMB_RUBRIC,
          series: [
            { iteration: 5, score: 0.5 },
            { iteration: 7, score: 0.55 }, // delta = +0.05
          ],
          attributableCandidate: { id: "c1", title: "On-threshold", outcome: "promoted" },
        }}
      />,
    );
    expect(at.getByText(/from candidate On-threshold/i)).toBeTruthy();
  });
});
