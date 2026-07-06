import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";

// SpecValidationStepper (mounted on the spec-phase path) uses next/navigation.
vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), prefetch: vi.fn() }),
}));

import { SessionRouteContent } from "./page";
import type { RlmDashboardEvent } from "@/lib/events/rlm-events";

const SPEC_GENERATION_STARTED: RlmDashboardEvent = {
  event: "spec_generation_started",
  timestamp: "2026-07-06T00:00:00Z",
};

const REPL_ITERATION: RlmDashboardEvent = {
  event: "repl_iteration",
  timestamp: "2026-07-06T00:00:05Z",
  iteration: 1,
  response: "Reasoning body alpha — orienting on the project.",
  code_blocks: [],
  sub_calls: 0,
  timing: 1.0,
};

describe("SessionRouteContent — spec-stepper vs reasoning-view gate", () => {
  it("shows the SpecValidationStepper during the spec phase (no iterations yet)", () => {
    render(<SessionRouteContent runId="prj_x" events={[SPEC_GENERATION_STARTED]} />);
    // Screen D — the stepper is mounted.
    expect(screen.getByText(/Generating reproduction spec/i)).toBeInTheDocument();
    // The reasoning log is NOT the mounted view.
    expect(screen.queryByTestId("session-reasoning-view")).not.toBeInTheDocument();
    expect(screen.queryByText(/Waiting for the agent/i)).not.toBeInTheDocument();
  });

  it("swaps to the reasoning view the moment an iteration arrives (even mid-spec-phase)", () => {
    render(
      <SessionRouteContent runId="prj_x" events={[SPEC_GENERATION_STARTED, REPL_ITERATION]} />
    );
    // The reasoning log is mounted and renders the sanitized response.
    expect(screen.getByTestId("session-reasoning-view")).toBeInTheDocument();
    expect(screen.getByText(/Reasoning body alpha/)).toBeInTheDocument();
    // The stepper is gone.
    expect(screen.queryByText(/Generating reproduction spec/i)).not.toBeInTheDocument();
  });

  it("a non-spec run (no spec events, no iterations) shows the reasoning view's empty state, not the stepper", () => {
    render(<SessionRouteContent runId="prj_x" events={[]} />);
    expect(screen.getByText(/Waiting for the agent/i)).toBeInTheDocument();
    expect(screen.queryByText(/Generating reproduction spec/i)).not.toBeInTheDocument();
  });
});
