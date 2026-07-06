import { render, screen } from "@testing-library/react";
import { describe, it, expect, beforeEach, vi } from "vitest";

const pushMock = vi.fn();
vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: pushMock, replace: vi.fn(), prefetch: vi.fn() }),
}));

import { SpecValidationStepper } from "./SpecValidationStepper";
import type { RlmDashboardEvent } from "@/lib/events/rlm-events";

const GENERATION_STARTED: RlmDashboardEvent = {
  event: "spec_generation_started",
  timestamp: "2026-07-06T00:00:00Z",
};

const GENERATED: RlmDashboardEvent = {
  event: "spec_generated",
  timestamp: "2026-07-06T00:00:01Z",
  leaf_count: 9,
};

const VALIDATION_STARTED: RlmDashboardEvent = {
  event: "spec_validation_started",
  timestamp: "2026-07-06T00:00:02Z",
  validator_model: "grok-4.3",
};

const VALIDATED_CLEAN: RlmDashboardEvent = {
  event: "spec_validated",
  timestamp: "2026-07-06T00:00:03Z",
  verdict: "verified",
  flagged_leaves: [],
};

const VALIDATED_FLAGGED: RlmDashboardEvent = {
  event: "spec_validated",
  timestamp: "2026-07-06T00:00:03Z",
  verdict: "flagged",
  flagged_leaves: ["L2", "L7"],
};

const UNRELATED: RlmDashboardEvent = {
  event: "run_warning",
  timestamp: "2026-07-06T00:00:05Z",
  level: "warn",
  code: "x",
  message: "y",
};

describe("SpecValidationStepper", () => {
  beforeEach(() => {
    pushMock.mockClear();
  });

  it("renders both steps pending before any spec-phase event, no navigation", () => {
    render(<SpecValidationStepper events={[]} runId="prj_test1" />);
    const items = screen.getAllByRole("listitem");
    expect(items).toHaveLength(2);
    expect(items[0]).toHaveAttribute("data-status", "pending");
    expect(items[1]).toHaveAttribute("data-status", "pending");
    expect(pushMock).not.toHaveBeenCalled();
  });

  it("marks step 1 active on spec_generation_started, without navigating", () => {
    render(<SpecValidationStepper events={[GENERATION_STARTED]} runId="prj_test1" />);
    const items = screen.getAllByRole("listitem");
    expect(items[0]).toHaveAttribute("data-status", "active");
    expect(items[1]).toHaveAttribute("data-status", "pending");
    expect(pushMock).not.toHaveBeenCalled();
  });

  it("marks step 1 done + shows the leaf count on spec_generated, without navigating", () => {
    render(
      <SpecValidationStepper events={[GENERATION_STARTED, GENERATED]} runId="prj_test1" />
    );
    const items = screen.getAllByRole("listitem");
    expect(items[0]).toHaveAttribute("data-status", "done");
    expect(items[0]).toHaveTextContent("9");
    expect(pushMock).not.toHaveBeenCalled();
  });

  it("marks step 2 active + shows the validator model on spec_validation_started, without navigating", () => {
    render(
      <SpecValidationStepper
        events={[GENERATION_STARTED, GENERATED, VALIDATION_STARTED]}
        runId="prj_test1"
      />
    );
    const items = screen.getAllByRole("listitem");
    expect(items[1]).toHaveAttribute("data-status", "active");
    expect(items[1]).toHaveTextContent("grok-4.3");
    expect(pushMock).not.toHaveBeenCalled();
  });

  it("marks step 2 done and navigates to /sessions/<runId> exactly once on spec_validated (clean verdict)", () => {
    const { rerender } = render(
      <SpecValidationStepper
        events={[GENERATION_STARTED, GENERATED, VALIDATION_STARTED]}
        runId="prj_test1"
      />
    );
    expect(pushMock).not.toHaveBeenCalled();

    rerender(
      <SpecValidationStepper
        events={[GENERATION_STARTED, GENERATED, VALIDATION_STARTED, VALIDATED_CLEAN]}
        runId="prj_test1"
      />
    );

    const items = screen.getAllByRole("listitem");
    expect(items[1]).toHaveAttribute("data-status", "done");
    expect(pushMock).toHaveBeenCalledTimes(1);
    expect(pushMock).toHaveBeenCalledWith("/sessions/prj_test1");

    // A further re-render on the same terminal state must not re-navigate.
    rerender(
      <SpecValidationStepper
        events={[GENERATION_STARTED, GENERATED, VALIDATION_STARTED, VALIDATED_CLEAN]}
        runId="prj_test1"
      />
    );
    expect(pushMock).toHaveBeenCalledTimes(1);
  });

  it("shows a non-blocking 'flagged' note but still navigates on a flagged verdict", () => {
    render(
      <SpecValidationStepper
        events={[GENERATION_STARTED, GENERATED, VALIDATION_STARTED, VALIDATED_FLAGGED]}
        runId="prj_test2"
      />
    );
    expect(screen.getByText(/2.*flagged/i)).toBeInTheDocument();
    expect(pushMock).toHaveBeenCalledWith("/sessions/prj_test2");
  });

  it("shows no flagged note on a clean verdict", () => {
    render(
      <SpecValidationStepper
        events={[GENERATION_STARTED, GENERATED, VALIDATION_STARTED, VALIDATED_CLEAN]}
        runId="prj_test1"
      />
    );
    expect(screen.queryByText(/flagged/i)).not.toBeInTheDocument();
  });

  it("ignores unrelated events interleaved in the stream", () => {
    render(
      <SpecValidationStepper
        events={[UNRELATED, GENERATION_STARTED, UNRELATED]}
        runId="prj_test1"
      />
    );
    const items = screen.getAllByRole("listitem");
    expect(items[0]).toHaveAttribute("data-status", "active");
    expect(pushMock).not.toHaveBeenCalled();
  });

  it("URI-encodes the runId in the redirect target", () => {
    render(
      <SpecValidationStepper
        events={[GENERATION_STARTED, GENERATED, VALIDATION_STARTED, VALIDATED_CLEAN]}
        runId="prj/weird id"
      />
    );
    expect(pushMock).toHaveBeenCalledWith(`/sessions/${encodeURIComponent("prj/weird id")}`);
  });
});
