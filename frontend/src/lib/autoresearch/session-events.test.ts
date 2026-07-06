import { describe, it, expect } from "vitest";
import {
  foldSpecPhase,
  INITIAL_SPEC_PHASE_STATE,
  shouldShowSpecStepper,
  type SpecPhaseState,
} from "./session-events";
import type {
  SpecGenerationStartedEvent,
  SpecGeneratedEvent,
  SpecValidationStartedEvent,
  SpecValidatedEvent,
  RunWarningEvent,
} from "@/lib/events/rlm-events";

const GENERATION_STARTED: SpecGenerationStartedEvent = {
  event: "spec_generation_started",
  timestamp: "2026-07-06T00:00:00Z",
};

const GENERATED: SpecGeneratedEvent = {
  event: "spec_generated",
  timestamp: "2026-07-06T00:00:01Z",
  leaf_count: 12,
};

const VALIDATION_STARTED: SpecValidationStartedEvent = {
  event: "spec_validation_started",
  timestamp: "2026-07-06T00:00:02Z",
  validator_model: "grok-4.3",
};

const VALIDATED_FLAGGED: SpecValidatedEvent = {
  event: "spec_validated",
  timestamp: "2026-07-06T00:00:03Z",
  verdict: "flagged",
  flagged_leaves: ["L2", "L7"],
};

const UNRELATED: RunWarningEvent = {
  event: "run_warning",
  timestamp: "2026-07-06T00:00:04Z",
  level: "warn",
  code: "some_code",
  message: "hello",
};

describe("foldSpecPhase", () => {
  it("starts at idle", () => {
    expect(INITIAL_SPEC_PHASE_STATE.stage).toBe("idle");
  });

  it("transitions idle -> generating on spec_generation_started", () => {
    const next = foldSpecPhase(INITIAL_SPEC_PHASE_STATE, GENERATION_STARTED);
    expect(next.stage).toBe("generating");
  });

  it("transitions generating -> generated with leafCount on spec_generated", () => {
    const started = foldSpecPhase(INITIAL_SPEC_PHASE_STATE, GENERATION_STARTED);
    const generated = foldSpecPhase(started, GENERATED);
    expect(generated.stage).toBe("generated");
    expect(generated.leafCount).toBe(12);
  });

  it("transitions generated -> validating with validatorModel on spec_validation_started", () => {
    const state = [GENERATION_STARTED, GENERATED, VALIDATION_STARTED].reduce(
      foldSpecPhase,
      INITIAL_SPEC_PHASE_STATE
    );
    expect(state.stage).toBe("validating");
    expect(state.validatorModel).toBe("grok-4.3");
  });

  it("transitions validating -> validated with verdict + flaggedLeaves on spec_validated", () => {
    const state = [GENERATION_STARTED, GENERATED, VALIDATION_STARTED, VALIDATED_FLAGGED].reduce(
      foldSpecPhase,
      INITIAL_SPEC_PHASE_STATE
    );
    expect(state.stage).toBe("validated");
    expect(state.verdict).toBe("flagged");
    expect(state.flaggedLeaves).toEqual(["L2", "L7"]);
  });

  it("carries the full accumulated state after the whole sequence", () => {
    const sequence = [GENERATION_STARTED, GENERATED, VALIDATION_STARTED, VALIDATED_FLAGGED];
    const final = sequence.reduce(foldSpecPhase, INITIAL_SPEC_PHASE_STATE);
    expect(final).toEqual({
      stage: "validated",
      leafCount: 12,
      validatorModel: "grok-4.3",
      verdict: "flagged",
      flaggedLeaves: ["L2", "L7"],
    });
  });

  it("ignores unrelated events — state unchanged", () => {
    const state = foldSpecPhase(INITIAL_SPEC_PHASE_STATE, GENERATION_STARTED);
    const next = foldSpecPhase(state, UNRELATED);
    expect(next).toEqual(state);
  });

  it("ignores an unrelated event even before any spec-phase event has fired", () => {
    const next = foldSpecPhase(INITIAL_SPEC_PHASE_STATE, UNRELATED);
    expect(next).toEqual(INITIAL_SPEC_PHASE_STATE);
  });

  it("does not mutate the input state object", () => {
    const state: SpecPhaseState = { stage: "idle" };
    foldSpecPhase(state, GENERATION_STARTED);
    expect(state).toEqual({ stage: "idle" });
  });

  it("does not mutate the caller's flagged_leaves array", () => {
    const flaggedLeaves = ["L2", "L7"];
    const event: SpecValidatedEvent = {
      event: "spec_validated",
      timestamp: "2026-07-06T00:00:03Z",
      verdict: "flagged",
      flagged_leaves: flaggedLeaves,
    };
    const state = foldSpecPhase(INITIAL_SPEC_PHASE_STATE, event);
    flaggedLeaves.push("L99");
    expect(state.flaggedLeaves).toEqual(["L2", "L7"]);
  });
});

describe("shouldShowSpecStepper", () => {
  it("is false at idle (a non-autonomous run with no spec events) — no regression", () => {
    expect(shouldShowSpecStepper({ stage: "idle" }, 0)).toBe(false);
  });

  it("is true while generating with no iterations yet", () => {
    expect(shouldShowSpecStepper({ stage: "generating" }, 0)).toBe(true);
  });

  it("is true while generated with no iterations yet", () => {
    expect(shouldShowSpecStepper({ stage: "generated", leafCount: 9 }, 0)).toBe(true);
  });

  it("is true while validating with no iterations yet", () => {
    expect(shouldShowSpecStepper({ stage: "validating", validatorModel: "grok-4.3" }, 0)).toBe(
      true
    );
  });

  it("is false once validation completes (swap to the reasoning view)", () => {
    expect(shouldShowSpecStepper({ stage: "validated", verdict: "verified" }, 0)).toBe(false);
  });

  it("is false the moment the first reasoning iteration arrives, even mid-spec-phase", () => {
    // A repl_iteration landing before spec_validated (e.g. the lifecycle
    // driver starting work) must still swap to the reasoning log.
    expect(shouldShowSpecStepper({ stage: "generating" }, 1)).toBe(false);
    expect(shouldShowSpecStepper({ stage: "validating" }, 3)).toBe(false);
  });
});
