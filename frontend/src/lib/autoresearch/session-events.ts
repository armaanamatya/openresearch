/**
 * Pure reducer for the spec-generation + external-validation phase
 * (alphaXiv screen D) — the pre-loop stage between launch and the live
 * session view. Folds the 4 spec-phase SSE events (T7/T8) into a small
 * state machine consumed by `SpecValidationStepper`.
 *
 * No React, no side effects — mirrors the shape of
 * `hooks/use-rlm-run.ts`'s `fold()` but scoped to just this one phase; that
 * file's own tree-based `fold()` treats these 4 events as a documented
 * no-op passthrough (see `lib/events/rlm-events.ts`).
 */

import type { RlmDashboardEvent } from "@/lib/events/rlm-events";

export type SpecPhaseStage = "idle" | "generating" | "generated" | "validating" | "validated";

export interface SpecPhaseState {
  stage: SpecPhaseStage;
  /** From spec_generated — the rubric leaf count, once known. */
  leafCount?: number;
  /** From spec_validation_started — the external validator's model label. */
  validatorModel?: string;
  /** From spec_validated — free-form verdict string (e.g. "verified"/"flagged"). */
  verdict?: string;
  /** From spec_validated — rubric leaf ids the validator flagged (advisory only). */
  flaggedLeaves?: string[];
}

export const INITIAL_SPEC_PHASE_STATE: SpecPhaseState = { stage: "idle" };

/**
 * Folds one event into `SpecPhaseState`. Every event outside the 4
 * spec-phase types is ignored (returns `state` unchanged) — safe to reduce
 * directly over a full, mixed `RlmDashboardEvent[]` stream.
 */
export function foldSpecPhase(state: SpecPhaseState, event: RlmDashboardEvent): SpecPhaseState {
  switch (event.event) {
    case "spec_generation_started":
      return { ...state, stage: "generating" };
    case "spec_generated":
      return { ...state, stage: "generated", leafCount: event.leaf_count };
    case "spec_validation_started":
      return { ...state, stage: "validating", validatorModel: event.validator_model };
    case "spec_validated":
      return {
        ...state,
        stage: "validated",
        verdict: event.verdict,
        flaggedLeaves: [...event.flagged_leaves],
      };
    default:
      return state;
  }
}
