"use client";

import { useEffect, useMemo, useRef } from "react";
import { useRouter } from "next/navigation";

import type { RlmDashboardEvent } from "@/lib/events/rlm-events";
import {
  foldSpecPhase,
  INITIAL_SPEC_PHASE_STATE,
  type SpecPhaseStage,
} from "@/lib/autoresearch/session-events";
import { Stepper, type StepperStep, type StepStatus } from "./ui/Stepper";
import styles from "./SpecValidationStepper.module.css";

export interface SpecValidationStepperProps {
  /** The SSE stream slice to fold. Only the 4 spec-phase events affect the
   * derived state — every other event type is ignored, so the caller may
   * pass the run's full mixed event array unfiltered. */
  events: RlmDashboardEvent[];
  /** Destination run id — the auto-redirect target once validation completes. */
  runId: string;
  className?: string;
}

function generatingStatus(stage: SpecPhaseStage): StepStatus {
  if (stage === "idle") return "pending";
  if (stage === "generating") return "active";
  return "done"; // generated | validating | validated
}

function validatingStatus(stage: SpecPhaseStage): StepStatus {
  if (stage === "validated") return "done";
  if (stage === "validating") return "active";
  return "pending"; // idle | generating | generated
}

/**
 * Spec-validation stepper (alphaXiv screen D) — the visible "Analyzing →
 * Generating spec → Validating with <model> → done" phase shown between
 * launch and the live session view. Folds the 4 spec-phase SSE events
 * (T7/T8) via `foldSpecPhase`, drives the T9b `Stepper`, and auto-navigates
 * to `/sessions/<runId>` once `spec_validated` lands. The validator's
 * verdict is advisory only — a "flagged" verdict shows a non-blocking note
 * but never withholds the redirect (mirrors the backend hook, which
 * annotates/drops rubric leaves but never halts the run).
 *
 * Renders inside a `.autoresearch`-wrapped ancestor (see
 * src/styles/autoresearch-tokens.css) — this component does not import
 * the token file itself.
 */
export function SpecValidationStepper({ events, runId, className }: SpecValidationStepperProps) {
  const router = useRouter();
  const specPhase = useMemo(
    () => events.reduce(foldSpecPhase, INITIAL_SPEC_PHASE_STATE),
    [events]
  );

  // Fire the redirect exactly once — a ref (not just the stage check alone)
  // guards against a re-render on the same terminal state re-triggering it.
  const hasNavigatedRef = useRef(false);
  useEffect(() => {
    if (specPhase.stage === "validated" && !hasNavigatedRef.current) {
      hasNavigatedRef.current = true;
      router.push(`/sessions/${encodeURIComponent(runId)}`);
    }
  }, [specPhase.stage, runId, router]);

  const flaggedCount = specPhase.flaggedLeaves?.length ?? 0;
  const showFlaggedNote = specPhase.stage === "validated" && flaggedCount > 0;

  const steps: StepperStep[] = [
    {
      label:
        specPhase.leafCount != null
          ? `Generating reproduction spec — ${specPhase.leafCount} leaves`
          : "Generating reproduction spec",
      status: generatingStatus(specPhase.stage),
    },
    {
      label: specPhase.validatorModel
        ? `Validating spec with ${specPhase.validatorModel}`
        : "Validating spec",
      status: validatingStatus(specPhase.stage),
    },
  ];

  return (
    <div className={[styles.wrapper, className].filter(Boolean).join(" ")}>
      <Stepper steps={steps} />
      {showFlaggedNote && (
        <p className={styles.flaggedNote}>
          {flaggedCount} leaf{flaggedCount === 1 ? "" : "es"} flagged for review
        </p>
      )}
    </div>
  );
}
