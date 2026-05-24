"use client";

import { useMemo } from "react";
import type { PrimitiveCallView, RlmRunState } from "../../../hooks/use-rlm-run";
import styles from "./pipeline-phase-strip.module.css";

interface PipelinePhaseStripProps {
  status: RlmRunState["status"];
  primitiveCalls: PrimitiveCallView[];
}

type PhaseId = "ingest" | "understand" | "plan" | "implement" | "run" | "verify";

interface Phase {
  id: PhaseId;
  label: string;
  title: string;
}

const PHASES: Phase[] = [
  {
    id: "ingest",
    label: "Ingest",
    title: "Resolve the paper, parse text, and prepare the project workspace.",
  },
  {
    id: "understand",
    label: "Understand",
    title: "Read paper sections, extract claims, and gather hyperparameters.",
  },
  {
    id: "plan",
    label: "Plan",
    title: "Detect environment, build an image, and plan the reproduction.",
  },
  {
    id: "implement",
    label: "Implement",
    title: "Write or repair the baseline code for the paper reproduction.",
  },
  {
    id: "run",
    label: "Run",
    title: "Execute the generated commands. RunPod pods are created lazily here.",
  },
  {
    id: "verify",
    label: "Verify",
    title: "Score the run against the rubric and write final_report.json.",
  },
];

const PHASE_INDEX = new Map(PHASES.map((phase, index) => [phase.id, index]));

function phaseForPrimitive(primitive: string): PhaseId {
  switch (primitive) {
    case "understand_section":
    case "extract_hyperparameters":
      return "understand";
    case "detect_environment":
    case "build_environment":
    case "plan_reproduction":
      return "plan";
    case "implement_baseline":
      return "implement";
    case "run_experiment":
      return "run";
    case "verify_against_rubric":
    case "propose_improvements":
    case "record_candidate_outcome":
      return "verify";
    default:
      return "ingest";
  }
}

function deriveCurrentPhase(status: RlmRunState["status"], primitiveCalls: PrimitiveCallView[]) {
  if (status === "completed" || status === "partial" || status === "failed") {
    return { current: "verify" as PhaseId, hasError: status === "failed" };
  }
  const latest = primitiveCalls[primitiveCalls.length - 1] ?? null;
  if (!latest) return { current: "ingest" as PhaseId, hasError: false };
  return {
    current: phaseForPrimitive(latest.primitive),
    hasError: latest.status === "error",
  };
}

export function PipelinePhaseStrip({ status, primitiveCalls }: PipelinePhaseStripProps) {
  const { current, hasError } = useMemo(
    () => deriveCurrentPhase(status, primitiveCalls),
    [status, primitiveCalls],
  );
  const currentIndex = PHASE_INDEX.get(current) ?? 0;

  return (
    <nav className={styles.strip} aria-label="RLM pipeline phase">
      {PHASES.map((phase, index) => {
        const state =
          index < currentIndex
            ? "done"
            : index === currentIndex
            ? hasError
              ? "error"
              : "current"
            : "pending";
        const className = [
          styles.step,
          state === "done" ? styles.done : "",
          state === "current" ? styles.current : "",
          state === "error" ? styles.error : "",
        ]
          .filter(Boolean)
          .join(" ");
        return (
          <span
            key={phase.id}
            className={className}
            aria-current={state === "current" || state === "error" ? "step" : undefined}
            title={`${phase.label}: ${phase.title}`}
          >
            <span className={styles.dot} aria-hidden="true" />
            {phase.label}
          </span>
        );
      })}
    </nav>
  );
}
