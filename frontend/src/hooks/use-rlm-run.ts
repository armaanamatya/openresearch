/**
 * useRlmRun — the pure reducer for RLM run state.
 *
 * Spec: docs/superpowers/specs/2026-05-21-rlm-phase4-frontend-design.md §5
 *
 * This file is extended by Task 4 (tree folding: candidate_proposed,
 * candidate_outcome, sub_rlm_spawned, sub_rlm_complete). Those events are
 * no-ops here but the TreeNode type is declared now per §6.
 */

import { useMemo } from "react";
import type {
  RlmDashboardEvent,
  ReplIterationEvent,
  PrimitiveCallEvent,
  RubricScoreEvent,
  RunCompleteEvent,
} from "../lib/events/rlm-events";

// ─── Member types ─────────────────────────────────────────────────────────────

/** Mirrors the repl_iteration payload verbatim — snake_case preserved. */
export interface IterationView {
  iteration: number;
  response: string;
  code_blocks: ReplIterationEvent["code_blocks"];
  sub_calls: number;
  timing: number | null;
}

export interface RubricArea {
  area: string;
  score: number;
  weight: number;
  status: "pass" | "partial" | "fail";
}

export interface PrimitiveCallView {
  primitive: string;
  status: "start" | "ok" | "error";
  args_summary: Record<string, unknown>;
  result_summary: string | null;
  iteration: number | null;
  rubric_delta: number | null;
  timestamp: string;
}

export interface SubRlmView {
  depth: number;
  model: string;
  prompt_preview: string;
  duration_ms: number | null;
  error: string | null;
  spawnedAt: string;
  completedAt: string | null;
}

/**
 * A node in the exploration tree — §6.
 * Populated by Task 4 (candidate_proposed, candidate_outcome,
 * sub_rlm_spawned, sub_rlm_complete). Declared here so types are available
 * across tasks.
 */
export interface TreeNode {
  id: string;
  kind: "paper" | "work" | "baseline" | "candidate" | "subrlm" | "declined-group";
  parentId: string | null;
  /** Iteration range this node covers. */
  iterationRange: [number, number];
  /** Only present on candidate nodes. */
  outcome?: "running" | "promoted" | "marginal" | "failed" | "skipped" | "declined";
  /** Score change from applying this candidate, if measured. */
  rubricDelta?: number | null;
  /** Candidate metadata (for kind==="candidate"). */
  candidate?: {
    id: string;
    title: string;
    category: string;
    description: string;
    reasoning: string;
  };
  /** Rubric score attached at rubric_score events (for baseline / work nodes). */
  rubricScore?: number | null;
}

// ─── RlmRunState ──────────────────────────────────────────────────────────────

export interface RlmRunState {
  status: "queued" | "running" | "completed" | "partial" | "failed";
  iterationCount: number;

  rubric: {
    current: number | null;
    baseline: number | null; // first rubric_score
    target: number | null;
    series: Array<{ iteration: number; score: number }>; // sparkline / timeline
    areas: RubricArea[]; // latest breakdown
  };

  variables: Record<
    string,
    { type: string; size: number; firstSeenIteration: number }
  >;

  /** Dense array: one entry per repl_iteration, in stream order. Length === iterationCount. */
  iterations: IterationView[];
  currentIteration: IterationView | null;

  tree: TreeNode[];
  primitiveCalls: PrimitiveCallView[];
  subRlms: SubRlmView[];

  report: {
    finalReportPath: string | null;
    costUsd: number | null;
    counts: {
      iterations: number;
      primitiveCalls: number;
      /** Populated by Task 4 (candidate_proposed). Zero here. */
      proposed: number;
      /** Populated by Task 4 (candidate_outcome). Zero here. */
      promoted: number;
    };
  } | null;
}

// ─── Initial state ────────────────────────────────────────────────────────────

export const INITIAL_RLM_STATE: RlmRunState = {
  status: "queued",
  iterationCount: 0,
  rubric: {
    current: null,
    baseline: null,
    target: null,
    series: [],
    areas: [],
  },
  variables: {},
  iterations: [],
  currentIteration: null,
  tree: [],
  primitiveCalls: [],
  subRlms: [],
  report: null,
};

// ─── Event handlers (pure) ────────────────────────────────────────────────────

function foldReplIteration(
  state: RlmRunState,
  ev: ReplIterationEvent
): RlmRunState {
  const view: IterationView = {
    iteration: ev.iteration,
    response: ev.response,
    code_blocks: ev.code_blocks,
    sub_calls: ev.sub_calls,
    timing: ev.timing,
  };

  // Dense array: push in stream order so iterations.length === iterationCount.
  const iterations = [...state.iterations, view];

  // Merge vars from all code blocks — first seen wins for firstSeenIteration.
  const variables = { ...state.variables };
  for (const block of ev.code_blocks) {
    for (const [name, meta] of Object.entries(block.vars)) {
      if (!(name in variables)) {
        variables[name] = {
          type: meta.type,
          size: meta.size,
          firstSeenIteration: ev.iteration,
        };
      } else {
        // Update size to the latest value but preserve firstSeenIteration.
        variables[name] = {
          ...variables[name],
          type: meta.type,
          size: meta.size,
        };
      }
    }
  }

  return {
    ...state,
    status: state.status === "queued" ? "running" : state.status,
    iterationCount: iterations.length,
    iterations,
    currentIteration: view,
    variables,
  };
}

function foldPrimitiveCall(
  state: RlmRunState,
  ev: PrimitiveCallEvent
): RlmRunState {
  const call: PrimitiveCallView = {
    primitive: ev.primitive,
    status: ev.status,
    args_summary: ev.args_summary,
    result_summary: ev.result_summary,
    iteration: ev.iteration,
    rubric_delta: ev.rubric_delta,
    timestamp: ev.timestamp,
  };
  return {
    ...state,
    primitiveCalls: [...state.primitiveCalls, call],
  };
}

function foldRubricScore(
  state: RlmRunState,
  ev: RubricScoreEvent
): RlmRunState {
  const isFirst = state.rubric.baseline === null;
  return {
    ...state,
    rubric: {
      baseline: isFirst ? ev.score : state.rubric.baseline,
      current: ev.score,
      target: ev.target,
      series: [...state.rubric.series, { iteration: ev.iteration, score: ev.score }],
      areas: ev.areas.map((a) => ({ ...a })),
    },
  };
}

function foldRunComplete(
  state: RlmRunState,
  ev: RunCompleteEvent
): RlmRunState {
  return {
    ...state,
    status: ev.status,
    report: {
      finalReportPath: ev.final_report_path,
      costUsd: ev.cost_usd,
      counts: {
        iterations: state.iterationCount,
        primitiveCalls: state.primitiveCalls.length,
        // proposed / promoted are populated by Task 4 (tree events).
        // They remain 0 here; the test pin states this is acceptable.
        proposed: 0,
        promoted: 0,
      },
    },
  };
}

// ─── Pure fold ────────────────────────────────────────────────────────────────

/**
 * Pure reducer — returns a new RlmRunState. Never mutates its inputs.
 *
 * Tree events (candidate_proposed, candidate_outcome, sub_rlm_spawned,
 * sub_rlm_complete) are no-ops here; Task 4 extends this function.
 */
export function fold(state: RlmRunState, event: RlmDashboardEvent): RlmRunState {
  switch (event.event) {
    case "repl_iteration":
      return foldReplIteration(state, event);
    case "primitive_call":
      return foldPrimitiveCall(state, event);
    case "rubric_score":
      return foldRubricScore(state, event);
    case "run_complete":
      return foldRunComplete(state, event);
    // Tree events — no-ops in Task 3, implemented in Task 4.
    case "candidate_proposed":
    case "candidate_outcome":
    case "sub_rlm_spawned":
    case "sub_rlm_complete":
      return state;
    default:
      // Exhaustiveness guard; unknown events return state unchanged.
      return state;
  }
}

// ─── React hook ───────────────────────────────────────────────────────────────

/**
 * useRlmRun(events) — folds the event array into RlmRunState.
 *
 * Pure — only re-computes when the events array reference changes.
 * The caller (WorkflowView or replay harness) owns the event accumulation.
 */
export function useRlmRun(events: RlmDashboardEvent[]): RlmRunState {
  return useMemo(() => events.reduce(fold, INITIAL_RLM_STATE), [events]);
}
