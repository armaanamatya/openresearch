/**
 * useRlmRun — the pure reducer for RLM run state.
 *
 * Spec: docs/superpowers/specs/2026-05-21-rlm-phase4-frontend-design.md §5 / §6
 *
 * Folds the 8 RLM dashboard events into `RlmRunState`. The four tree events
 * (candidate_proposed, candidate_outcome, sub_rlm_spawned, sub_rlm_complete)
 * build `RlmRunState.tree` — a flat node list, each node carrying `parentId`.
 */

import { useMemo, useRef, useState, useCallback } from "react";
import type {
  RlmDashboardEvent,
  ReplIterationEvent,
  PrimitiveCallEvent,
  RubricScoreEvent,
  RunCompleteEvent,
  CandidateProposedEvent,
  CandidateOutcomeEvent,
  SubRlmSpawnedEvent,
  SubRlmCompleteEvent,
  RunWarningEvent,
  IterationHeartbeatEvent,
  GpuResolvedEvent,
  ExperimentCompletedEvent,
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

/**
 * A single rubric leaf — the finest-grained criterion the grader inspects.
 * Populated from `event.areas[i].leaves` when the backend enriches the
 * rubric_score event; absent on older/live events (degrade gracefully).
 */
export interface LeafDetail {
  id: string;
  label: string;
  /** Leaf score 0–1, or `null` when `status === "unavailable"`. */
  score: number | null;
  status: "pass" | "partial" | "fail" | "unavailable";
  /** Justification for the leaf's score (grader/model text). */
  why: string;
}

export interface RubricArea {
  area: string;
  score: number;
  weight: number;
  status: "pass" | "partial" | "fail";
  /** Per-leaf breakdown — present only on enriched rubric_score events. */
  leaves?: LeafDetail[];
}

/** A weakest-leaf digest entry — the grader's "what to fix next" hint. */
export interface WeakLeaf {
  id: string;
  /** Leaf score 0–1, or `null` when the leaf had no numeric score. */
  score: number | null;
  why: string;
  area: string;
}

/** A recent execution error surfaced near the rubric score. */
export interface RecentError {
  kind: string;
  message: string;
  iteration: number;
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

/** A warning emitted by the backend watchdog (e.g. SDK aclose deadlock). */
export interface RunWarning {
  level: RunWarningEvent["level"];
  code: string;
  message: string;
  timestamp: string;
}

/**
 * GpuPlan — the resolved GPU provisioning plan, populated when a gpu_resolved
 * SSE event is received.  Mirrors the backend GpuPlan Pydantic model fields
 * that are relevant to the UI badge (§4.5, dynamic-gpu-selection 2026-05-23).
 */
export interface GpuPlan {
  runpod_id: string;
  short_name: string;
  vram_gb: number;
  gpu_count: number;
  cloud_type: string;
  sku_usd_per_hr: number;
  total_usd_per_hr: number;
  container_disk_gb: number;
  volume_gb: number;
  source: "paper" | "fallback" | "catalog";
  requirements: {
    estimated_vram_gb: number | null;
    paper_gpu_string: string | null;
    paper_gpu_count: number | null;
    reasoning: string;
    confidence: number;
  };
  ladder_remaining: number;
  resolved_at: string;
}

/** Display-annotation phase a trunk (`work`) node belongs to — §6. Not load-bearing. */
export type TreePhase = "comprehension" | "environment" | "baseline-build";

/** Phase label for primitive nodes — maps the primitive name to a lifecycle phase. */
export type PrimitivePhase =
  | "ingest"
  | "understand"
  | "environment"
  | "plan"
  | "implement"
  | "run"
  | "verify"
  | "improve";

/**
 * A node in the exploration tree — §6.
 * Built by Task 4 (candidate_proposed, candidate_outcome, sub_rlm_spawned,
 * sub_rlm_complete) plus the trunk (repl_iteration / primitive_call / rubric_score).
 */
export interface TreeNode {
  id: string;
  kind:
    | "paper"
    | "work"
    | "baseline"
    | "candidate"
    | "subrlm"
    | "declined-group"
    | "primitive"
    | "llm_primitive";
  parentId: string | null;
  /** Display title — uniform across node kinds. */
  title: string;
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
    displayTitle?: string;
    category: string;
    description: string;
    reasoning: string;
  };
  /** propose_improvements round this node belongs to (candidate / declined-group). */
  round?: number;
  /** Number of declined candidates folded into a declined-group node. */
  declinedCount?: number;
  /** Phase label of a trunk (`work`) node — a §6 display annotation. */
  phase?: TreePhase;
  /** Phase label for primitive/llm_primitive nodes. */
  primitivePhase?: PrimitivePhase | null;
  /** Rubric score attached at rubric_score events (for baseline / work nodes). */
  rubricScore?: number | null;
  /** The raw primitive name (for primitive/llm_primitive nodes). */
  primitiveName?: string;
}

// ─── RlmRunState ──────────────────────────────────────────────────────────────

export interface RlmRunState {
  status: "queued" | "running" | "completed" | "partial" | "failed";
  iterationCount: number;

  rubric: {
    current: number | null;
    baseline: number | null; // first rubric_score
    /** The HIGHEST score over the whole run (max of `series`), or null before
     * any rubric_score. The honest headline number: a failed final attempt
     * should not erase the best result the run actually achieved. Computed in
     * the reducer from `series`, so it works on existing data with no backend
     * change. OPTIONAL on the type so pre-existing rubric literals (e.g. test
     * fixtures) stay valid; the reducer + INITIAL_RLM_STATE always set it. */
    best?: number | null;
    target: number | null;
    series: Array<{ iteration: number; score: number }>; // sparkline / timeline
    areas: RubricArea[]; // latest breakdown
    // --- Phase-4-forward-compat (spec 2026-05-23-rubric-climb-leaderboard §4.2)
    /** Snapshot of `areas` BEFORE the most recent rubric_score event. Empty
     * before the second rubric_score lands. Used by RubricStrip to highlight
     * areas whose status rank just increased (fail -> partial -> pass). */
    previousAreas: RubricArea[];
    /** The most-recent candidate the root was working on. Set by
     * candidate_proposed; outcome updated by candidate_outcome on id-match.
     * Sticky: a long primitive-only stretch keeps the prior candidate live
     * until a new candidate_proposed overwrites it. */
    attributableCandidate: {
      id: string;
      title: string;
      outcome: TreeNode["outcome"] | null;
    } | null;
    /**
     * The grader's weakest-leaf digest from the latest enriched rubric_score
     * event, or `undefined` when the event carried no `weak_leaves`. Used by
     * RubricDetail to surface "what to fix next".
     */
    weakLeaves?: WeakLeaf[];
    /**
     * Recent execution errors for the breakdown's "Recent errors" section.
     * Prefer the backend-enriched `recent_errors` when present; otherwise this
     * is back-filled from the EXISTING stream (run_warning failures +
     * primitive_call errors) so specific failures show even before the
     * enrichment ships. Newest last; capped at the last few. Empty array when
     * nothing has failed (honest empty-state). OPTIONAL on the type so
     * pre-existing rubric literals stay valid; the reducer + INITIAL_RLM_STATE
     * always set it (consumers treat `undefined` as "no errors").
     */
    recentErrors?: RecentError[];
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

  /**
   * Internal reducer bookkeeping — candidate_outcome events that arrived before
   * their matching candidate_proposed (§5.3 out-of-order tolerance). Keyed by
   * candidate id; drained when the proposal lands. Not part of the rendered UI.
   */
  _pendingOutcomes: Record<
    string,
    { outcome: NonNullable<TreeNode["outcome"]>; rubricDelta: number | null; iteration: number }
  >;

  report: {
    finalReportPath: string | null;
    costUsd: number | null;
    counts: {
      iterations: number;
      primitiveCalls: number;
      proposed: number;
      promoted: number;
    };
  } | null;

  /** Watchdog warnings accumulated during the run. Newest last. */
  warnings: RunWarning[];

  /**
   * ISO-8601 timestamp of the most recent ``iteration_heartbeat`` event, or
   * ``null`` if no heartbeat has arrived yet.  The UI derives "wedged" status
   * when ``status === "running"`` and ``Date.now() - new Date(lastHeartbeatAt) > 60_000``.
   */
  lastHeartbeatAt: string | null;

  /**
   * The most-recent gpu_resolved payload for this run, or null before the
   * resolve_gpu_requirements primitive completes.  Consumed by the
   * NodeDetailSidebar GpuPlan badge (dynamic-gpu-selection 2026-05-23).
   */
  gpuPlan: GpuPlan | null;

  /**
   * Per-model metrics from the latest experiment_completed event, or null when
   * no experiment has completed or the paper only tests a single model variant.
   * Keyed by Python-identifier-safe model short name (e.g. "qwen2_5_3b").
   * Values are flat metric dicts (metric name → number).
   * Lane γ: per-model multi-model UI (2026-05-23).
   */
  perModelMetrics: Record<string, Record<string, number>> | null;
}

// ─── Initial state ────────────────────────────────────────────────────────────

export const INITIAL_RLM_STATE: RlmRunState = {
  status: "queued",
  iterationCount: 0,
  rubric: {
    current: null,
    baseline: null,
    best: null,
    target: null,
    series: [],
    areas: [],
    previousAreas: [],
    attributableCandidate: null,
    weakLeaves: undefined,
    recentErrors: [],
  },
  variables: {},
  iterations: [],
  currentIteration: null,
  tree: [],
  primitiveCalls: [],
  subRlms: [],
  _pendingOutcomes: {},
  report: null,
  warnings: [],
  lastHeartbeatAt: null,
  gpuPlan: null,
  perModelMetrics: null,
};

// ─── Tree helpers (pure) ──────────────────────────────────────────────────────

const PAPER_NODE_ID = "paper";

/**
 * Primitives that invoke the LLM client — rendered as pulsing llm_primitive
 * nodes in the constellation view. All others become plain primitive nodes.
 * Pure file-I/O / no-op primitives are excluded entirely from visualization
 * (see VISUALIZED_PRIMITIVES below).
 */
const LLM_USING_PRIMITIVES = new Set([
  "understand_section",
  "extract_hyperparameters",
  "propose_improvements",
  "recommend_next_tool",
  "verify_against_rubric",
  "plan_reproduction",
  "detect_environment",
]);

/**
 * Primitives that should NOT produce a node in the constellation.
 * These are pure file I/O or heartbeat-style calls with no meaningful viz value.
 */
const NON_VISUALIZED_PRIMITIVES = new Set([
  "check_user_messages",
  "respond_to_user",
  "record_candidate_outcome",
  "iteration_heartbeat",
]);

/** How many recent errors to retain for the rubric breakdown's error section. */
const MAX_RECENT_ERRORS = 3;

/**
 * Append an error to the bounded recentErrors ring, keeping only the most
 * recent MAX_RECENT_ERRORS. De-dupes on (kind, message) against the current
 * tail so a primitive that retries the same failure doesn't spam the list.
 * Pure: returns a new array.
 */
function pushRecentError(
  errors: RecentError[] | undefined,
  next: RecentError
): RecentError[] {
  const base = errors ?? [];
  const tail = base[base.length - 1];
  if (tail && tail.kind === next.kind && tail.message === next.message) {
    return base;
  }
  const appended = [...base, next];
  return appended.length > MAX_RECENT_ERRORS
    ? appended.slice(appended.length - MAX_RECENT_ERRORS)
    : appended;
}

/**
 * run_warning codes that denote an actual failure (vs. a benign notice).
 * A code matches when it contains one of these substrings (case-insensitive).
 * Kept broad so new failure codes surface without a code change; the
 * forced_iteration warning is explicitly EXCLUDED — it is a policy notice
 * ("do more work"), not an execution failure.
 */
const FAILURE_WARNING_SUBSTRINGS = ["fail", "error", "oom", "deadlock", "crash", "timeout"];

function isFailureWarning(ev: RunWarningEvent): boolean {
  if (ev.level === "error") return true;
  const code = (ev.code || "").toLowerCase();
  if (code.includes("forced_iteration")) return false;
  return FAILURE_WARNING_SUBSTRINGS.some((s) => code.includes(s));
}

/**
 * Map a primitive name to its PrimitivePhase for the constellation view.
 * Returns null for primitives not explicitly mapped (they still get a node,
 * just without a phase label).
 */
function primitivePhaseFor(primitive: string): PrimitivePhase | null {
  switch (primitive) {
    case "understand_section":
    case "extract_hyperparameters":
      return "understand";
    case "detect_environment":
    case "build_environment":
      return "environment";
    case "plan_reproduction":
      return "plan";
    case "implement_baseline":
      return "implement";
    case "run_experiment":
      return "run";
    case "verify_against_rubric":
      return "verify";
    case "propose_improvements":
    case "recommend_next_tool":
      return "improve";
    default:
      return null;
  }
}

/** User-friendly label for a primitive name. */
function friendlyPrimitiveTitle(primitive: string): string {
  switch (primitive) {
    case "understand_section": return "Understand";
    case "extract_hyperparameters": return "Extract Hyperparams";
    case "detect_environment": return "Detect Env";
    case "build_environment": return "Build Env";
    case "plan_reproduction": return "Plan";
    case "implement_baseline": return "Implement";
    case "run_experiment": return "Run Experiment";
    case "verify_against_rubric": return "Verify";
    case "propose_improvements": return "Propose";
    case "recommend_next_tool": return "Recommend";
    default: return primitive.replace(/_/g, " ");
  }
}

/** §6 phase derivation — map a primitive name to its trunk phase, or null. */
function phaseFor(primitive: string): TreePhase | null {
  switch (primitive) {
    case "understand_section":
    case "extract_hyperparameters":
      return "comprehension";
    case "detect_environment":
    case "build_environment":
      return "environment";
    case "implement_baseline":
    case "run_experiment":
      return "baseline-build";
    default:
      // verify_against_rubric / propose_improvements / unknown — not a trunk phase.
      return null;
  }
}

/** Title-cased label for each trunk phase. */
function titleForPhase(phase: TreePhase): string {
  switch (phase) {
    case "comprehension":
      return "Comprehension";
    case "environment":
      return "Environment";
    case "baseline-build":
      return "Baseline build";
  }
}

/** Ensure the `paper` root exists. Returns the same array if it already does. */
function ensurePaper(tree: TreeNode[]): TreeNode[] {
  if (tree.some((n) => n.kind === "paper")) return tree;
  const paper: TreeNode = {
    id: PAPER_NODE_ID,
    kind: "paper",
    parentId: null,
    title: "Paper",
    iterationRange: [0, 0],
  };
  return [paper, ...tree];
}

/** The last `work` (trunk) node, or undefined. Trunk nodes are appended in order. */
function lastWorkNode(tree: TreeNode[]): TreeNode | undefined {
  for (let i = tree.length - 1; i >= 0; i--) {
    if (tree[i].kind === "work") return tree[i];
  }
  return undefined;
}

/**
 * The frontier parent for a candidate / sub-RLM node when no explicit
 * `parent_id` is given — §5.3:
 *   (a) the most-recent node with `outcome === "promoted"`;
 *   (b) else the previous fan's parent;
 *   (c) else the `baseline` node.
 * The most-recent scan is reverse insertion order (latest insert = array end).
 */
function frontierParent(tree: TreeNode[]): string {
  // (a) most-recent promoted candidate
  for (let i = tree.length - 1; i >= 0; i--) {
    const n = tree[i];
    if (n.kind === "candidate" && n.outcome === "promoted") return n.id;
  }
  // (b) the previous fan's parent — the parentId of the most-recent candidate
  for (let i = tree.length - 1; i >= 0; i--) {
    const n = tree[i];
    if (n.kind === "candidate" && n.parentId != null) return n.parentId;
  }
  // (c) the baseline node
  const baseline = tree.find((n) => n.kind === "baseline");
  if (baseline) return baseline.id;
  // Last-resort fallback so a node is never orphaned (e.g. a fan before any
  // rubric_score). The paper root always exists once any event has folded.
  return PAPER_NODE_ID;
}

// ─── Event handlers (pure) ────────────────────────────────────────────────────

function foldReplIteration(
  state: RlmRunState,
  ev: ReplIterationEvent
): RlmRunState {
  const view: IterationView = {
    iteration: ev.iteration,
    response: ev.response,
    // Copies the array; block objects (vars/stdout_meta/stderr_meta) remain shared
    // with the input event, which is safe because events are treated as immutable.
    code_blocks: [...ev.code_blocks],
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

  // §6 trunk: a primitive drives phase derivation + trunk-node iteration-ranges.
  // When the phase changes from the last work node, open a new work node;
  // otherwise extend the current one. verify_against_rubric / propose_improvements
  // map to no phase — they leave the trunk untouched.
  let tree = state.tree;
  const phase = phaseFor(ev.primitive);
  if (phase !== null) {
    const current = lastWorkNode(tree);
    if (!current || current.phase !== phase) {
      // Open a new trunk node, parented on the previous work node (chain) or paper.
      const parentId = current ? current.id : PAPER_NODE_ID;
      const iter = ev.iteration ?? 0;
      const node: TreeNode = {
        id: `work-${phase}-${tree.filter((n) => n.kind === "work").length + 1}`,
        kind: "work",
        parentId,
        title: titleForPhase(phase),
        iterationRange: [iter, iter],
        phase,
      };
      tree = [...tree, node];
    } else if (ev.iteration != null) {
      // Extend the current trunk node's iteration range.
      const [lo, hi] = current.iterationRange;
      tree = tree.map((n) =>
        n === current
          ? {
              ...n,
              iterationRange: [Math.min(lo, ev.iteration!), Math.max(hi, ev.iteration!)],
            }
          : n
      );
    }
  }

  // Add a constellation node for completed, visualizable primitives.
  if (ev.status === "ok" && !NON_VISUALIZED_PRIMITIVES.has(ev.primitive)) {
    const isLlm = LLM_USING_PRIMITIVES.has(ev.primitive);
    const kind: TreeNode["kind"] = isLlm ? "llm_primitive" : "primitive";
    // Parent = the current work node if it exists, else paper.
    const parentId = lastWorkNode(tree)?.id ?? PAPER_NODE_ID;
    const primitiveNode: TreeNode = {
      id: `prim-${ev.primitive}-${state.primitiveCalls.length + 1}`,
      kind,
      parentId,
      title: friendlyPrimitiveTitle(ev.primitive),
      iterationRange: [ev.iteration ?? 0, ev.iteration ?? 0],
      primitiveName: ev.primitive,
      primitivePhase: primitivePhaseFor(ev.primitive),
    };
    tree = [...tree, primitiveNode];
  }

  // Fallback enrichment: a primitive that errored is a concrete failure signal.
  // Fold it into recentErrors so the rubric breakdown can name the specific
  // failure (e.g. "run_experiment: ModuleNotFoundError ...") before the backend
  // enriches rubric_score. The result_summary is value-free metadata already.
  let rubric = state.rubric;
  if (ev.status === "error") {
    const recentErrors = pushRecentError(state.rubric.recentErrors, {
      kind: ev.primitive,
      message: ev.result_summary?.trim() || "primitive raised an exception",
      iteration: ev.iteration ?? state.iterationCount,
    });
    if (recentErrors !== state.rubric.recentErrors) {
      rubric = { ...state.rubric, recentErrors };
    }
  }

  return {
    ...state,
    status: state.status === "queued" ? "running" : state.status,
    tree,
    primitiveCalls: [...state.primitiveCalls, call],
    rubric,
  };
}

function foldRubricScore(
  state: RlmRunState,
  ev: RubricScoreEvent
): RlmRunState {
  const isFirst = state.rubric.baseline === null;

  // §6: the `baseline` node is created at the first rubric_score event.
  let tree = state.tree;
  if (isFirst && !tree.some((n) => n.kind === "baseline")) {
    const parentId = lastWorkNode(tree)?.id ?? PAPER_NODE_ID;
    const baseline: TreeNode = {
      id: "baseline",
      kind: "baseline",
      parentId,
      title: "Baseline",
      iterationRange: [ev.iteration, ev.iteration],
      rubricScore: ev.score,
    };
    tree = [...tree, baseline];
  } else {
    // Attach the score to the most-recently-inserted candidate node (even if that
    // candidate has already resolved), falling back to the baseline node. This
    // records the rubric reading taken while that node was live.
    let frontierIdx = -1;
    for (let i = tree.length - 1; i >= 0; i--) {
      if (tree[i].kind === "candidate") {
        frontierIdx = i;
        break;
      }
    }
    if (frontierIdx === -1) {
      frontierIdx = tree.findIndex((n) => n.kind === "baseline");
    }
    if (frontierIdx !== -1) {
      tree = tree.map((n, i) =>
        i === frontierIdx ? { ...n, rubricScore: ev.score } : n
      );
    }
  }

  const series = [...state.rubric.series, { iteration: ev.iteration, score: ev.score }];
  // best-of-run: the honest headline. max over the full trajectory so a failed
  // final attempt reads as a transient dip, not a regression.
  const best = series.reduce((m, p) => Math.max(m, p.score), ev.score);

  // Per-leaf detail rides through when the backend enriched the event; older
  // events have areas[i].leaves === undefined, which we preserve as-is.
  const areas: RubricArea[] = ev.areas.map((a) => ({
    area: a.area,
    score: a.score,
    weight: a.weight,
    status: a.status,
    leaves: a.leaves ? a.leaves.map((l) => ({ ...l })) : undefined,
  }));

  // weakLeaves: prefer the event's pre-computed digest. When absent, derive a
  // fallback from any leaves present on the areas (lowest-scoring first) so the
  // "what to fix" hint still works on partially-enriched events. Leave it
  // undefined when there is nothing to show.
  let weakLeaves: WeakLeaf[] | undefined;
  if (ev.weak_leaves && ev.weak_leaves.length > 0) {
    weakLeaves = ev.weak_leaves.map((w) => ({ ...w }));
  } else {
    const derived: WeakLeaf[] = [];
    for (const a of ev.areas) {
      for (const l of a.leaves ?? []) {
        if (l.status === "fail" || l.status === "partial") {
          derived.push({ id: l.id, score: l.score, why: l.why, area: a.area });
        }
      }
    }
    if (derived.length > 0) {
      // Lowest score first; treat a null score as the weakest (sorts to front).
      const rank = (s: number | null) => (s == null ? -Infinity : s);
      derived.sort((x, y) => rank(x.score) - rank(y.score));
      weakLeaves = derived.slice(0, 5);
    }
  }

  // recentErrors: when the event carries an explicit list, it WINS (the backend
  // knows best). Otherwise keep the stream-derived accumulator untouched (it is
  // back-filled from run_warning / primitive_call errors elsewhere).
  let recentErrors = state.rubric.recentErrors;
  if (ev.recent_errors && ev.recent_errors.length > 0) {
    recentErrors = ev.recent_errors
      .map((e) => ({ kind: e.kind, message: e.message, iteration: e.iteration }))
      .slice(-MAX_RECENT_ERRORS);
  }

  return {
    ...state,
    tree,
    rubric: {
      baseline: isFirst ? ev.score : state.rubric.baseline,
      current: ev.score,
      best,
      target: ev.target,
      series,
      // §4.2: snapshot the prior areas array BEFORE overwriting so the strip
      // can detect status flips (fail->partial->pass) at render time.
      previousAreas: state.rubric.areas.map((a) => ({ ...a })),
      areas,
      attributableCandidate: state.rubric.attributableCandidate,
      weakLeaves,
      recentErrors,
    },
  };
}

/**
 * Upsert the per-round `declined-group` node and bump its `declinedCount`.
 * Parent = the declined candidate's own parent so the group sits at the fan's
 * level. Returns a new tree array.
 */
function applyDeclinedGroup(
  tree: TreeNode[],
  round: number,
  candidateParentId: string | null,
  iteration: number
): TreeNode[] {
  const groupId = `declined-group-r${round}`;
  const existing = tree.find((n) => n.id === groupId);
  if (existing) {
    const newCount = (existing.declinedCount ?? 0) + 1;
    return tree.map((n) =>
      n === existing
        ? {
            ...n,
            declinedCount: newCount,
            title: `${newCount} declined`,
            iterationRange: [
              Math.min(n.iterationRange[0], iteration),
              Math.max(n.iterationRange[1], iteration),
            ],
          }
        : n
    );
  }
  const group: TreeNode = {
    id: groupId,
    kind: "declined-group",
    parentId: candidateParentId,
    title: "1 declined",
    iterationRange: [iteration, iteration],
    round,
    declinedCount: 1,
  };
  return [...tree, group];
}

function foldCandidateProposed(
  state: RlmRunState,
  ev: CandidateProposedEvent
): RlmRunState {
  // Two wire shapes arrive under event="candidate_proposed":
  //  - canonical RLM shape: { iteration, round, candidate: { id, title, … } }
  //    (sse_bridge.build_candidate_proposed_event)
  //  - flat BES shape: { candidate_id, cluster_id, score, failed } — NO nested
  //    candidate object, no iteration/round (rdr/controller.py + bes_rlm.py).
  // Normalize the flat shape into the canonical one; drop events carrying
  // neither (a malformed live event must never throw out of the reducer).
  const flat = ev as unknown as {
    candidate_id?: unknown;
    cluster_id?: unknown;
    failed?: unknown;
  };
  let candidate = ev.candidate;
  let flatFailed = false;
  if (!candidate || typeof candidate.id !== "string") {
    if (typeof flat.candidate_id !== "string" || flat.candidate_id === "") {
      return state;
    }
    flatFailed = flat.failed === true;
    candidate = {
      id: flat.candidate_id,
      title: flat.candidate_id,
      category:
        typeof flat.cluster_id === "string" ? flat.cluster_id : "candidate",
      description: "",
      reasoning: "",
    };
  }
  const iteration = typeof ev.iteration === "number" ? ev.iteration : 0;
  const round = typeof ev.round === "number" ? ev.round : 1;

  const parentId = ev.parent_id ?? frontierParent(state.tree);

  // A buffered outcome may have arrived first (§5.3 out-of-order).
  const pending = state._pendingOutcomes[candidate.id];

  const node: TreeNode = {
    id: `candidate-${candidate.id}`,
    kind: "candidate",
    parentId,
    title: candidate.display_title ?? candidate.title,
    iterationRange: [iteration, iteration],
    round,
    candidate: {
      id: candidate.id,
      title: candidate.title,
      displayTitle: candidate.display_title,
      category: candidate.category,
      description: candidate.description,
      reasoning: candidate.reasoning,
    },
    outcome: pending?.outcome ?? (flatFailed ? "failed" : undefined),
    rubricDelta: pending?.rubricDelta ?? null,
  };

  // Dedupe by node.id — the RLM root sometimes re-emits candidates with
  // non-globally-unique IDs across iterations (e.g. "path_1", "path_2").
  // Appending blindly produces React duplicate-key warnings AND duplicated
  // tree edges. If the candidate already exists, replace it in-place with the
  // latest payload (newest iteration wins).
  const existingIdx = state.tree.findIndex((n) => n.id === node.id);
  let tree: TreeNode[] = existingIdx === -1
    ? [...state.tree, node]
    : state.tree.map((n, i) => (i === existingIdx ? { ...n, ...node } : n));

  // If the buffered outcome was a declined, fold it into the round's group now.
  // Use the buffered outcome event's iteration so the declined-group iterationRange
  // reflects when the decline happened, not when the proposal was received.
  if (pending?.outcome === "declined") {
    tree = applyDeclinedGroup(tree, round, parentId, pending.iteration);
  }

  // Drain the pending entry once consumed.
  const _pendingOutcomes = { ...state._pendingOutcomes };
  delete _pendingOutcomes[candidate.id];

  // §4.2: update the live attributableCandidate pointer. If a buffered
  // outcome was drained above, its outcome rides through on the pointer.
  const rubric = {
    ...state.rubric,
    attributableCandidate: {
      id: candidate.id,
      title: candidate.title,
      outcome: pending?.outcome ?? null,
    },
  };

  return { ...state, tree, _pendingOutcomes, rubric };
}

function foldCandidateOutcome(
  state: RlmRunState,
  ev: CandidateOutcomeEvent
): RlmRunState {
  // Defensive: drop a malformed event (no candidate_id) instead of crashing.
  if (typeof ev.candidate_id !== "string" || ev.candidate_id === "") {
    return state;
  }
  // The BES selector emits outcome="selected" for the pool winner
  // (rdr/controller.py + bes_rlm.py) — render it as a promotion.
  const outcome =
    (ev.outcome as string) === "selected" ? "promoted" : ev.outcome;
  const rubricDelta = ev.rubric_delta ?? null;
  const iteration = typeof ev.iteration === "number" ? ev.iteration : 0;

  const nodeIdx = state.tree.findIndex(
    (n) => n.kind === "candidate" && n.candidate?.id === ev.candidate_id
  );

  // Out-of-order: the matching candidate_proposed has not arrived — buffer it.
  if (nodeIdx === -1) {
    return {
      ...state,
      _pendingOutcomes: {
        ...state._pendingOutcomes,
        [ev.candidate_id]: { outcome, rubricDelta, iteration },
      },
    };
  }

  const node = state.tree[nodeIdx];
  let tree = state.tree.map((n, i) =>
    i === nodeIdx
      ? { ...n, outcome, rubricDelta }
      : n
  );

  // A declined candidate also collapses into its round's declined-group node.
  if (outcome === "declined" && node.round != null) {
    tree = applyDeclinedGroup(tree, node.round, node.parentId, iteration);
  }

  // §4.2: if this outcome matches the live attributable candidate, update its
  // outcome on the pointer. Otherwise leave the pointer unchanged.
  const rubric =
    state.rubric.attributableCandidate?.id === ev.candidate_id
      ? {
          ...state.rubric,
          attributableCandidate: {
            ...state.rubric.attributableCandidate,
            outcome,
          },
        }
      : state.rubric;

  return { ...state, tree, rubric };
}

function foldSubRlmSpawned(
  state: RlmRunState,
  ev: SubRlmSpawnedEvent
): RlmRunState {
  const view: SubRlmView = {
    depth: ev.depth,
    model: ev.model,
    prompt_preview: ev.prompt_preview,
    duration_ms: null,
    error: null,
    spawnedAt: ev.timestamp,
    completedAt: null,
  };

  // A recursion marker node — parented via the frontier rule (§5.3), consistent
  // with how candidate fans attach.
  const parentId = frontierParent(state.tree);
  const node: TreeNode = {
    id: `subrlm-${state.subRlms.length + 1}`,
    kind: "subrlm",
    parentId,
    title: "Sub-RLM",
    iterationRange: [state.iterationCount, state.iterationCount],
  };

  return {
    ...state,
    subRlms: [...state.subRlms, view],
    tree: [...state.tree, node],
  };
}

function foldSubRlmComplete(
  state: RlmRunState,
  ev: SubRlmCompleteEvent
): RlmRunState {
  // Resolve the oldest still-open sub-RLM at this depth+model.
  const idx = state.subRlms.findIndex(
    (s) => s.completedAt === null && s.depth === ev.depth && s.model === ev.model
  );
  if (idx === -1) return state;

  const subRlms = state.subRlms.map((s, i) =>
    i === idx
      ? { ...s, duration_ms: ev.duration_ms, error: ev.error, completedAt: ev.timestamp }
      : s
  );
  return { ...state, subRlms };
}

function foldExperimentCompleted(
  state: RlmRunState,
  ev: ExperimentCompletedEvent
): RlmRunState {
  // Only update perModelMetrics when the experiment succeeded and per_model is
  // present.  A failed experiment does not overwrite prior successful data.
  if (!ev.success || !ev.per_model || Object.keys(ev.per_model).length === 0) {
    return state;
  }
  return { ...state, perModelMetrics: ev.per_model };
}

function foldRunWarning(
  state: RlmRunState,
  ev: RunWarningEvent
): RlmRunState {
  const warning: RunWarning = {
    level: ev.level,
    code: ev.code,
    message: ev.message,
    timestamp: ev.timestamp,
  };
  // Fallback enrichment: surface failure-coded warnings in the rubric
  // breakdown's "Recent errors" section even before the backend ships
  // recent_errors on the rubric_score event. Benign notices (e.g.
  // forced_iteration) are filtered out by isFailureWarning.
  let rubric = state.rubric;
  if (isFailureWarning(ev)) {
    const recentErrors = pushRecentError(state.rubric.recentErrors, {
      kind: ev.code || ev.level,
      message: ev.message,
      iteration: state.iterationCount,
    });
    if (recentErrors !== state.rubric.recentErrors) {
      rubric = { ...state.rubric, recentErrors };
    }
  }
  return { ...state, warnings: [...state.warnings, warning], rubric };
}

function foldGpuResolved(
  state: RlmRunState,
  ev: GpuResolvedEvent
): RlmRunState {
  const gpuPlan: GpuPlan = {
    runpod_id: ev.runpod_id,
    short_name: ev.short_name,
    vram_gb: ev.vram_gb,
    gpu_count: ev.gpu_count,
    cloud_type: ev.cloud_type,
    sku_usd_per_hr: ev.sku_usd_per_hr,
    total_usd_per_hr: ev.total_usd_per_hr,
    container_disk_gb: ev.container_disk_gb,
    volume_gb: ev.volume_gb,
    source: ev.source,
    requirements: { ...ev.requirements },
    ladder_remaining: ev.ladder_remaining,
    resolved_at: ev.resolved_at,
  };
  return { ...state, gpuPlan };
}

function foldRunComplete(
  state: RlmRunState,
  ev: RunCompleteEvent
): RlmRunState {
  const proposed = state.tree.filter((n) => n.kind === "candidate").length;
  const promoted = state.tree.filter(
    (n) => n.kind === "candidate" && n.outcome === "promoted"
  ).length;
  return {
    ...state,
    status: ev.status,
    report: {
      finalReportPath: ev.final_report_path,
      costUsd: ev.cost_usd,
      counts: {
        iterations: state.iterationCount,
        primitiveCalls: state.primitiveCalls.length,
        proposed,
        promoted,
      },
    },
  };
}

// ─── Pure fold ────────────────────────────────────────────────────────────────

/**
 * Pure reducer — returns a new RlmRunState. Never mutates its inputs.
 *
 * The `paper` root is lazily created on the first event so `INITIAL_RLM_STATE`
 * stays genuinely empty.
 */
export function fold(state: RlmRunState, event: RlmDashboardEvent): RlmRunState {
  // Every event ensures the paper root exists before its own handling.
  const tree = ensurePaper(state.tree);
  const seeded: RlmRunState = tree === state.tree ? state : { ...state, tree };

  switch (event.event) {
    case "repl_iteration":
      return foldReplIteration(seeded, event);
    case "primitive_call":
      return foldPrimitiveCall(seeded, event);
    case "rubric_score":
      return foldRubricScore(seeded, event);
    case "candidate_proposed":
      return foldCandidateProposed(seeded, event);
    case "candidate_outcome":
      return foldCandidateOutcome(seeded, event);
    case "sub_rlm_spawned":
      return foldSubRlmSpawned(seeded, event);
    case "sub_rlm_complete":
      return foldSubRlmComplete(seeded, event);
    case "run_complete":
      return foldRunComplete(seeded, event);
    case "run_warning":
      return foldRunWarning(seeded, event);
    case "iteration_heartbeat":
      // Store the heartbeat timestamp so the UI can detect a wedged run
      // (status === "running" && Date.now() - new Date(lastHeartbeatAt) > 60_000).
      return { ...seeded, lastHeartbeatAt: (event as IterationHeartbeatEvent).timestamp };
    case "user_message":
    case "user_message_response":
      // Consumed by other hooks; tree reducer is a no-op.
      return seeded;
    case "experiment_completed":
      return foldExperimentCompleted(seeded, event);
    case "gpu_resolved":
      return foldGpuResolved(seeded, event);
    default:
      // Exhaustiveness guard; unknown events return state unchanged.
      return seeded;
  }
}

/**
 * Crash-proof fold — the hooks fold LIVE SSE events, so a malformed event must
 * never throw out of the reducer and take down the whole lab page (2026-06-11:
 * a flat BES candidate_proposed with no nested `candidate` object did exactly
 * that). Logs the offending event and returns the prior state.
 */
export function safeFold(
  state: RlmRunState,
  event: RlmDashboardEvent
): RlmRunState {
  try {
    return fold(state, event);
  } catch (err) {
    console.error(
      "[use-rlm-run] dropped malformed event:",
      (event as { event?: string } | null | undefined)?.event,
      err
    );
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
  return useMemo(() => events.reduce(safeFold, INITIAL_RLM_STATE), [events]);
}

// ─── rAF-batched hook ─────────────────────────────────────────────────────────

export interface UseRlmRunBatchedResult {
  state: RlmRunState;
  addEvent: (ev: RlmDashboardEvent) => void;
  reset: () => void;
}

/**
 * useRlmRunBatched — push-based variant of useRlmRun that batches state
 * updates into a single requestAnimationFrame flush.
 *
 * Motivation: during heavy primitive calls the SSE stream emits 50–200 events
 * in rapid succession. Each `addEvent` call queues the event into
 * `pendingEventsRef`; the FIRST call in a batch schedules a rAF. When the
 * frame fires, all queued events are folded in one pass and a single setState
 * is issued — keeping the tree-layout and force-simulation from running 200
 * times per second.
 *
 * Guarantees:
 *  - Event ordering preserved (FIFO queue, folded left-to-right).
 *  - No events dropped (flush drains the entire pending buffer atomically).
 *  - A second batch before the first frame fires is merged into the same frame.
 *  - Cleanup: the pending rAF handle is cancelled on unmount.
 *
 * @param initialEvents Optional array of events to fold synchronously as the
 *   initial state (lazy initializer). This ensures test renders that pass a
 *   full fixture array see the folded state on the first render, matching the
 *   behavior of the pure useRlmRun(events) hook.
 */
export function useRlmRunBatched(initialEvents?: RlmDashboardEvent[]): UseRlmRunBatchedResult {
  const [state, setState] = useState<RlmRunState>(() =>
    initialEvents && initialEvents.length > 0
      ? initialEvents.reduce(safeFold, INITIAL_RLM_STATE)
      : INITIAL_RLM_STATE
  );

  // Stable ref: events waiting to be folded into state on the next frame.
  const pendingEventsRef = useRef<RlmDashboardEvent[]>([]);
  // The rAF handle, or null when no frame is pending.
  const rafHandleRef = useRef<number | null>(null);
  // The most-recently committed state — used as the fold starting point so
  // we always append to (not replay) the already-committed state.
  // Initialized to match the lazy initializer above.
  const committedStateRef = useRef<RlmRunState>(
    initialEvents && initialEvents.length > 0
      ? initialEvents.reduce(safeFold, INITIAL_RLM_STATE)
      : INITIAL_RLM_STATE
  );

  // Flush: drain pendingEventsRef, fold into committedState, commit.
  const flush = useCallback(() => {
    rafHandleRef.current = null;
    const events = pendingEventsRef.current;
    if (events.length === 0) return;
    pendingEventsRef.current = [];
    const nextState = events.reduce(safeFold, committedStateRef.current);
    committedStateRef.current = nextState;
    setState(nextState);
  }, []);

  const addEvent = useCallback((ev: RlmDashboardEvent) => {
    pendingEventsRef.current.push(ev);
    // Schedule a rAF only on the first event in a batch; subsequent events
    // in the same JavaScript task reuse the existing scheduled frame.
    if (rafHandleRef.current === null) {
      rafHandleRef.current = requestAnimationFrame(flush);
    }
  }, [flush]);

  // Reset: cancel any pending rAF, clear the event queue, and restore to the
  // initial state. Called when the active run changes (e.g., a new run starts
  // and the event array is cleared to []).
  const reset = useCallback(() => {
    if (rafHandleRef.current !== null) {
      cancelAnimationFrame(rafHandleRef.current);
      rafHandleRef.current = null;
    }
    pendingEventsRef.current = [];
    committedStateRef.current = INITIAL_RLM_STATE;
    setState(INITIAL_RLM_STATE);
  }, []);

  return { state, addEvent, reset };
}
