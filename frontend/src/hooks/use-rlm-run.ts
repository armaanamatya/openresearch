/**
 * useRlmRun — the pure reducer for RLM run state.
 *
 * Spec: docs/superpowers/specs/2026-05-21-rlm-phase4-frontend-design.md §5 / §6
 *
 * Folds the 8 RLM dashboard events into `RlmRunState`. The four tree events
 * (candidate_proposed, candidate_outcome, sub_rlm_spawned, sub_rlm_complete)
 * build `RlmRunState.tree` — a flat node list, each node carrying `parentId`.
 */

import { useMemo } from "react";
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

/** A warning emitted by the backend watchdog (e.g. SDK aclose deadlock). */
export interface RunWarning {
  level: RunWarningEvent["level"];
  code: string;
  message: string;
  timestamp: string;
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
    previousAreas: [],
    attributableCandidate: null,
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

  return {
    ...state,
    status: state.status === "queued" ? "running" : state.status,
    tree,
    primitiveCalls: [...state.primitiveCalls, call],
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

  return {
    ...state,
    tree,
    rubric: {
      baseline: isFirst ? ev.score : state.rubric.baseline,
      current: ev.score,
      target: ev.target,
      series: [...state.rubric.series, { iteration: ev.iteration, score: ev.score }],
      // §4.2: snapshot the prior areas array BEFORE overwriting so the strip
      // can detect status flips (fail->partial->pass) at render time.
      previousAreas: state.rubric.areas.map((a) => ({ ...a })),
      areas: ev.areas.map((a) => ({ ...a })),
      attributableCandidate: state.rubric.attributableCandidate,
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
  const parentId = ev.parent_id ?? frontierParent(state.tree);

  // A buffered outcome may have arrived first (§5.3 out-of-order).
  const pending = state._pendingOutcomes[ev.candidate.id];

  const node: TreeNode = {
    id: `candidate-${ev.candidate.id}`,
    kind: "candidate",
    parentId,
    title: ev.candidate.display_title ?? ev.candidate.title,
    iterationRange: [ev.iteration, ev.iteration],
    round: ev.round,
    candidate: {
      id: ev.candidate.id,
      title: ev.candidate.title,
      displayTitle: ev.candidate.display_title,
      category: ev.candidate.category,
      description: ev.candidate.description,
      reasoning: ev.candidate.reasoning,
    },
    outcome: pending?.outcome,
    rubricDelta: pending?.rubricDelta ?? null,
  };

  let tree = [...state.tree, node];

  // If the buffered outcome was a declined, fold it into the round's group now.
  // Use the buffered outcome event's iteration so the declined-group iterationRange
  // reflects when the decline happened, not when the proposal was received.
  if (pending?.outcome === "declined") {
    tree = applyDeclinedGroup(tree, ev.round, parentId, pending.iteration);
  }

  // Drain the pending entry once consumed.
  const _pendingOutcomes = { ...state._pendingOutcomes };
  delete _pendingOutcomes[ev.candidate.id];

  // §4.2: update the live attributableCandidate pointer. If a buffered
  // outcome was drained above, its outcome rides through on the pointer.
  const rubric = {
    ...state.rubric,
    attributableCandidate: {
      id: ev.candidate.id,
      title: ev.candidate.title,
      outcome: pending?.outcome ?? null,
    },
  };

  return { ...state, tree, _pendingOutcomes, rubric };
}

function foldCandidateOutcome(
  state: RlmRunState,
  ev: CandidateOutcomeEvent
): RlmRunState {
  const nodeIdx = state.tree.findIndex(
    (n) => n.kind === "candidate" && n.candidate?.id === ev.candidate_id
  );

  // Out-of-order: the matching candidate_proposed has not arrived — buffer it.
  if (nodeIdx === -1) {
    return {
      ...state,
      _pendingOutcomes: {
        ...state._pendingOutcomes,
        [ev.candidate_id]: { outcome: ev.outcome, rubricDelta: ev.rubric_delta, iteration: ev.iteration },
      },
    };
  }

  const node = state.tree[nodeIdx];
  let tree = state.tree.map((n, i) =>
    i === nodeIdx
      ? { ...n, outcome: ev.outcome, rubricDelta: ev.rubric_delta }
      : n
  );

  // A declined candidate also collapses into its round's declined-group node.
  if (ev.outcome === "declined" && node.round != null) {
    tree = applyDeclinedGroup(tree, node.round, node.parentId, ev.iteration);
  }

  // §4.2: if this outcome matches the live attributable candidate, update its
  // outcome on the pointer. Otherwise leave the pointer unchanged.
  const rubric =
    state.rubric.attributableCandidate?.id === ev.candidate_id
      ? {
          ...state.rubric,
          attributableCandidate: {
            ...state.rubric.attributableCandidate,
            outcome: ev.outcome,
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
  return { ...state, warnings: [...state.warnings, warning] };
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
    default:
      // Exhaustiveness guard; unknown events return state unchanged.
      return seeded;
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
