// RLM dashboard event types — the 8 event shapes consumed by the RLM lab UI.
// Spec: docs/superpowers/specs/2026-05-21-rlm-phase4-frontend-design.md §4.
// This file is the foundation every other Phase 4 task imports.
// contract.ts (the legacy 14-stage event types) is left untouched; this is a
// separate file. Removal of contract.ts is Phase 6.

// ─── §4.1 Existing events (emitted today) ────────────────────────────────────

export interface ReplIterationEvent {
  event: "repl_iteration";
  timestamp: string;
  iteration: number; // 1-based
  response: string; // root model's reasoning text, ≤4000 chars
  code_blocks: Array<{
    code: string;
    stdout_meta: { length: number; prefix: string; has_traceback: boolean };
    stderr_meta: { length: number; prefix: string; has_traceback: boolean };
    vars: Record<string, { type: string; size: number }>; // REPL var manifest
    sub_calls: number;
  }>;
  sub_calls: number;
  timing: number | null;
}

export interface PrimitiveCallEvent {
  event: "primitive_call";
  timestamp: string;
  primitive: string; // "understand_section" | … (the 9 primitives)
  status: "start" | "ok" | "error";
  args_summary: Record<string, unknown>; // value-free (type/length only)
  result_summary: string | null; // value-free, e.g. "list[3]"
  iteration: number | null;
  rubric_delta: number | null;
}

export interface SubRlmSpawnedEvent {
  event: "sub_rlm_spawned";
  timestamp: string;
  depth: number;
  model: string;
  prompt_preview: string; // preview ≤200 chars
}

export interface SubRlmCompleteEvent {
  event: "sub_rlm_complete";
  timestamp: string;
  depth: number;
  model: string;
  duration_ms: number;
  error: string | null;
}

export interface RunCompleteEvent {
  event: "run_complete";
  timestamp: string;
  status: "completed" | "partial" | "failed";
  iterations: number;
  rubric_score: number | null;
  cost_usd: number | null;
  final_report_path: string | null;
}

// ─── §4.2 New events — the fixture contract (NOT emitted yet) ─────────────────

export interface CandidateProposedEvent {
  event: "candidate_proposed";
  timestamp: string;
  iteration: number;
  round: number; // 1-based; one propose_improvements() call = one fan
  /** Node this branches from. STRONGLY RECOMMENDED from the backend; if absent the reducer infers it (§5.3). */
  parent_id?: string;
  candidate: {
    id: string; // stable, e.g. "c5"
    title: string; // short, model-generated (improvement name) — not corpus
    category: string; // free-form tag, e.g. "optimizer" — not corpus
    description: string; // 1–2 sentences, model-generated
    reasoning: string; // why the root proposed it
  };
}

export interface CandidateOutcomeEvent {
  event: "candidate_outcome";
  timestamp: string;
  iteration: number;
  candidate_id: string;
  outcome:
    | "running"
    | "promoted"
    | "marginal"
    | "failed"
    | "skipped"
    | "declined";
  rubric_delta: number | null; // score change this candidate produced, if measured
}

export interface RubricScoreEvent {
  event: "rubric_score";
  timestamp: string;
  iteration: number;
  score: number; // overall, 0–1
  target: number; // rubric target, 0–1
  areas: Array<{
    area: string;
    score: number;
    weight: number;
    status: "pass" | "partial" | "fail";
  }>;
}

// ─── Union type + discriminant helpers ───────────────────────────────────────

// ─── §4.3 Chat-steering events (user messages + assistant responses) ────────────

export interface UserMessageEvent {
  event: "user_message";
  timestamp: string;
  content: string;
}

export interface UserMessageResponseEvent {
  event: "user_message_response";
  timestamp: string;
  message: string;
}

/** Emitted by the stderr watchdog when a degraded condition is detected. */
export interface RunWarningEvent {
  event: "run_warning";
  timestamp: string;
  level: "warn" | "error";
  code: string;
  message: string;
}

// ─── §4.4 Heartbeat event — liveness signal ────────────────────────────────────

export interface IterationHeartbeatEvent {
  event: "iteration_heartbeat";
  timestamp: string;
  /** 1-based root-loop iteration, or null before the first iteration is logged. */
  iteration: number | null;
  /** Monotonic per-process counter; increments on every heartbeat() call. */
  counter: number;
  /** Optional note from the root model, e.g. "about to implement_baseline". */
  note: string;
}

export const RLM_EVENT_TYPES = [
  "repl_iteration",
  "primitive_call",
  "sub_rlm_spawned",
  "sub_rlm_complete",
  "run_complete",
  "candidate_proposed",
  "candidate_outcome",
  "rubric_score",
  "user_message",
  "user_message_response",
  "run_warning",
  "iteration_heartbeat",
] as const;

export type RlmDashboardEvent =
  | ReplIterationEvent
  | PrimitiveCallEvent
  | SubRlmSpawnedEvent
  | SubRlmCompleteEvent
  | RunCompleteEvent
  | CandidateProposedEvent
  | CandidateOutcomeEvent
  | RubricScoreEvent
  | UserMessageEvent
  | UserMessageResponseEvent
  | RunWarningEvent
  | IterationHeartbeatEvent;

export function isRlmEvent(value: unknown): value is RlmDashboardEvent {
  if (typeof value !== "object" || value === null) return false;
  const ev = (value as { event?: unknown }).event;
  return (
    typeof ev === "string" &&
    (RLM_EVENT_TYPES as readonly string[]).includes(ev)
  );
}

// ─── Compile-time guard: RLM_EVENT_TYPES must list exactly the RlmDashboardEvent discriminants ───
// If the array and union ever diverge, `npx tsc --noEmit` will fail here.
type _Equal<A, B> =
  (<T>() => T extends A ? 1 : 2) extends (<T>() => T extends B ? 1 : 2) ? true : false;
type _Expect<T extends true> = T;
export type _RlmEventTypesInSync = _Expect<
  _Equal<RlmDashboardEvent["event"], (typeof RLM_EVENT_TYPES)[number]>
>;
