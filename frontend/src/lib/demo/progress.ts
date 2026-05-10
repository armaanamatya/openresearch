/**
 * Pure derivations from a LiveDemoRunState used to render the progress strip.
 *
 * The pipeline emits structured stderr lines like:
 *   ==================================================
 *     > Starting: paper_understood
 *   ==================================================
 *     [paper-understanding] (12s) tool: Read /home/.../parsed_full_text.txt
 *     ✓ paper_understood done in 66s
 *     X FAILED: paper_understood -- ValueError: ...
 *
 * We parse those to surface "what is the agent doing right now" and
 * "how long since the last sign of life" without coupling to backend
 * internals.
 */
import type { LiveDemoRunState } from "./demo-run-types";

export const STALL_THRESHOLD_SECONDS = 90;

/** Canonical pipeline stage order for percent-complete estimation. */
export const PIPELINE_STAGES: ReadonlyArray<{ key: string; label: string }> = [
  { key: "paper_understood", label: "Reading paper" },
  { key: "environment_resolved", label: "Resolving environment" },
  { key: "baseline_implemented", label: "Implementing baseline" },
  { key: "baseline_executed", label: "Running baseline" },
  { key: "paths_explored", label: "Exploring improvements" },
  { key: "verification_complete", label: "Verifying results" }
];

const STAGE_LABEL_BY_KEY = new Map(PIPELINE_STAGES.map((s) => [s.key, s.label]));

export interface ProgressSnapshot {
  /** Canonical stage key the agent is currently on, or null if unknown. */
  currentStageKey: string | null;
  /** Human-friendly label for the current stage. */
  currentStageLabel: string;
  /** Latest activity line from the log (e.g. "Read parsed_full_text.txt"). */
  lastActivityText: string | null;
  /** Seconds since the last log line, computed at `nowMs`. */
  lastActivitySeconds: number | null;
  /** Total elapsed seconds since `startedAt`, computed at `nowMs`. */
  elapsedSeconds: number | null;
  /** Percent complete in [0, 1] based on completed stages and gates. */
  percentComplete: number;
  /** True iff status is running and no log activity for >= STALL_THRESHOLD. */
  isStalled: boolean;
  /** Number of completed stages (those with a "✓ <stage> done" log line). */
  completedStageCount: number;
  /** Most recent failure message, if any. */
  failureText: string | null;
}

const STAGE_START_RE = />\s*Starting:\s*([a-z0-9_]+)/i;
const STAGE_DONE_RE = /✓\s*([a-z0-9_]+)\s+done/i;
const STAGE_FAIL_RE = /X\s*FAILED:\s*([a-z0-9_]+)\s*--\s*(.+)$/i;
const ACTIVITY_RE = /\[(?<agent>[^\]]+)\]\s*(?:\((?<elapsed>\d+)s\)\s*)?(?<rest>.+)$/;

interface LogParseResult {
  currentStageKey: string | null;
  lastActivityText: string | null;
  completedStages: Set<string>;
  failureText: string | null;
}

export function parseLog(log: string): LogParseResult {
  const lines = log ? log.split(/\r?\n/) : [];
  let currentStageKey: string | null = null;
  let lastActivityText: string | null = null;
  let failureText: string | null = null;
  const completedStages = new Set<string>();

  for (const line of lines) {
    const startMatch = STAGE_START_RE.exec(line);
    if (startMatch) {
      currentStageKey = startMatch[1].toLowerCase();
      continue;
    }
    const doneMatch = STAGE_DONE_RE.exec(line);
    if (doneMatch) {
      completedStages.add(doneMatch[1].toLowerCase());
      continue;
    }
    const failMatch = STAGE_FAIL_RE.exec(line);
    if (failMatch) {
      failureText = `${failMatch[1]}: ${failMatch[2].trim()}`;
      continue;
    }
    const activityMatch = ACTIVITY_RE.exec(line.trim());
    if (activityMatch?.groups?.rest) {
      // Truncate aggressive — UI shows the tail, not a paragraph.
      lastActivityText = activityMatch.groups.rest.slice(0, 160);
    }
  }

  return { currentStageKey, lastActivityText, completedStages, failureText };
}

export function computeProgress(
  state: LiveDemoRunState,
  nowMs: number = Date.now()
): ProgressSnapshot {
  const parsed = parseLog(state.log ?? "");
  const startedAtMs = state.startedAt ? Date.parse(state.startedAt) : NaN;
  const updatedAtMs = state.updatedAt ? Date.parse(state.updatedAt) : NaN;

  const elapsedSeconds = Number.isFinite(startedAtMs)
    ? Math.max(0, Math.floor((nowMs - startedAtMs) / 1000))
    : null;
  const lastActivitySeconds = Number.isFinite(updatedAtMs)
    ? Math.max(0, Math.floor((nowMs - updatedAtMs) / 1000))
    : elapsedSeconds;

  const totalStages = PIPELINE_STAGES.length;
  const completedStageCount = parsed.completedStages.size;
  // Half-credit for an in-flight stage so the bar moves while the agent is
  // working, not only when it crosses a stage boundary.
  const inFlightCredit =
    parsed.currentStageKey && !parsed.completedStages.has(parsed.currentStageKey)
      ? 0.5
      : 0;
  const percentComplete = Math.min(
    1,
    (completedStageCount + inFlightCredit) / totalStages
  );

  const currentStageKey = parsed.currentStageKey;
  const currentStageLabel = currentStageKey
    ? STAGE_LABEL_BY_KEY.get(currentStageKey) ?? currentStageKey
    : state.status === "running"
      ? "Starting…"
      : statusLabel(state.status);

  const isStalled =
    state.status === "running" &&
    lastActivitySeconds !== null &&
    lastActivitySeconds >= STALL_THRESHOLD_SECONDS;

  return {
    currentStageKey,
    currentStageLabel,
    lastActivityText: parsed.lastActivityText,
    lastActivitySeconds,
    elapsedSeconds,
    percentComplete,
    isStalled,
    completedStageCount,
    failureText: parsed.failureText ?? state.error ?? null
  };
}

function statusLabel(status: LiveDemoRunState["status"]): string {
  switch (status) {
    case "queued":
      return "Queued";
    case "running":
      return "Running";
    case "stopped":
      return "Stopped";
    case "completed":
      return "Completed";
    case "failed":
      return "Failed";
    default:
      return status;
  }
}

export function formatDuration(seconds: number | null): string {
  if (seconds === null || !Number.isFinite(seconds)) return "—";
  if (seconds < 60) return `${seconds}s`;
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  if (m < 60) return `${m}m ${s.toString().padStart(2, "0")}s`;
  const h = Math.floor(m / 60);
  const remM = m % 60;
  return `${h}h ${remM.toString().padStart(2, "0")}m`;
}
