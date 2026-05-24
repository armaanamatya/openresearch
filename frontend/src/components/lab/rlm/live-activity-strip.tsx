"use client";

import { useMemo } from "react";
import type { PrimitiveCallView, SubRlmView } from "../../../hooks/use-rlm-run";
import { describePrimitiveError } from "./primitive-error";

/**
 * Always-visible narration of what the agent is doing RIGHT NOW.
 *
 * Designed to never look "blank" — the 2026-05-23 user complaint:
 * "user has no clue what is going on atp, just 7 mins doing nothing, we
 * should have a view so the user can understand what the agent is doing
 * at all times". The existing dashboard only updates when a primitive_call
 * or repl_iteration event lands; between events the UI sat silent for
 * minutes despite the agent actively thinking.
 *
 * Derivation order (most → least specific):
 *   1. A primitive_call is in flight (status="start" with no matching ok/error)
 *      → "Running <primitive> · <Xs>".
 *   2. A sub_rlm is in flight (most recent sub_rlm_spawned has no matching complete)
 *      → "Sub-RLM querying paper · <Xs>".
 *   3. We just completed a repl_iteration → "Iteration X done · root thinking <Xs>".
 *   4. We have nothing → "Starting up · reading paper... <Xs>".
 *
 * The Xs counter always advances (driven by the parent's clock tick),
 * so the strip is visibly alive even during the longest primitives.
 */

export interface LiveActivityStripProps {
  status: "queued" | "running" | "completed" | "partial" | "failed";
  iterationCount: number;
  primitiveCalls: PrimitiveCallView[];
  subRlms: SubRlmView[];
  lastHeartbeatAt: string | null;
  /** Client-side clock tick from the parent; avoids Date.now() during render. */
  nowMs: number | null;
  /** When the run was kicked off — fallback "Xs elapsed" anchor when no events yet. */
  startedAt?: string | null;
}

interface Narration {
  icon: string;
  label: string;
  /** Seconds elapsed for the current activity (Xs counter). */
  secs: number | null;
  /** Tooltip with more detail. */
  detail: string;
  /** Tone — "info" (normal), "muted" (idle/queued), "warn" (slow), "err" (failed). */
  tone: "info" | "muted" | "warn" | "err";
  /** "Next likely" hint — shown as a thin second line when present. Inferred
   *  from the static primitive dependency graph (NEXT_LIKELY). Omitted when
   *  the next step isn't predictable (e.g. between sub-RLMs, or when no
   *  primitive has landed yet). */
  nextLikely?: string | null;
  supporting?: string | null;
}

/**
 * Static "what usually comes after X" map for the LiveActivityStrip's
 * next-likely hint. Derived from the canonical pipeline phases in
 * CLAUDE.md (Ingest → Understand → Plan → Implement → Run → Verify).
 * Misses → hint omitted; never raises.
 */
const NEXT_LIKELY: Record<string, string> = {
  understand_section: "extract_hyperparameters or detect_environment",
  extract_hyperparameters: "detect_environment",
  detect_environment: "build_environment",
  build_environment: "plan_reproduction",
  plan_reproduction: "implement_baseline",
  implement_baseline: "run_experiment",
  run_experiment: "verify_against_rubric",
  verify_against_rubric: "propose_improvements or final_report",
  propose_improvements: "record_candidate_outcome",
  record_candidate_outcome: "next iteration or final_report",
};

function diffSecs(thenIso: string | null, nowMs: number | null): number | null {
  if (!thenIso || nowMs === null) return null;
  const then = new Date(thenIso).getTime();
  if (Number.isNaN(then)) return null;
  return Math.max(0, Math.floor((nowMs - then) / 1000));
}

function narrate(props: LiveActivityStripProps): Narration {
  const { status, iterationCount, primitiveCalls, subRlms, lastHeartbeatAt, nowMs, startedAt } = props;

  if (status === "failed") {
    return {
      icon: "✖",
      label: "Run failed",
      secs: null,
      detail: "The run subprocess exited non-zero — check runner.stderr.log for the traceback.",
      tone: "err",
    };
  }
  if (status === "completed" || status === "partial") {
    return {
      icon: status === "completed" ? "✓" : "◐",
      label: status === "completed" ? "Run completed" : "Run completed (partial verification)",
      secs: null,
      detail: "final_report.json is on disk. Open the report rail for the full breakdown.",
      tone: status === "completed" ? "info" : "warn",
    };
  }
  if (status === "queued") {
    return {
      icon: "•",
      label: "Queued — backend acknowledging…",
      secs: diffSecs(startedAt ?? null, nowMs),
      detail: "Run subprocess has been spawned but has not yet emitted its first event.",
      tone: "muted",
    };
  }

  // status === "running": find the most specific activity.

  // 1. In-flight primitive — walk primitiveCalls backwards for a "start" with no later terminator.
  for (let i = primitiveCalls.length - 1; i >= 0; i--) {
    const c = primitiveCalls[i];
    if (c.status !== "start") continue;
    let terminated = false;
    for (let j = i + 1; j < primitiveCalls.length; j++) {
      if (primitiveCalls[j].primitive === c.primitive && primitiveCalls[j].status !== "start") {
        terminated = true;
        break;
      }
    }
    if (!terminated) {
      const secs = diffSecs(c.timestamp, nowMs);
      const slow = secs !== null && secs > 60;
      return {
        icon: "▶",
        label: `Running ${c.primitive}`,
        secs,
        detail:
          `Primitive '${c.primitive}' is in flight. Long primitives (implement_baseline 5-15 min, ` +
          `build_environment 1-5 min, run_experiment up to 30 min) regularly produce minutes of silence ` +
          `between events — this is expected.`,
        tone: slow ? "warn" : "info",
        nextLikely: NEXT_LIKELY[c.primitive] ?? null,
      };
    }
  }

  // 2. In-flight sub-RLM — most recent sub_rlm without a matching complete.
  // SubRlmView has duration_ms !== null only when complete, so a null duration = in flight.
  const inFlightSubRlm = [...subRlms].reverse().find((s) => s.duration_ms === null);
  if (inFlightSubRlm) {
    // SubRlmView doesn't carry the start timestamp directly in this projection,
    // so we anchor "since" on lastHeartbeatAt as a coarse proxy (still moves visibly).
    const secs = diffSecs(lastHeartbeatAt, nowMs);
    const summary = inFlightSubRlm.prompt_preview?.slice(0, 80) || "(no preview)";
    return {
      icon: "↳",
      label: `Sub-RLM depth ${inFlightSubRlm.depth} querying paper`,
      secs,
      detail: `Sub-RLM prompt preview: ${summary}…`,
      tone: "info",
    };
  }

  // 3. Between iterations — the root model is thinking about its next REPL turn.
  if (iterationCount > 0) {
    const secs = diffSecs(lastHeartbeatAt, nowMs);
    const slow = secs !== null && secs > 120;
    // If the most recent terminated primitive was an ERROR, the root is
    // not just "thinking" — it's recovering from a failure. Surface the
    // structured error detail (pydantic field paths etc., from binding.py's
    // T2 fix) so the user sees what went wrong AND that recovery is in
    // progress, not stuck.
    const lastTerminated = [...primitiveCalls]
      .reverse()
      .find((c) => c.status === "ok" || c.status === "error");
    if (lastTerminated && lastTerminated.status === "error") {
      const errDetail = describePrimitiveError(lastTerminated);
      return {
        icon: "↻",
        label: `Recovering from ${lastTerminated.primitive} error — root deciding retry path`,
        secs,
        detail:
          `Last error: ${errDetail.reason}\n\n` +
          `The root REPL got the full traceback and is constructing the ` +
          `next REPL turn (typically a retry, an alternative approach, or ` +
          `a fallback primitive). No action needed — the rlm adaptive loop ` +
          `is designed for this.`,
        tone: "warn",
        supporting: `Reason: ${errDetail.reason} Recovery: retry, fallback, or repair on the next REPL turn.`,
      };
    }
    return {
      icon: "…",
      label: `Iteration ${iterationCount} complete — root thinking about next turn`,
      secs,
      detail:
        "No primitive is in flight; the rlm root is constructing its next REPL turn. " +
        "Typically 5-60s; can be longer when the root needs to read large paper slices.",
      tone: slow ? "warn" : "info",
    };
  }

  // 4. Pre-first-iteration cold start.
  const secs = diffSecs(startedAt ?? null, nowMs);
  const slow = secs !== null && secs > 300;
  return {
    icon: "…",
    label: "Starting up — root model reading the paper",
    secs,
    detail:
      "The runpod cold path (pod create + image pull) can take 3-5 min before the first " +
      "repl_iteration event lands. The agent is alive and working.",
    tone: slow ? "warn" : "muted",
  };
}

function toneColors(tone: Narration["tone"]): { bg: string; ink: string; dot: string } {
  switch (tone) {
    case "info":
      return { bg: "var(--accent-soft)", ink: "var(--accent-ink)", dot: "var(--accent)" };
    case "warn":
      return { bg: "var(--warn-soft)", ink: "var(--warn-ink)", dot: "var(--warn)" };
    case "err":
      return { bg: "var(--err-soft)", ink: "var(--err)", dot: "var(--err)" };
    case "muted":
    default:
      return { bg: "var(--chip)", ink: "var(--ink-2)", dot: "var(--muted-2)" };
  }
}

export function LiveActivityStrip(props: LiveActivityStripProps) {
  const narration = useMemo(() => narrate(props), [props]);
  const colors = toneColors(narration.tone);

  return (
    <>
      <div
        role="status"
        aria-live="polite"
        title={narration.detail}
        style={{
          display: "flex",
          alignItems: "center",
          gap: "12px",
          padding: "8px 14px",
          margin: "0 0 8px 0",
          borderRadius: "8px",
          background: colors.bg,
          color: colors.ink,
          fontSize: "0.85rem",
          fontWeight: 500,
          lineHeight: 1.4,
          border: `1px solid ${colors.dot}`,
          minHeight: "36px",
        }}
      >
        <span
          aria-hidden="true"
          data-live-pulse
          style={{
            width: "10px",
            height: "10px",
            borderRadius: "50%",
            background: colors.dot,
            flex: "0 0 auto",
            // Pulse the dot whenever an activity is in flight to make
            // "this is alive" unmistakable even when seconds tick slowly.
            animation:
              props.status === "running" ? "rlmLivePulse 1.6s ease-in-out infinite" : "none",
          }}
        />
        <span style={{ flex: "0 0 auto", fontSize: "1rem" }}>{narration.icon}</span>
        <span style={{ flex: 1, minWidth: 0 }}>
          <span style={{ display: "block" }}>{narration.label}</span>
          {narration.nextLikely && (
            <span
              style={{
                display: "block",
                fontSize: "0.74rem",
                fontWeight: 500,
                opacity: 0.78,
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
              }}
            >
              next likely: {narration.nextLikely}
            </span>
          )}
          {narration.supporting && (
            <span
              style={{
                display: "block",
                fontSize: "0.74rem",
                fontWeight: 500,
                opacity: 0.86,
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
              }}
            >
              {narration.supporting}
            </span>
          )}
        </span>
        {narration.secs !== null && (
          <span
            style={{
              flex: "0 0 auto",
              fontVariantNumeric: "tabular-nums",
              fontSize: "0.78rem",
              opacity: 0.85,
            }}
          >
            {narration.secs}s
          </span>
        )}
      </div>
      <style>{`
        @keyframes rlmLivePulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.45; transform: scale(0.85); }
        }
        @media (prefers-reduced-motion: reduce) {
          [data-live-pulse] {
            animation: none !important;
          }
        }
      `}</style>
    </>
  );
}
