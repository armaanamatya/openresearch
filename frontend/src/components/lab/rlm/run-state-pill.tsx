"use client";

import { useMemo, type ReactElement } from "react";
import type {
  RunStateKind,
  RunStateSubstate,
} from "../../../lib/events/rlm-events";
import styles from "./run-state-pill.module.css";

/**
 * Derived-run-state pill — the plain-language renderer for the workflow-polish
 * liveness contract. Replaces the legacy `status pill + "no signal Xs" chip +
 * iteration counter` ambiguity with one signal a non-engineer can read.
 *
 * Source field: `RlmRunState.runStateKind` / `runStateSubstate`, populated by
 * the `run_state` SSE event emitted by `RunStateComputer`
 * (`backend/agents/rlm/run_state.py`).
 *
 * Spec: `docs/superpowers/specs/2026-05-27-derived-run-state-contract-design.md`.
 */
export interface RunStatePillProps {
  kind: RunStateKind | null;
  substate: RunStateSubstate | null;
  /** If true, the pill is rendered as a compact chip beside the status pill. */
  compact?: boolean;
}

interface RenderedState {
  tone: "neutral" | "active" | "watch" | "alert" | "ok" | "err";
  label: string;
  detail: string | null;
}

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  if (m < 60) return s ? `${m}m${s.toString().padStart(2, "0")}s` : `${m}m`;
  const h = Math.floor(m / 60);
  const mm = m % 60;
  return mm ? `${h}h${mm.toString().padStart(2, "0")}m` : `${h}h`;
}

function renderState(
  kind: RunStateKind | null,
  substate: RunStateSubstate | null,
): RenderedState {
  if (kind === null) {
    return { tone: "neutral", label: "queued", detail: null };
  }
  const prim = substate?.primitive ?? null;
  const sActive = substate?.seconds_active ?? 0;
  const lastFile = substate?.last_file_touched ?? null;
  const reason = substate?.reason ?? null;
  switch (kind) {
    case "initializing":
      return {
        tone: "neutral",
        label: "starting up",
        detail: null,
      };
    case "working":
      if (lastFile && prim) {
        return {
          tone: "active",
          label: `writing ${lastFile}`,
          detail: `${prim} · ${formatDuration(sActive)}`,
        };
      }
      if (lastFile) {
        return {
          tone: "active",
          label: `writing ${lastFile}`,
          detail: formatDuration(sActive),
        };
      }
      if (prim) {
        return {
          tone: "active",
          label: prim,
          detail: formatDuration(sActive),
        };
      }
      return { tone: "active", label: "working", detail: null };
    case "idle":
      // Idle means "primitive in flight but no file activity for >60s" —
      // typical for LLM-bound primitives (plan, propose, verify) that wait
      // on a network round-trip. Honest about this; not an alarm.
      if (prim) {
        return {
          tone: "watch",
          label: `waiting on ${prim}`,
          detail: formatDuration(sActive),
        };
      }
      return { tone: "watch", label: "waiting", detail: null };
    case "stuck":
      // Stuck means mtime is past the 240s stall threshold AND heartbeats are
      // stale. The backend pre-emit watchdog escalates at the same threshold,
      // so this aligns with the orchestrator's own view.
      return {
        tone: "alert",
        label: prim ? `no progress on ${prim}` : "no progress",
        detail: `${formatDuration(sActive)} · investigating`,
      };
    case "interrupted":
      return {
        tone: "alert",
        label: "interrupted",
        detail: reason ?? "process disappeared",
      };
    case "completed":
      return { tone: "ok", label: "completed", detail: null };
    case "failed":
      return {
        tone: "err",
        label: "failed",
        detail: reason ?? null,
      };
    default:
      return { tone: "neutral", label: kind, detail: null };
  }
}

export function RunStatePill({
  kind,
  substate,
  compact = false,
}: RunStatePillProps): ReactElement | null {
  const rendered = useMemo(
    () => renderState(kind, substate),
    [kind, substate],
  );
  // Show nothing before any run_state has landed; the legacy status pill
  // covers the initial render.
  if (kind === null) return null;

  const className = [
    styles.pill,
    styles[`tone-${rendered.tone}`],
    compact ? styles.compact : "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <span
      className={className}
      role="status"
      aria-live="polite"
      data-kind={kind}
      title={
        rendered.detail
          ? `${rendered.label} — ${rendered.detail}`
          : rendered.label
      }
    >
      <span
        className={`${styles.dot} ${styles[`dot-${rendered.tone}`]}`}
        aria-hidden
      />
      <span className={styles.label}>{rendered.label}</span>
      {rendered.detail ? (
        <span className={styles.detail}>{rendered.detail}</span>
      ) : null}
    </span>
  );
}
