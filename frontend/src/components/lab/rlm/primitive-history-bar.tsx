"use client";

import { useState } from "react";
import type { PrimitiveCallView } from "../../../hooks/use-rlm-run";
import styles from "./primitive-history-bar.module.css";

interface PrimitiveHistoryBarProps {
  calls: PrimitiveCallView[];
}

/**
 * PrimitiveHistoryBar — collapsible bottom strip listing primitive calls.
 *
 * Spec: docs/superpowers/specs/2026-05-21-rlm-phase4-frontend-design.md §7
 *
 * Collapsed (default): a single bar showing the call count summary.
 * Expanded: a reverse-chronological list (newest first), one row per call.
 * Error rows are highlighted with the coral (--err) token.
 */
export function PrimitiveHistoryBar({ calls }: PrimitiveHistoryBarProps) {
  const [collapsed, setCollapsed] = useState(true);

  // BUG-NEW-031 (companion fix): every primitive emits a start AND a terminal
  // event, so raw `calls.length` is 2× the real invocation count. Filter to
  // terminal events (ok/error/coerced) — one row per invocation.
  const terminalCalls = calls.filter((c) => c.status !== "start");

  // Reverse-chronological order: newest call first.
  const reversedCalls = collapsed ? [] : [...terminalCalls].reverse();

  return (
    <div className={styles.bar}>
      <button
        type="button"
        className={styles.toggle}
        onClick={() => setCollapsed((c) => !c)}
        aria-expanded={!collapsed}
        title="Primitive calls are the tool invocations made by the root REPL. Long gaps are normal while a primitive is in flight."
      >
        {collapsed ? "▸" : "▾"} primitive call history —{" "}
        {terminalCalls.length > 0
          ? `${terminalCalls.length} calls`
          : "waiting for first primitive"}
      </button>

      {!collapsed && calls.length === 0 && (
        <p className={styles.empty}>
          Primitive calls appear after the root starts reading the paper and selecting tools.
        </p>
      )}

      {!collapsed && calls.length > 0 && (
        <ol className={styles.list} aria-label="Primitive call history">
          {reversedCalls.map((call, idx) => (
            <li
              key={idx}
              data-testid="primitive-call-row"
              className={[
                styles.row,
                call.status === "error" ? styles.rowError : "",
              ]
                .filter(Boolean)
                .join(" ")}
            >
              <span className={styles.primitive}>{call.primitive}</span>
              <span className={styles.status}>{call.status}</span>
              {call.iteration !== null && (
                <span className={styles.iteration}>
                  iter {call.iteration}
                </span>
              )}
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}
