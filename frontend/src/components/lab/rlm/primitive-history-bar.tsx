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

  // Reverse-chronological order: newest call first.
  // Computed once per render; a shallow copy so the original is not mutated.
  const reversedCalls = collapsed ? [] : [...calls].reverse();

  return (
    <div className={styles.bar}>
      <button
        type="button"
        className={styles.toggle}
        onClick={() => setCollapsed((c) => !c)}
        aria-expanded={!collapsed}
      >
        {collapsed ? "▸" : "▾"} primitive call history —{" "}
        {calls.length} calls
      </button>

      {!collapsed && (
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
