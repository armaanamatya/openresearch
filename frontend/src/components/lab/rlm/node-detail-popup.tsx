"use client";

import { useEffect, useRef } from "react";
import type { TreeNode, IterationView } from "../../../hooks/use-rlm-run";
import styles from "./node-detail-popup.module.css";

export interface NodeDetailPopupProps {
  node: TreeNode;
  iteration: IterationView | null;
  onClose: () => void;
}

/**
 * NodeDetailPopup — floating detail card shown when an exploration-tree node
 * is clicked (spec §7/§8, §9, §14).
 *
 * Shows:
 *   - Node title + identity (kind, round if present)
 *   - Candidate metadata: category, rubricDelta
 *   - Iteration response (reasoning)
 *   - Code blocks: code text + stdout metadata (length, has_traceback)
 *     — NOT raw stdout, NOT vars values (corpus-safety §14)
 *
 * Dismissed via a close button (accessible name /close|dismiss/i) or Escape key.
 */
export function NodeDetailPopup({ node, iteration, onClose }: NodeDetailPopupProps) {
  const { kind, title, round, candidate, rubricDelta } = node;

  // Close button ref — focused on mount so focus moves into the popup when it
  // opens (spec §9 focus dance). ExplorationCanvas restores focus to the
  // triggering TreeNode when this popup closes.
  const closeBtnRef = useRef<HTMLButtonElement | null>(null);

  // Register Escape keydown listener; clean up on unmount or onClose change.
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [onClose]);

  // Move focus into the popup on mount (focus-in half of the §9 focus dance).
  useEffect(() => {
    closeBtnRef.current?.focus();
  }, []);

  return (
    <div
      className={styles.popup}
      data-testid="node-detail-popup"
      role="dialog"
      aria-modal="true"
      aria-label={`Node detail: ${title}`}
    >
      {/* Header: title + identity + dismiss */}
      <div className={styles.header}>
        <div className={styles.identity}>
          <span className={styles.title}>{title}</span>
          <span className={styles.kindBadge}>{kind}</span>
          {round != null && (
            <span className={styles.roundBadge}>round {round}</span>
          )}
        </div>
        <button
          type="button"
          ref={closeBtnRef}
          className={styles.closeBtn}
          aria-label="Close node detail"
          onClick={onClose}
        >
          {/* × character for visual affordance */}
          <svg
            width="14"
            height="14"
            viewBox="0 0 14 14"
            aria-hidden="true"
            fill="none"
          >
            <line
              x1="2" y1="2" x2="12" y2="12"
              stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"
            />
            <line
              x1="12" y1="2" x2="2" y2="12"
              stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"
            />
          </svg>
        </button>
      </div>

      {/* Candidate metadata — only when kind === "candidate" */}
      {kind === "candidate" && candidate && (
        <div className={styles.candidateMeta}>
          <span className={styles.category}>{candidate.category}</span>
          <span className={styles.description}>{candidate.description}</span>
          {rubricDelta != null && (
            <span className={styles.rubricDelta}>
              {rubricDelta >= 0 ? "+" : ""}
              {rubricDelta.toFixed(2)} rubric
            </span>
          )}
        </div>
      )}

      {/* Iteration detail */}
      {iteration === null ? (
        <p className={styles.noDetail}>no iteration detail</p>
      ) : (
        <div className={styles.iterationBody}>
          {/* Reasoning / response */}
          <p className={styles.response}>{iteration.response}</p>

          {/* Code blocks — code text + stdout metadata only; never raw stdout */}
          {iteration.code_blocks?.map((block, i) => (
            <div key={i} className={styles.codeBlock}>
              <pre className={styles.code}>{block.code}</pre>

              {/* stdout_meta: length + traceback flag; NOT the raw prefix, NOT vars */}
              {block.stdout_meta && (
                <div className={styles.stdoutMeta}>
                  <span className={styles.stdoutLength}>
                    stdout: {String(block.stdout_meta.length)} chars
                  </span>
                  {block.stdout_meta.has_traceback && (
                    <span className={styles.tracebackBadge}>traceback</span>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
