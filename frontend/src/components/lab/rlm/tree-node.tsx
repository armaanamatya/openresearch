"use client";

import type { TreeNode as TreeNodeData } from "../../../hooks/use-rlm-run";
import styles from "./tree-node.module.css";

export interface TreeNodeProps {
  node: TreeNodeData;
  selected: boolean;
  onSelect: (id: string) => void;
}

/**
 * TreeNode — one node card in the exploration tree (spec §6, §8, §9).
 *
 * Renders as a <button> for keyboard-focusability. The root element carries
 * a data-outcome attribute when the node has an outcome, so outcome is
 * conveyed by text (badge) + data attribute — never color alone (spec §9).
 *
 * Node kinds (spec §6):
 *   paper        — root, no outcome
 *   work         — trunk step (comprehension / environment / baseline-build)
 *   baseline     — first scored milestone, no outcome
 *   candidate    — improvement hypothesis with outcome badge + category subtitle
 *   subrlm       — recursion marker ("Sub-RLM"), no outcome
 *   declined-group — collapsed declined candidates
 *
 * Outcome palette (spec §9 → tokens.css):
 *   promoted   → --accent (forest green)
 *   marginal   → --warn (amber)
 *   failed     → --err (coral)
 *   running    → --hermes (purple, pulsing)
 *   skipped    → --muted-2 (grey, dashed border)
 *   declined   → --muted-2 (grey, dashed border)
 */
export function TreeNode({ node, selected, onSelect }: TreeNodeProps) {
  const { id, kind, title, outcome, rubricDelta } = node;

  // aria-label combines title + outcome for screen-reader users.
  const ariaLabel = outcome ? `${title} — ${outcome}` : title;

  // CSS classes: base + kind + optional outcome modifier + selected.
  const kindClass = styles[`kind-${kind}`] ?? "";
  const outcomeClass = outcome ? (styles[`outcome-${outcome}`] ?? "") : "";
  const selectedClass = selected ? styles.selected : "";

  return (
    <button
      type="button"
      className={[styles.node, kindClass, outcomeClass, selectedClass]
        .filter(Boolean)
        .join(" ")}
      data-outcome={outcome}
      aria-label={ariaLabel}
      aria-pressed={selected}
      onClick={() => onSelect(id)}
    >
      {/* Title row — always shown */}
      <span className={styles.title}>{title}</span>

      {/* Candidate: category subtitle + rubric delta */}
      {kind === "candidate" && node.candidate && (
        <span className={styles.subtitle}>
          {node.candidate.category}
          {rubricDelta != null && (
            <span className={styles.delta}>
              {rubricDelta >= 0 ? "+" : ""}
              {rubricDelta.toFixed(2)}
            </span>
          )}
        </span>
      )}

      {/* Outcome badge — shows the outcome text for all nodes that have one.
          This is the text that getByText(/promoted/) matches in tests, and
          satisfies the spec requirement that outcome is shown by text not color. */}
      {outcome && (
        <span className={styles.outcomeBadge} aria-hidden="true">
          {outcome}
        </span>
      )}
    </button>
  );
}
