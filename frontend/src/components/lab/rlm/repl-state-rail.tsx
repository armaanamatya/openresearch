"use client";

import type { CSSProperties } from "react";
import type { RlmRunState } from "../../../hooks/use-rlm-run";
import styles from "./repl-state-rail.module.css";

interface ReplStateRailProps {
  variables: RlmRunState["variables"];
  primitives: string[];
  collapsed: boolean;
  onToggle: () => void;
  style?: CSSProperties;
}

/** Returns true when a variable is "not set" — NoneType or never seen in a real iteration. */
function isUnset(meta: RlmRunState["variables"][string]): boolean {
  return meta.type === "NoneType" || meta.firstSeenIteration === 0;
}

/**
 * Human-readable size string.
 * ≥ 1 000 000 → "N.Nm MB"; ≥ 1 000 → "Nk"; 0 → "—"; else byte count.
 */
function fmtSize(bytes: number): string {
  if (bytes === 0) return "—";
  if (bytes >= 1_000_000) return `${(bytes / 1_000_000).toFixed(1)} MB`;
  if (bytes >= 1_000) return `${Math.round(bytes / 1_000)}k`;
  return `${bytes} B`;
}

export function ReplStateRail({
  variables,
  primitives,
  collapsed,
  onToggle,
  style,
}: ReplStateRailProps) {
  const entries = Object.entries(variables);
  const totalCount = entries.length;
  const setCount = entries.filter(([, meta]) => !isUnset(meta)).length;

  const buttonLabel = collapsed ? "Expand REPL state rail" : "Collapse REPL state rail";

  if (collapsed) {
    // Lane Z (2026-05-25): collapsed rail is now a clickable column with a
    // rotated vertical title so users always know what they're collapsing.
    // The chevron sits at the top + the title runs down the rail + the
    // count badge anchors the bottom. The whole thing is clickable.
    return (
      <aside className={`${styles.rail} ${styles.railCollapsed}`} aria-label="REPL state rail" data-testid="repl-state-rail">
        <button
          className={styles.collapsedClickable}
          aria-label={buttonLabel}
          aria-expanded={false}
          onClick={onToggle}
          type="button"
          title="Expand REPL state"
        >
          {/* Chevron-right (expand) at top */}
          <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true" fill="none">
            <polyline points="5,2 9,7 5,12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          <span className={styles.collapsedTitle}>REPL state</span>
          <span
            className={styles.collapsedCount}
            title={`${setCount} of ${totalCount} tracked REPL variables are set.`}
          >
            {setCount}/{totalCount}
          </span>
        </button>
      </aside>
    );
  }

  return (
    <aside className={styles.rail} style={style} aria-label="REPL state rail" data-testid="repl-state-rail">
      {/* Header row — Lane Z: full-width clickable title bar */}
      <button
        className={styles.titleBar}
        onClick={onToggle}
        aria-label={buttonLabel}
        aria-expanded={true}
        type="button"
      >
        <span className={styles.titleBarHeading}>REPL state</span>
        <span
          className={styles.setCount}
          title={`${setCount} of ${totalCount} tracked REPL variables are set. Variables arrive with repl_iteration events.`}
        >
          {setCount}/{totalCount} set
        </span>
        {/* Chevron-left (collapse) at right edge */}
        <svg className={styles.titleBarChevron} width="14" height="14" viewBox="0 0 14 14" aria-hidden="true" fill="none">
          <polyline points="9,2 5,7 9,12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>

      {/* Variables section */}
      <section className={styles.section} aria-label="Variables">
        <p className={styles.sectionLabel}>variables</p>
        {entries.length === 0 ? (
          <p className={styles.empty}>
            no variables yet — they appear after the first REPL iteration
          </p>
        ) : (
          <ul className={styles.varList}>
            {entries.map(([name, meta]) => {
              const unset = isUnset(meta);
              return (
                <li
                  key={name}
                  className={`${styles.varRow}${unset ? ` ${styles.varRowDimmed}` : ""}`}
                >
                  <span className={styles.varName}>{name}</span>
                  <span className={styles.varType}>{meta.type}</span>
                  <span className={styles.varSize}>{fmtSize(meta.size)}</span>
                </li>
              );
            })}
          </ul>
        )}
      </section>

      {/* Primitives section */}
      <section className={styles.section} aria-label="Primitives">
        <p className={styles.sectionLabel}>primitives</p>
        {primitives.length === 0 ? (
          <p className={styles.empty}>
            waiting for primitive calls — paper understanding usually comes first
          </p>
        ) : (
          <ul className={styles.primitiveList}>
            {primitives.map((name) => (
              <li key={name} className={styles.primitiveRow}>
                <span className={styles.primitiveName}>{name}</span>
              </li>
            ))}
          </ul>
        )}
      </section>
    </aside>
  );
}
