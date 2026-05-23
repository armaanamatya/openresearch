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
    return (
      <aside className={`${styles.rail} ${styles.railCollapsed}`} aria-label="REPL state rail" data-testid="repl-state-rail">
        <button
          className={styles.toggleBtn}
          aria-label={buttonLabel}
          onClick={onToggle}
          type="button"
        >
          {/* Compact icon: two horizontal lines (≡) */}
          <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true" fill="none">
            <line x1="2" y1="4" x2="12" y2="4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
            <line x1="2" y1="8" x2="12" y2="8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
            <line x1="2" y1="12" x2="12" y2="12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
          </svg>
        </button>
        <span className={styles.collapsedCount}>{setCount}/{totalCount}</span>
      </aside>
    );
  }

  return (
    <aside className={styles.rail} style={style} aria-label="REPL state rail" data-testid="repl-state-rail">
      {/* Header row */}
      <div className={styles.header}>
        <span className={styles.heading}>REPL state</span>
        <span className={styles.setCount}>{setCount}/{totalCount} set</span>
        <button
          className={styles.toggleBtn}
          aria-label={buttonLabel}
          onClick={onToggle}
          type="button"
        >
          {/* Chevron-left collapse icon */}
          <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true" fill="none">
            <polyline points="9,2 5,7 9,12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </button>
      </div>

      {/* Variables section */}
      <section className={styles.section} aria-label="Variables">
        <p className={styles.sectionLabel}>variables</p>
        {entries.length === 0 ? (
          <p className={styles.empty}>no variables yet</p>
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
          <p className={styles.empty}>none</p>
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
