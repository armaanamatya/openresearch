"use client";

import { useCallback, useEffect, useState, type ReactNode } from "react";

import styles from "./collapsible-panel.module.css";

interface CollapsiblePanelProps {
  /** Stable key used to persist the collapsed state in localStorage. */
  storageKey: string;
  /** Always-visible label in the header strip. */
  title: string;
  /**
   * Compact summary rendered in the header in BOTH states (e.g. the headline
   * score / counts), so the key numbers stay visible even when minimized.
   */
  summary?: ReactNode;
  /**
   * Collapsed state used for the first (SSR + pre-hydration) render, before the
   * persisted preference is read. Keeping it deterministic avoids a hydration
   * mismatch — the stored value is applied in an effect on mount.
   */
  defaultCollapsed?: boolean;
  /** Optional test id forwarded to the panel root. */
  testId?: string;
  children: ReactNode;
}

const STORAGE_PREFIX = "reprolab:panel-collapsed:";

function Chevron() {
  return (
    <svg
      className={styles.chevronSvg}
      width="10"
      height="10"
      viewBox="0 0 10 10"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M3.5 2L7 5 3.5 8" />
    </svg>
  );
}

/**
 * CollapsiblePanel — a minimizable, height-bounded wrapper for the stacked
 * score panels (scorecard, leaf-score breakdown) in the RLM lab.
 *
 * Two invariants it enforces so the constellation graph stays visible:
 *  - the body is capped (`max-height` + internal scroll) so a long leaf list
 *    never grows to consume the whole column;
 *  - the body unmounts when collapsed, reclaiming all of its vertical space.
 *
 * Collapse state is persisted per `storageKey` so the user's choice sticks
 * across runs and reloads.
 */
export function CollapsiblePanel({
  storageKey,
  title,
  summary,
  defaultCollapsed = false,
  testId,
  children,
}: CollapsiblePanelProps) {
  const [collapsed, setCollapsed] = useState(defaultCollapsed);

  // Hydrate the persisted preference on mount (client-only). The first render
  // always uses `defaultCollapsed` so server and client markup agree — reading
  // localStorage in a lazy initializer instead would risk a hydration mismatch.
  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(STORAGE_PREFIX + storageKey);
      if (raw !== "1" && raw !== "0") return;
      // eslint-disable-next-line react-hooks/set-state-in-effect -- one-time hydration of a persisted UI pref; the SSR-safe alternative to a lazy initializer.
      setCollapsed(raw === "1");
    } catch {
      /* localStorage unavailable (private mode / SSR) — keep the default. */
    }
  }, [storageKey]);

  const toggle = useCallback(() => {
    setCollapsed((c) => {
      const next = !c;
      try {
        window.localStorage.setItem(STORAGE_PREFIX + storageKey, next ? "1" : "0");
      } catch {
        /* ignore persistence failures — the toggle still works in-session. */
      }
      return next;
    });
  }, [storageKey]);

  const bodyId = `panel-body-${storageKey.replace(/[^a-z0-9]+/gi, "-")}`;

  return (
    <section
      className={styles.root}
      data-collapsed={collapsed ? "true" : "false"}
      data-testid={testId}
    >
      <button
        type="button"
        className={styles.header}
        aria-expanded={!collapsed}
        aria-controls={bodyId}
        onClick={toggle}
      >
        <span className={styles.chevron} data-collapsed={collapsed ? "true" : "false"}>
          <Chevron />
        </span>
        <span className={styles.title}>{title}</span>
        {summary != null && <span className={styles.summary}>{summary}</span>}
        <span className={styles.spacer} />
        <span className={styles.action}>{collapsed ? "Show" : "Minimize"}</span>
      </button>
      {!collapsed && (
        <div id={bodyId} className={styles.body}>
          {children}
        </div>
      )}
    </section>
  );
}
