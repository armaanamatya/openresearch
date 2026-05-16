"use client";

import { useEffect, useRef, useState } from "react";
import styles from "./library-table.module.css";

const STATUS_OPTIONS: { id: string; label: string; value: string | undefined }[] = [
  { id: "all", label: "All", value: undefined },
  { id: "running", label: "Running", value: "running" },
  { id: "completed", label: "Completed", value: "completed" },
  { id: "failed", label: "Failed", value: "failed" }
];

export interface LibraryFiltersValue {
  status?: string;
  q?: string;
  order_by: string;
}

/**
 * Sits above the table. Owns the debounce timer for the text input so
 * keystrokes don't fire a fetch on every character — the parent only
 * sees a committed value after 300ms of quiet. Status pills and the
 * sort dropdown commit immediately (they aren't typed).
 */
export function LibraryFilters({
  value,
  onChange
}: {
  value: LibraryFiltersValue;
  onChange: (next: LibraryFiltersValue) => void;
}) {
  const [query, setQuery] = useState(value.q ?? "");
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Keep the controlled input synced when the parent resets it (e.g.,
  // via URL navigation). Without this, an external change to value.q
  // would be invisible inside this component.
  useEffect(() => {
    setQuery(value.q ?? "");
  }, [value.q]);

  useEffect(() => {
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, []);

  const commitQuery = (next: string) => {
    setQuery(next);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      const trimmed = next.trim();
      onChange({ ...value, q: trimmed === "" ? undefined : trimmed });
    }, 300);
  };

  return (
    <div className={styles.filters}>
      <div className={styles.pills} role="tablist" aria-label="Status filter">
        {STATUS_OPTIONS.map((opt) => {
          const active = (value.status ?? undefined) === opt.value;
          return (
            <button
              key={opt.id}
              type="button"
              role="tab"
              aria-selected={active}
              className={`${styles.pill}${active ? ` ${styles.pillActive}` : ""}`}
              onClick={() => onChange({ ...value, status: opt.value })}
            >
              {opt.label}
            </button>
          );
        })}
      </div>
      <input
        type="search"
        className={styles.search}
        placeholder="Search source label…"
        value={query}
        onChange={(e) => commitQuery(e.target.value)}
        aria-label="Search runs by source label"
      />
      <select
        className={styles.sort}
        value={value.order_by}
        onChange={(e) => onChange({ ...value, order_by: e.target.value })}
        aria-label="Sort by"
      >
        <option value="updated_at">Sort: Updated</option>
        <option value="completed_at">Sort: Completed</option>
      </select>
    </div>
  );
}
