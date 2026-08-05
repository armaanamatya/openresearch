import type { LeaderboardRow } from "./types";

/**
 * Filter leaderboard rows for the Recent Runs home panel:
 * - Exclude rows with status === "interrupted" (orphan-swept noise)
 * - Return at most `cap` rows (default 8)
 */
export function filterRecentRows(rows: LeaderboardRow[], cap = 8): LeaderboardRow[] {
  return rows.filter((r) => r.status !== "interrupted").slice(0, cap);
}

/**
 * Interrupted rows are excluded from the main Recent Runs list above (most
 * are orphan-swept noise), but a genuinely interrupted run is exactly the
 * case an operator may want to resume — hiding it entirely gives no path
 * back to it from the UI. This is a SEPARATE, additive list (never merged
 * into `filterRecentRows`'s output) so the main panel's "no interrupted
 * rows" contract stays intact; callers render it as its own small section.
 */
export function filterInterruptedRows(rows: LeaderboardRow[], cap = 3): LeaderboardRow[] {
  return rows.filter((r) => r.status === "interrupted").slice(0, cap);
}
