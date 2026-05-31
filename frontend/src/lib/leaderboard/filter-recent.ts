import type { LeaderboardRow } from "./types";

/**
 * Filter leaderboard rows for the Recent Runs home panel:
 * - Exclude rows with status === "interrupted" (orphan-swept noise)
 * - Return at most `cap` rows (default 8)
 */
export function filterRecentRows(rows: LeaderboardRow[], cap = 8): LeaderboardRow[] {
  return rows.filter((r) => r.status !== "interrupted").slice(0, cap);
}
