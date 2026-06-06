import "server-only";
import type { LeaderboardRow } from "./types";

function backendUrl(): string {
  return ((process.env.OPENRESEARCH_BACKEND_URL ?? process.env.REPROLAB_BACKEND_URL) ?? "http://127.0.0.1:8000").replace(/\/$/, "");
}

export interface FetchLeaderboardParams {
  orderBy?: string;
  limit?: number;
}

export async function fetchLeaderboardRows(
  params: FetchLeaderboardParams = {}
): Promise<{ rows: LeaderboardRow[]; error: string | null }> {
  const qs = new URLSearchParams();
  if (params.orderBy) qs.set("order_by", params.orderBy);
  if (typeof params.limit === "number") qs.set("limit", String(params.limit));

  const url = `${backendUrl()}/leaderboard${qs.toString() ? `?${qs.toString()}` : ""}`;

  try {
    const resp = await fetch(url, {
      cache: "no-store",
      headers: { accept: "application/json" },
    });
    if (!resp.ok) {
      return { rows: [], error: `Leaderboard unavailable (HTTP ${resp.status}).` };
    }
    const body = (await resp.json()) as LeaderboardRow[];
    return {
      rows: Array.isArray(body) ? body : [],
      error: Array.isArray(body) ? null : "Leaderboard response was not a list.",
    };
  } catch {
    return { rows: [], error: "Leaderboard unavailable." };
  }
}
