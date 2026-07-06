import "@/styles/autoresearch-tokens.css";

import { backendBaseUrl } from "@/lib/demo/server-run";
import { PaperLanding } from "@/components/autoresearch/PaperLanding";
import { RepoConfirm } from "@/components/autoresearch/RepoConfirm";
import styles from "./page.module.css";

// This page reads `searchParams` (the ?confirm=1 stage toggle) and does a
// live backend fetch on the confirm stage — never statically cached.
export const dynamic = "force-dynamic";

interface RepoSuggestionResponse {
  repo_url?: string | null;
}

// GET /papers/{arxiv_id}/repo (T3b) — fetched server-side, the same pattern
// LabPage uses for fetchAuthStatus, so the browser never needs to reach the
// FastAPI backend directly (there is no CORS layer — see CLAUDE.md's "one
// image, two processes" note). Fail-soft end to end: any network error,
// non-2xx, or empty suggestion collapses to "" and RepoField just renders
// blank and editable — the run's own in-loop resolver still finds the repo.
async function fetchRepoSuggestion(arxivId: string): Promise<string> {
  try {
    const response = await fetch(
      `${backendBaseUrl()}/papers/${encodeURIComponent(arxivId)}/repo`,
      { cache: "no-store" }
    );
    if (!response.ok) return "";
    const data = (await response.json()) as RepoSuggestionResponse;
    return data.repo_url ?? "";
  } catch {
    return "";
  }
}

export default async function AbsPage({
  params,
  searchParams,
}: {
  params: Promise<{ arxivId: string }>;
  searchParams: Promise<{ confirm?: string | string[] }>;
}) {
  const [{ arxivId }, sp] = await Promise.all([params, searchParams]);
  const confirmRaw = Array.isArray(sp.confirm) ? sp.confirm[0] : sp.confirm;
  const isConfirmStage = confirmRaw === "1";

  // Only fetch the repo suggestion when it's actually needed — the
  // paper-landing stage has nowhere to show it yet.
  const initialRepoUrl = isConfirmStage ? await fetchRepoSuggestion(arxivId) : "";

  return (
    <div className={`autoresearch ${styles.page}`}>
      {isConfirmStage ? (
        <RepoConfirm arxivId={arxivId} initialRepoUrl={initialRepoUrl} />
      ) : (
        <PaperLanding arxivId={arxivId} />
      )}
    </div>
  );
}
