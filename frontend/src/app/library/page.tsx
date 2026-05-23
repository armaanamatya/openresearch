import { fetchRecentRunsResult, fetchRunListResult } from "@/lib/runs/server-list";
import { LibraryShell } from "@/components/library/library-shell";

// The library page is a server component: it hydrates the table with
// an initial dataset so first paint shows real rows (no spinner-then-pop).
// Subsequent filter / search interactions go through /api/runs/list on
// the client, so the server-rendered set is only the seed.
export const dynamic = "force-dynamic";

export default async function LibraryPage({
  searchParams
}: {
  searchParams: Promise<{ status?: string; q?: string; order_by?: string }>;
}) {
  const params = await searchParams;
  const [runResult, recentResult] = await Promise.all([
    fetchRunListResult({
      limit: 100,
      status: params.status,
      q: params.q,
      order_by: params.order_by ?? "updated_at"
    }),
    fetchRecentRunsResult(8)
  ]);
  return (
    <LibraryShell
      initialRuns={runResult.runs}
      initialError={runResult.error}
      initialParams={{
        status: params.status,
        q: params.q,
        order_by: params.order_by ?? "updated_at"
      }}
      recents={recentResult.runs}
      recentsError={recentResult.error}
    />
  );
}
