"use client";

import type { RecentRunSummary, RunSummary } from "@/lib/runs/server-list";
import { LabSidebar } from "@/components/lab/lab-sidebar";
import { LibraryTable } from "./library-table";

/**
 * The /library page's outer chrome. Wraps the table in the same shell
 * the lab uses (sidebar + content), so navigation between Lab and
 * Library doesn't reflow the page. The sidebar is a client component,
 * so this shell must be too — the page itself stays server-side.
 */
export function LibraryShell({
  initialRuns,
  initialError,
  initialParams,
  recents,
  recentsError
}: {
  initialRuns: RunSummary[];
  initialError?: string | null;
  initialParams: { status?: string; q?: string; order_by?: string };
  recents: RecentRunSummary[];
  recentsError?: string | null;
}) {
  return (
    <div className="openResearch">
      <div className="layout">
        <LabSidebar
          active="library"
          recents={recents}
          recentsError={recentsError}
        />
        <main className="content">
          <LibraryTable
            initialRuns={initialRuns}
            initialError={initialError}
            initialParams={initialParams}
          />
        </main>
      </div>
    </div>
  );
}
