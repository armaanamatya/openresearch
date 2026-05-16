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
  initialParams,
  recents
}: {
  initialRuns: RunSummary[];
  initialParams: { status?: string; q?: string; order_by?: string };
  recents: RecentRunSummary[];
}) {
  return (
    <div className="reproLab">
      <div className="layout">
        <LabSidebar
          active="library"
          onBrandClick={() => {
            // Brand click returns to the lab — matches the LabShell
            // behavior, where it doubles as a "fresh session" affordance.
            if (typeof window !== "undefined") window.location.href = "/lab";
          }}
          recents={recents}
        />
        <main className="content">
          <LibraryTable initialRuns={initialRuns} initialParams={initialParams} />
        </main>
      </div>
    </div>
  );
}
