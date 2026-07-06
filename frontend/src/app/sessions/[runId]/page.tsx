"use client";

import { use, useMemo } from "react";

import "@/styles/autoresearch-tokens.css";
import { useRun } from "@/hooks/use-run";
import { isRlmEvent } from "@/lib/events/rlm-events";
import { SessionReasoningView } from "@/components/autoresearch/SessionReasoningView";
import { SessionRail, type SessionRailSession } from "@/components/autoresearch/ui/SessionRail";
import styles from "./page.module.css";

/** A short, honest session label derived from the run's own startedAt
 * timestamp — "New session" until the backend has stamped one (a freshly
 * queued run). Never fabricates a title from data we don't have. */
function sessionTitleFor(startedAt: string | null | undefined): string {
  if (!startedAt) return "New session";
  const date = new Date(startedAt);
  if (Number.isNaN(date.getTime())) return "New session";
  return `Session — ${date.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  })}`;
}

/**
 * The live agentic-reasoning session route (alphaXiv screen `_5`).
 * Subscribes to the run's SSE stream by reusing use-run.ts's existing
 * EventSource/dashboardEvents transport verbatim (the same mechanism
 * LabShell uses for the dark lab) — no new transport, no polling loop.
 *
 * A "use client" page (not the async-Server-Component shape T11's /abs
 * page uses) because it needs to open the live subscription itself; Next
 * still hands params as a Promise here, unwrapped via React's `use()`.
 */
export default function SessionPage({
  params,
}: {
  params: Promise<{ runId: string }>;
}) {
  const { runId } = use(params);

  // A stub keyed to THIS route's runId. useRun()'s auto-resume effect
  // trusts a supplied initialRun and skips its own ?projectId=/localStorage
  // fallback entirely — which would otherwise risk restoring a DIFFERENT
  // run than the one named in this page's URL (e.g. a stale
  // "last launched" pointer). The real status/paperTitle/etc. arrive
  // moments later over the same live EventSource this hook opens, since
  // both "queued" and "running" trigger it.
  const initialRun = useMemo(
    () => ({
      projectId: runId,
      outputDir: "",
      runMode: "rlm" as const,
      status: "queued" as const,
      payload: null,
      log: "",
    }),
    [runId]
  );

  const { run, dashboardEvents } = useRun(initialRun);
  const events = useMemo(() => dashboardEvents.filter(isRlmEvent), [dashboardEvents]);

  const sessions: SessionRailSession[] = useMemo(
    () => [{ id: runId, title: sessionTitleFor(run?.startedAt), active: true }],
    [runId, run?.startedAt]
  );

  return (
    <div className={`autoresearch ${styles.page}`}>
      <SessionRail projectName={runId} sessions={sessions} />
      <SessionReasoningView runId={runId} events={events} className={styles.main} />
    </div>
  );
}
