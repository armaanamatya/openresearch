"use client";

import { use, useMemo } from "react";

import "@/styles/autoresearch-tokens.css";
import { useRun } from "@/hooks/use-run";
import { useRlmRun } from "@/hooks/use-rlm-run";
import { isRlmEvent, type RlmDashboardEvent } from "@/lib/events/rlm-events";
import {
  foldSpecPhase,
  shouldShowSpecStepper,
  INITIAL_SPEC_PHASE_STATE,
} from "@/lib/autoresearch/session-events";
import { SessionReasoningView } from "@/components/autoresearch/SessionReasoningView";
import { SpecValidationStepper } from "@/components/autoresearch/SpecValidationStepper";
import { GpuStatusStrip } from "@/components/autoresearch/GpuStatusStrip";
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

export interface SessionRouteContentProps {
  runId: string;
  /** The already-`isRlmEvent`-filtered stream for this run. */
  events: RlmDashboardEvent[];
  /** The run's startedAt, once the live stream has stamped one. */
  startedAt?: string | null;
}

/**
 * The route body: the SessionRail + the screen-D gate (spec-validation
 * stepper vs the live reasoning view). Split out from the default page
 * export — which owns the `use(params)` + `useRun` live subscription — so
 * the stepper-vs-view swap is unit-testable from a plain event array,
 * without mocking the SSE transport or unwrapping the params Promise.
 *
 * Screen-D gate: during the spec-generation + external-validation phase
 * (and before the root loop's first reasoning iteration) it shows the
 * SpecValidationStepper; otherwise the live reasoning log. A run with NO
 * spec events stays `idle` and goes straight to the reasoning/"Waiting…"
 * state (the non-autonomous path is unchanged). The stepper's own redirect
 * to /sessions/<runId> is a harmless same-URL no-op — we're already here.
 */
export function SessionRouteContent({ runId, events, startedAt }: SessionRouteContentProps) {
  const sessions: SessionRailSession[] = useMemo(
    () => [{ id: runId, title: sessionTitleFor(startedAt), active: true }],
    [runId, startedAt]
  );

  const rlmState = useRlmRun(events);
  const specPhase = useMemo(
    () => events.reduce(foldSpecPhase, INITIAL_SPEC_PHASE_STATE),
    [events]
  );
  const showStepper = shouldShowSpecStepper(specPhase, rlmState.iterations.length);

  return (
    <div className={`autoresearch ${styles.page}`}>
      <SessionRail projectName={runId} sessions={sessions} />
      <div className={styles.main}>
        {/* The gpu_resolved plan is already folded into RlmRunState by the
         * SAME useRlmRun(events) call above (rlmState.gpuPlan) — no new SSE
         * subscription or shared-reducer edit needed to surface it here. */}
        <GpuStatusStrip gpuPlan={rlmState.gpuPlan} status={rlmState.status} />
        {showStepper ? (
          <SpecValidationStepper runId={runId} events={events} />
        ) : (
          <SessionReasoningView runId={runId} events={events} className={styles.mainInner} />
        )}
      </div>
    </div>
  );
}

/**
 * The live agentic-reasoning session route (alphaXiv screens D → `_5`).
 * Subscribes to the run's SSE stream by reusing use-run.ts's existing
 * EventSource/dashboardEvents transport verbatim (the same mechanism
 * LabShell uses for the dark lab) — no new transport, no polling loop —
 * then delegates rendering to SessionRouteContent.
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

  return <SessionRouteContent runId={runId} events={events} startedAt={run?.startedAt} />;
}
