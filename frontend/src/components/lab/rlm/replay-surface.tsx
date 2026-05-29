"use client";

import { useEffect, useMemo, useState, type ReactNode } from "react";
import { useSearchParams } from "next/navigation";

import { isRlmEvent, type RlmDashboardEvent } from "@/lib/events/rlm-events";
import { useReplayDriver } from "@/hooks/use-replay-driver";
import { RlmLab } from "./rlm-lab";
import { ReplayControls } from "./replay-controls";

interface ReplayMeta {
  count: number;
  earliestTs: string | null;
  latestTs: string | null;
}

/**
 * URL-gated replay surface. When the lab URL carries ?replay=<projectId>, the lab
 * renders a completed run's recorded timeline through the SAME <RlmLab> components,
 * driven by a scrubbable cursor instead of the live SSE stream. Mirrors the
 * ?rlmFixture=1 precedent — pure visualization, never touches a live run.
 */
export function RlmReplayContent({ children }: { children: ReactNode }) {
  const searchParams = useSearchParams();
  const replayProjectId = searchParams?.get("replay") ?? null;
  if (replayProjectId) {
    return <ReplaySurface projectId={replayProjectId} />;
  }
  return <>{children}</>;
}

function Notice({ text }: { text: string }) {
  return <div className="content" style={{ padding: 24, opacity: 0.8 }}>{text}</div>;
}

interface ReplayData {
  projectId: string;
  events: RlmDashboardEvent[];
  meta: ReplayMeta;
}

function ReplaySurface({ projectId }: { projectId: string }) {
  // Keyed by projectId so a stale result/error never applies to a newer request,
  // and "loading" is DERIVED (data null or for a previous projectId) rather than
  // set synchronously in the effect body (avoids the cascading-render lint rule).
  const [data, setData] = useState<ReplayData | null>(null);
  const [error, setError] = useState<{ projectId: string; message: string } | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch(`/api/demo/replay-events?projectId=${encodeURIComponent(projectId)}`, { cache: "no-store" })
      .then((r) => r.json())
      .then((d: { events?: unknown[]; metadata?: ReplayMeta }) => {
        if (cancelled) return;
        const raw = Array.isArray(d.events) ? d.events : [];
        setData({
          projectId,
          events: raw.filter(isRlmEvent) as RlmDashboardEvent[],
          meta: d.metadata ?? { count: raw.length, earliestTs: null, latestTs: null },
        });
      })
      .catch((e: unknown) => {
        if (!cancelled) {
          setError({ projectId, message: e instanceof Error ? e.message : "Failed to load replay" });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [projectId]);

  if (error && error.projectId === projectId) {
    return <Notice text={`Replay unavailable for ${projectId}: ${error.message}`} />;
  }
  if (!data || data.projectId !== projectId) {
    return <Notice text={`Loading replay for ${projectId}…`} />;
  }
  if (data.events.length === 0) {
    return <Notice text={`No recorded events to replay for ${projectId}.`} />;
  }
  return <ReplayInner projectId={projectId} events={data.events} meta={data.meta} />;
}

function ReplayInner({
  projectId,
  events,
  meta,
}: {
  projectId: string;
  events: RlmDashboardEvent[];
  meta: ReplayMeta;
}) {
  const driver = useReplayDriver(events);
  const runMeta = useMemo(
    () => ({
      projectId,
      paperTitle: "Run replay",
      paperMeta: `${meta.count} recorded events`,
      // Drives RlmLab's elapsed clock; completedAt present ⇒ it freezes at the real
      // run duration rather than ticking against the wall clock.
      startedAt: meta.earliestTs,
      completedAt: meta.latestTs,
    }),
    [projectId, meta],
  );

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", minHeight: 0 }}>
      <ReplayControls driver={driver} />
      <div style={{ flex: 1, minHeight: 0 }}>
        <RlmLab events={driver.events} runMeta={runMeta} isActive={false} />
      </div>
    </div>
  );
}
