"use client";

import { useMemo, useState } from "react";
import type { DashboardSnapshot } from "../../lib/dashboard/contracts";
import { buildDashboardModel } from "../../lib/dashboard/normalize";
import { ActivityFeed } from "./activity-feed";
import { CitationPanel } from "./citation-panel";
import { PipelineOverview } from "./pipeline-overview";
import { TaskDetailPanel } from "./task-detail-panel";
import { TopologyView } from "./topology-view";

export function DashboardShell({ snapshot }: { snapshot: DashboardSnapshot }) {
  const model = useMemo(() => buildDashboardModel(snapshot), [snapshot]);
  const [selectedTaskId, setSelectedTaskId] = useState(model.orderedTasks[0]?.task_id ?? "");
  const [selectedFeedId, setSelectedFeedId] = useState(model.feed[0]?.id ?? "");

  const activeDetail = model.detailsByTaskId[selectedTaskId];
  const activeFeed =
    model.feed.find((item) => item.id === selectedFeedId) ?? model.feed[0] ?? null;
  const activeCitations = activeFeed?.citations.length
    ? activeFeed.citations
    : activeDetail?.citations ?? [];

  return (
    <main className="page-shell">
      <section className="hero">
        <span className="eyebrow">Issue 21 dashboard scaffold</span>
        <div className="hero-grid">
          <PipelineOverview
            projectName={snapshot.project_name}
            runId={snapshot.run_id}
            currentStage={snapshot.current_stage}
            stageSummary={snapshot.stage_summary}
            taskCount={model.orderedTasks.length}
            feedCount={model.feed.length}
            citationCount={Object.keys(model.citationsBySourceId).length}
          />
          <section className="panel">
            <div className="panel-inner">
              <h2 className="section-title">Integration boundary</h2>
              <p className="section-copy">
                This branch uses PRD-shaped fixtures and a typed normalizer because the
                upstream frontend shell and backend event stream are not on `main` yet.
                The page structure is intentionally ready to swap from fixture snapshots to
                real SSE or WebSocket payloads without changing the panel components.
              </p>
              <div className="warning-banner">
                Missing dependencies detected: no merged app shell from issue `#20`, no
                backend event stream contract from issues `#16`, `#18`, or `#11`.
              </div>
            </div>
          </section>
        </div>
      </section>

      <section className="content-grid">
        <div className="split-stack">
          <TopologyView
            currentStage={snapshot.current_stage}
            nodes={model.topologyNodes}
            edges={model.topologyEdges}
            selectedTaskId={selectedTaskId}
            onSelectTask={(taskId) => setSelectedTaskId(taskId)}
          />
          <ActivityFeed
            items={model.feed}
            selectedFeedId={selectedFeedId}
            onSelectFeed={(feedId, taskId) => {
              setSelectedFeedId(feedId);
              setSelectedTaskId(taskId);
            }}
          />
        </div>
        <div className="split-stack">
          {activeDetail ? <TaskDetailPanel detail={activeDetail} /> : null}
          <CitationPanel
            citations={activeCitations}
            heading={activeFeed ? activeFeed.title : activeDetail?.task.title ?? "Citations"}
          />
        </div>
      </section>
    </main>
  );
}
