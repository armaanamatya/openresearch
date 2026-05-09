import type { PipelineStage } from "../../lib/dashboard/contracts";

const stageLabels: Record<PipelineStage, string> = {
  plan: "Plan",
  baseline: "Baseline",
  improvement: "Improvement",
  verification: "Verification"
};

export function PipelineOverview({
  projectName,
  runId,
  currentStage,
  stageSummary,
  taskCount,
  feedCount,
  citationCount
}: {
  projectName: string;
  runId: string;
  currentStage: PipelineStage;
  stageSummary: string;
  taskCount: number;
  feedCount: number;
  citationCount: number;
}) {
  return (
    <section className="panel">
      <div className="panel-inner">
        <div className="chip-row">
          <span className="chip">{projectName}</span>
          <span className="chip">Run {runId}</span>
          <span className="chip pill-active">Current stage: {stageLabels[currentStage]}</span>
        </div>
        <h1 style={{ fontSize: "clamp(2rem, 3vw, 3.5rem)", marginBottom: 12 }}>
          Agent topology, evidence, and delegation at a glance.
        </h1>
        <p className="section-copy">{stageSummary}</p>
        <div className="stats-grid">
          <div className="stat-card">
            <span className="stat-value">{taskCount}</span>
            <span className="stat-label">Tracked tasks</span>
          </div>
          <div className="stat-card">
            <span className="stat-value">{feedCount}</span>
            <span className="stat-label">Stream events</span>
          </div>
          <div className="stat-card">
            <span className="stat-value">{citationCount}</span>
            <span className="stat-label">Unique citations</span>
          </div>
          <div className="stat-card">
            <span className="stat-value">PRD</span>
            <span className="stat-label">Contract source</span>
          </div>
        </div>
      </div>
    </section>
  );
}
