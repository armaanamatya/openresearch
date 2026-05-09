import type { PipelineStage, TaskStatus } from "../../lib/dashboard/contracts";
import type { TopologyEdge, TopologyNode } from "../../lib/dashboard/normalize";

const stageCopy: Record<PipelineStage, string> = {
  plan: "Setup, orchestration, and initial run contracts.",
  baseline: "Environment recovery and baseline execution tasks.",
  improvement: "Delegated path agents and branch-local investigations.",
  verification: "Verifier reviews and promotion gates."
};

const statusLabel: Record<TaskStatus, string> = {
  created: "Created",
  context_prepared: "Context prepared",
  running: "Running",
  artifact_submitted: "Artifact submitted",
  verification_pending: "Verification pending",
  verified: "Verified",
  completed: "Completed",
  failed: "Failed",
  blocked_requires_human: "Blocked"
};

export function TopologyView({
  currentStage,
  nodes,
  edges,
  selectedTaskId,
  onSelectTask
}: {
  currentStage: PipelineStage;
  nodes: TopologyNode[];
  edges: TopologyEdge[];
  selectedTaskId: string;
  onSelectTask: (taskId: string) => void;
}) {
  const nodeById = Object.fromEntries(nodes.map((node) => [node.taskId, node]));

  return (
    <section className="panel">
      <div className="panel-inner">
        <h2 className="section-title">Agent topology</h2>
        <p className="section-copy">
          The graph is derived from task records and parent-child delegation metadata, not
          hand-authored view state. Selecting a node drives the task detail and citation
          panels.
        </p>
        <ul className="topology-stage-list" style={{ marginTop: 18 }}>
          {(["plan", "baseline", "improvement", "verification"] as PipelineStage[]).map(
            (stage) => (
              <li key={stage} className="topology-stage">
                <div className="chip-row">
                  <span className={`pill ${currentStage === stage ? "pill-active" : ""}`}>
                    {stage}
                  </span>
                </div>
                <p className="section-copy" style={{ marginTop: 10 }}>
                  {stageCopy[stage]}
                </p>
              </li>
            )
          )}
        </ul>
        <div className="topology-map" role="img" aria-label="Agent topology map">
          <svg aria-hidden="true">
            {edges.map((edge) => {
              const from = nodeById[edge.from];
              const to = nodeById[edge.to];

              if (!from || !to) {
                return null;
              }

              return (
                <line
                  key={`${edge.from}-${edge.to}`}
                  x1={from.x}
                  y1={from.y}
                  x2={to.x}
                  y2={to.y}
                  stroke="rgba(14, 124, 102, 0.35)"
                  strokeWidth="2"
                  strokeLinecap="round"
                />
              );
            })}
          </svg>
          {nodes.map((node) => (
            <button
              key={node.taskId}
              className="topology-node"
              data-selected={node.taskId === selectedTaskId}
              style={{ left: `${node.x}px`, top: `${node.y}px` }}
              onClick={() => onSelectTask(node.taskId)}
              type="button"
            >
              <div className="node-kicker">{node.agentId}</div>
              <div className="node-title">{node.title}</div>
              <div className="chip-row">
                <span className="chip">{node.stage}</span>
                <span className="chip">
                  <span className={`status-dot status-${node.status}`} />
                  {statusLabel[node.status]}
                </span>
              </div>
              <p className="section-copy" style={{ marginTop: 10, fontSize: "0.92rem" }}>
                {node.summary}
              </p>
            </button>
          ))}
        </div>
      </div>
    </section>
  );
}
