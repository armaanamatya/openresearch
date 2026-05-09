"use client";

import type {
  AgentNode,
  ApprovalStatus,
  DashboardEventAdapter,
  DataPanel,
  MessageEntry,
  ProgressEntry,
  ReasoningEntry
} from "@/lib/events/contract";
import { createMockEventAdapter } from "@/lib/events/mock-event-adapter";

import { useDashboardState } from "./use-dashboard-state";

const defaultAdapter = createMockEventAdapter();

interface DashboardShellProps {
  adapter?: DashboardEventAdapter;
}

function statusTone(
  status: AgentNode["status"] | ProgressEntry["status"] | ApprovalStatus
): string {
  switch (status) {
    case "completed":
    case "passed":
    case "approved":
      return "status-positive";
    case "failed":
    case "rejected":
      return "status-negative";
    case "caveat":
    case "waiting":
      return "status-caution";
    default:
      return "status-neutral";
  }
}

function AgentCard({ agent }: { agent: AgentNode }) {
  return (
    <article className="agent-card">
      <div className="card-heading">
        <div>
          <p className="eyebrow">{agent.type}</p>
          <h3>{agent.label}</h3>
        </div>
        <span className={`status-pill ${statusTone(agent.status)}`}>{agent.status}</span>
      </div>
      <p className="card-copy">{agent.currentTask}</p>
      <dl className="agent-meta">
        <div>
          <dt>Parent</dt>
          <dd>{agent.parentId ?? "root"}</dd>
        </div>
        <div>
          <dt>Outputs</dt>
          <dd>{agent.outputTargetIds.join(", ") || "None yet"}</dd>
        </div>
        <div>
          <dt>Context</dt>
          <dd>{agent.contextVariables.join(", ") || "Pending"}</dd>
        </div>
      </dl>
    </article>
  );
}

function ReasoningItem({ item }: { item: ReasoningEntry }) {
  return (
    <li className="stream-item">
      <div className="stream-meta">
        <span>{item.agentLabel}</span>
        <span>{item.stepType}</span>
      </div>
      <h3>{item.title}</h3>
      <p>{item.detail}</p>
      <div className="tag-row">
        {item.citations.map((citation) => (
          <span className="citation-tag" key={citation.id}>
            {citation.label}
          </span>
        ))}
      </div>
    </li>
  );
}

function MessageItem({ item }: { item: MessageEntry }) {
  return (
    <li className="stream-item">
      <div className="stream-meta">
        <span>{item.fromAgentId}</span>
        <span>{item.timestamp}</span>
      </div>
      <h3>{item.summary}</h3>
      <p>{item.detail}</p>
      <p className="muted">To: {item.toAgentId}</p>
    </li>
  );
}

function ProgressCard({ item }: { item: ProgressEntry }) {
  return (
    <article className="progress-card">
      <div className="card-heading">
        <div>
          <p className="eyebrow">{item.stage}</p>
          <h3>{item.stage.toUpperCase()}</h3>
        </div>
        <span className={`status-pill ${statusTone(item.status)}`}>{item.status}</span>
      </div>
      <p className="card-copy">{item.detail}</p>
    </article>
  );
}

function PanelCard({ panel }: { panel: DataPanel }) {
  return (
    <article className="panel-card">
      <p className="eyebrow">{panel.title}</p>
      <h3>{panel.summary}</h3>
      <ul>
        {panel.items.map((item) => (
          <li key={item}>{item}</li>
        ))}
      </ul>
    </article>
  );
}

export function DashboardShell({ adapter }: DashboardShellProps) {
  const snapshot = useDashboardState(adapter ?? defaultAdapter);

  return (
    <main className="dashboard-shell">
      <section className="hero-panel">
        <div>
          <p className="kicker">ReproLab operations</p>
          <h1>Agent Lab</h1>
          <p className="hero-copy">
            A stable frontend shell for watching the reproduction pipeline reason,
            cite evidence, and coordinate work before the live backend stream exists.
          </p>
        </div>
        <div className="hero-stats">
          <div>
            <span className="stat-value">{snapshot.agents.length}</span>
            <span className="stat-label">agents tracked</span>
          </div>
          <div>
            <span className="stat-value">{snapshot.citations.length}</span>
            <span className="stat-label">citations linked</span>
          </div>
          <div>
            <span className="stat-value">{snapshot.approvals.length}</span>
            <span className="stat-label">approvals visible</span>
          </div>
        </div>
      </section>

      <section className="dashboard-grid">
        <section className="panel panel-wide">
          <div className="panel-header">
            <div>
              <p className="eyebrow">Live topology</p>
              <h2>Topology</h2>
            </div>
            <p className="muted">Parent-child agent relationships and context flow.</p>
          </div>
          <div className="agent-grid">
            {snapshot.agents.map((agent) => (
              <AgentCard agent={agent} key={agent.id} />
            ))}
          </div>
        </section>

        <section className="panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow">Evidence in motion</p>
              <h2>Reasoning Stream</h2>
            </div>
          </div>
          <ul className="stream-list">
            {snapshot.reasoning.map((item) => (
              <ReasoningItem item={item} key={item.id} />
            ))}
          </ul>
        </section>

        <section className="panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow">Coordination feed</p>
              <h2>Messages</h2>
            </div>
          </div>
          <ul className="stream-list">
            {snapshot.messages.map((item) => (
              <MessageItem item={item} key={item.id} />
            ))}
          </ul>
        </section>

        <section className="panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow">Trust chain</p>
              <h2>Citations</h2>
            </div>
          </div>
          <ul className="citation-list">
            {snapshot.citations.map((citation) => (
              <li key={citation.id}>
                <div className="card-heading">
                  <strong>{citation.label}</strong>
                  <span className={`status-pill ${statusTone("running")}`}>
                    {citation.trustLevel}
                  </span>
                </div>
                <p>{citation.excerpt}</p>
                <p className="muted">{citation.sourceType}</p>
              </li>
            ))}
          </ul>
        </section>

        <section className="panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow">Gate health</p>
              <h2>Pipeline Progress</h2>
            </div>
          </div>
          <div className="progress-grid">
            {snapshot.progress.map((item) => (
              <ProgressCard item={item} key={item.stage} />
            ))}
          </div>
          <div className="approval-callout">
            <p className="eyebrow">Pending approvals</p>
            {snapshot.approvals.map((approval) => (
              <article className="approval-card" key={approval.id}>
                <div className="card-heading">
                  <h3>{approval.title}</h3>
                  <span className={`status-pill ${statusTone("waiting")}`}>
                    {approval.status}
                  </span>
                </div>
                <p>{approval.detail}</p>
              </article>
            ))}
          </div>
        </section>

        <section className="panel panel-wide">
          <div className="panel-header">
            <div>
              <p className="eyebrow">Research surface</p>
              <h2>Data Panels</h2>
            </div>
          </div>
          <div className="panel-grid">
            {snapshot.dataPanels.map((panel) => (
              <PanelCard key={panel.id} panel={panel} />
            ))}
          </div>
        </section>
      </section>
    </main>
  );
}
