import type { TaskDetail } from "../../lib/dashboard/normalize";

export function TaskDetailPanel({ detail }: { detail: TaskDetail }) {
  const { task, lineage, descendants, relatedFeed, citations } = detail;

  return (
    <section className="panel">
      <div className="panel-inner">
        <h2 className="section-title">Task detail and delegation lineage</h2>
        <div className="detail-card">
          <div className="chip-row">
            <span className="pill">{task.agent_id}</span>
            <span className="pill">{task.stage}</span>
            <span className="pill">{task.status}</span>
            <span className="pill">{task.scope}</span>
          </div>
          <h3 className="detail-title" style={{ marginTop: 14 }}>
            {task.title}
          </h3>
          <p className="section-copy">{task.summary}</p>
          <div className="detail-meta">
            <span>Budget: {task.assigned_budget}</span>
            <span>Timeout: {task.assigned_timeout}</span>
            <span>Write scope: {task.write_scope}</span>
          </div>
        </div>

        <div style={{ marginTop: 18 }}>
          <h3 className="section-title">Lineage</h3>
          <div className="lineage-row">
            {lineage.map((entry, index) => (
              <span className="lineage-item" key={entry.task_id}>
                <span className="pill">{entry.agent_id}</span>
                <span>{entry.task_id}</span>
                {index < lineage.length - 1 ? <span className="arrow">-&gt;</span> : null}
              </span>
            ))}
          </div>
        </div>

        <div style={{ marginTop: 18 }}>
          <h3 className="section-title">Delegation footprint</h3>
          <ul className="detail-list">
            <li className="detail-card">
              <strong>{descendants.length}</strong> downstream tasks
            </li>
            <li className="detail-card">
              <strong>{relatedFeed.length}</strong> related feed events
            </li>
            <li className="detail-card">
              <strong>{citations.length}</strong> citations across task + feed
            </li>
          </ul>
        </div>

        <div style={{ marginTop: 18 }}>
          <h3 className="section-title">Structured outputs</h3>
          <div className="detail-card">
            <pre style={{ margin: 0, whiteSpace: "pre-wrap" }}>
              {JSON.stringify(task.structured_outputs, null, 2)}
            </pre>
          </div>
        </div>
      </div>
    </section>
  );
}
