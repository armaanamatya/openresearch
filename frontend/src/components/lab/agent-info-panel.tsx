"use client";

import type { LiveDemoRunState } from "@/lib/demo/demo-run-types";
import { usePresentationMode } from "@/lib/presentation-mode";

import { HermesAuditPanel } from "./hermes-audit-panel";
import { ICONS, type IconKey } from "./icons";
import type { NodeState, WorkflowNode } from "./node-config";
import { ScriptPanel } from "./script-panel";
import type { Status } from "./status";
import { StatusPill } from "./status";

import "./agent-info-panel.css";

export function AgentInfoPanel({
  failedNodeId,
  logEntries,
  node,
  run,
  state,
  telemetry
}: {
  failedNodeId: string | null;
  logEntries: Array<{ id: string; msg: string; time: string }>;
  node: WorkflowNode;
  run: LiveDemoRunState;
  state: NodeState;
  telemetry: LiveDemoRunState["telemetry"];
}) {
  const mode = usePresentationMode();
  const agentLabel = mode === "demo" ? node.demo_label : node.internal_label;

  const tones = {
    info: { icBg: "var(--info-soft)", icFg: "#3b48d1" },
    accent: { icBg: "var(--accent-soft)", icFg: "var(--accent-ink)" },
    hermes: { icBg: "var(--hermes-soft)", icFg: "var(--hermes)" },
    neutral: { icBg: "var(--chip)", icFg: "var(--muted)" }
  } as const;
  const tone = tones[node.tone];
  const status: Status =
    failedNodeId === node.id
      ? "attention"
      : state === "done"
        ? "completed"
        : state === "running"
          ? node.tone === "hermes"
            ? "auditing"
            : "running"
          : "queued";

  return (
    <div>
      <div className="agent-head">
        <div className="agent-icon" style={{ background: tone.icBg, color: tone.icFg }}>
          {ICONS[node.icon as IconKey]}
        </div>
        <div>
          <div className="eyebrow">Agent</div>
          <div className="agent-name">{agentLabel}</div>
        </div>
      </div>
      <StatusPill status={status} />
      <div className="agent-section">
        <div className="eyebrow">Task</div>
        <div className="agent-task">{node.step}</div>
        <div className="agent-role">{node.role}</div>
      </div>
      <div className="agent-detail">{node.detail}</div>
      {run.payload?.summary.stage ? (
        <div className="agent-section">
          <div className="eyebrow">Backend stage</div>
          <div className="agent-task">{run.payload.summary.stage}</div>
        </div>
      ) : null}
      {telemetry && telemetry.length > 0 ? (
        <div className="agent-section">
          <div className="eyebrow">Telemetry</div>
          <div className="telemetry-list">
            {telemetry.map((record, index) => (
              <div key={`${record.agent_id ?? "agent"}-${index}`} className="telemetry-row">
                <span className="telemetry-name">{record.agent_id ?? "agent"}</span>
                <span className="telemetry-meta">
                  {record.duration_seconds ? `${record.duration_seconds.toFixed(1)}s` : "active"}
                </span>
              </div>
            ))}
          </div>
        </div>
      ) : null}
      {state === "running" ? (
        <div className="agent-section">
          <div className="eyebrow">Progress</div>
          <div className="agent-progress">
            <div
              className="wf-bar"
              style={{ background: node.tone === "hermes" ? "var(--hermes)" : "var(--accent)" }}
            />
          </div>
        </div>
      ) : null}
      {logEntries.length > 0 ? (
        <div className="agent-section">
          <div className="eyebrow">Latest log</div>
          <ul className="agent-log-list">
            {logEntries.slice(0, 6).map((entry) => (
              <li key={entry.id} className="agent-log-item">
                <span className="mono agent-log-time">{entry.time}</span>
                <span className="agent-log-msg">{entry.msg}</span>
              </li>
            ))}
          </ul>
        </div>
      ) : null}
      {node.kind === "audit" ? <HermesAuditPanel run={run} /> : null}
      {node.kind === "report" ? <ScriptPanel run={run} /> : null}
    </div>
  );
}
