"use client";

import React, { useMemo } from "react";

import "./agent-timeline-rail.css";

export type DashboardLiveEvent = {
  event: string;
  timestamp?: string;
  agentId?: string;
  agentLabel?: string;
  agent?: { id?: string; label?: string; status?: string; currentTask?: string };
  stepType?: string;
  title?: string;
  detail?: string;
  changeType?: string;
  fromAgentId?: string;
  toAgentId?: string;
  variableName?: string;
  summary?: string;
  panel?: { title?: string; summary?: string; overallStatus?: string };
};

interface AgentLive {
  agentId: string;
  label: string;
  status: string;
  currentTask: string;
  timestamp: string;
}

interface ReasoningItem {
  agentId: string;
  agentLabel: string;
  stepType: string;
  title: string;
  detail: string;
  timestamp: string;
}

interface ContextItem {
  agentId: string;
  changeType: string;
  title: string;
  detail: string;
  fromAgentId?: string;
  toAgentId?: string;
  timestamp: string;
}

const MAX_LIVE_AGENTS = 8;
const MAX_REASONING = 12;
const MAX_CONTEXT = 6;
const MAX_DECISIONS = 10;

function shortTime(value: string | undefined): string {
  if (!value) return "--:--";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return "--:--";
  return parsed.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function digestEvents(events: DashboardLiveEvent[]): {
  liveAgents: AgentLive[];
  reasoning: ReasoningItem[];
  context: ContextItem[];
} {
  const agentByLatest = new Map<string, AgentLive>();
  const reasoning: ReasoningItem[] = [];
  const context: ContextItem[] = [];

  for (const evt of events) {
    if (evt.event === "agent_started" || evt.event === "agent_completed" || evt.event === "agent_failed") {
      const id = evt.agent?.id ?? evt.agentId;
      if (!id) continue;
      agentByLatest.set(id, {
        agentId: id,
        label: evt.agent?.label ?? id,
        status: evt.agent?.status ?? evt.event.replace("agent_", ""),
        currentTask: evt.agent?.currentTask ?? "",
        timestamp: evt.timestamp ?? ""
      });
    } else if (evt.event === "agent_reasoning_step") {
      reasoning.push({
        agentId: evt.agentId ?? "",
        agentLabel: evt.agentLabel ?? evt.agentId ?? "Agent",
        stepType: evt.stepType ?? "step",
        title: evt.title ?? "",
        detail: evt.detail ?? "",
        timestamp: evt.timestamp ?? ""
      });
    } else if (evt.event === "shared_state_updated" || evt.event === "context_enrichment") {
      context.push({
        agentId: evt.agentId ?? "",
        changeType: evt.changeType ?? evt.event,
        title: evt.title ?? evt.variableName ?? "",
        detail: evt.detail ?? evt.summary ?? "",
        fromAgentId: evt.fromAgentId,
        toAgentId: evt.toAgentId,
        timestamp: evt.timestamp ?? ""
      });
    }
  }

  const liveAgents = Array.from(agentByLatest.values()).slice(-MAX_LIVE_AGENTS).reverse();
  return {
    liveAgents,
    reasoning: reasoning.slice(-MAX_REASONING).reverse(),
    context: context.slice(-MAX_CONTEXT).reverse()
  };
}

function statusDot(status: string): string {
  if (status === "running") return "var(--accent)";
  if (status === "completed") return "var(--ink)";
  if (status === "failed") return "var(--warn)";
  return "var(--muted-2)";
}

interface Props {
  events: DashboardLiveEvent[];
  decisions: string[];
  onAgentClick?: (agentId: string) => void;
}

export function AgentTimelineRail({ events, decisions, onAgentClick }: Props) {
  const { liveAgents, reasoning, context } = useMemo(() => digestEvents(events), [events]);
  const decisionList = useMemo(
    () => [...decisions].slice(-MAX_DECISIONS).reverse(),
    [decisions]
  );

  return (
    <aside className="timeline-rail" aria-label="Agent timeline">
      <header className="timeline-rail-head">
        <span className="eyebrow">Live agents</span>
        <span className="timeline-rail-count">{liveAgents.length}</span>
      </header>
      <ul className="timeline-list timeline-agents">
        {liveAgents.length === 0 ? (
          <li className="timeline-empty">Waiting for the first agent to start...</li>
        ) : (
          liveAgents.map((agent) => (
            <li
              key={agent.agentId}
              className={`timeline-agent timeline-agent-${agent.status}`}
              onClick={() => onAgentClick?.(agent.agentId)}
              tabIndex={onAgentClick ? 0 : -1}
              role={onAgentClick ? "button" : undefined}
            >
              <span className="timeline-agent-dot" style={{ background: statusDot(agent.status) }} />
              <div className="timeline-agent-body">
                <div className="timeline-agent-label">{agent.label}</div>
                {agent.currentTask ? (
                  <div className="timeline-agent-task">{agent.currentTask}</div>
                ) : null}
              </div>
              <span className="timeline-agent-time">{shortTime(agent.timestamp)}</span>
            </li>
          ))
        )}
      </ul>

      <header className="timeline-rail-head">
        <span className="eyebrow">Reasoning</span>
      </header>
      <ul className="timeline-list">
        {reasoning.length === 0 ? (
          <li className="timeline-empty">No reasoning steps yet.</li>
        ) : (
          reasoning.map((step, index) => (
            <li key={`${step.agentId}-${index}`} className="timeline-reason">
              <div className="timeline-reason-head">
                <span className="timeline-reason-agent">{step.agentLabel}</span>
                <span className="timeline-reason-type">{step.stepType}</span>
                <span className="timeline-reason-time">{shortTime(step.timestamp)}</span>
              </div>
              <div className="timeline-reason-title">{step.title}</div>
              {step.detail ? <div className="timeline-reason-detail">{step.detail}</div> : null}
            </li>
          ))
        )}
      </ul>

      <header className="timeline-rail-head">
        <span className="eyebrow">Context handoffs</span>
      </header>
      <ul className="timeline-list">
        {context.length === 0 ? (
          <li className="timeline-empty">No context handoffs yet.</li>
        ) : (
          context.map((item, index) => (
            <li key={`${item.agentId}-${index}`} className="timeline-context">
              <div className="timeline-context-head">
                <span className="timeline-context-route">
                  {item.fromAgentId ?? item.agentId}
                  {item.toAgentId ? ` -> ${item.toAgentId}` : ""}
                </span>
                <span className="timeline-context-type">{item.changeType}</span>
              </div>
              <div className="timeline-context-title">{item.title}</div>
              {item.detail ? <div className="timeline-context-detail">{item.detail}</div> : null}
            </li>
          ))
        )}
      </ul>

      <header className="timeline-rail-head">
        <span className="eyebrow">Decisions</span>
      </header>
      <ul className="timeline-list">
        {decisionList.length === 0 ? (
          <li className="timeline-empty">No decisions logged yet.</li>
        ) : (
          decisionList.map((decision, index) => (
            <li key={index} className="timeline-decision">{decision}</li>
          ))
        )}
      </ul>
    </aside>
  );
}
