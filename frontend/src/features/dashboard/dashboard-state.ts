import type {
  AgentNode,
  ApprovalEntry,
  ContextEnrichmentEvent,
  DashboardEvent,
  DashboardSnapshot,
  DataPanel,
  MessageEntry,
  ProgressEntry,
  QueryExecutedEvent,
  ReasoningEntry,
  SharedStateUpdatedEvent
} from "@/lib/events/contract";

function upsertById<T extends { id: string }>(items: T[], item: T): T[] {
  const next = items.filter((current) => current.id !== item.id);
  return [item, ...next];
}

function mergeAgent(agents: AgentNode[], agent: AgentNode): AgentNode[] {
  return upsertById(agents, agent);
}

function appendCitations(
  citations: DashboardSnapshot["citations"],
  nextCitations: DashboardSnapshot["citations"]
): DashboardSnapshot["citations"] {
  const merged = [...nextCitations, ...citations];
  const unique = new Map(merged.map((citation) => [citation.id, citation]));
  return Array.from(unique.values());
}

function pushReasoning(
  reasoning: ReasoningEntry[],
  entry: ReasoningEntry
): ReasoningEntry[] {
  return upsertById(reasoning, entry);
}

function pushMessage(messages: MessageEntry[], entry: MessageEntry): MessageEntry[] {
  return upsertById(messages, entry);
}

function upsertApproval(
  approvals: ApprovalEntry[],
  approval: ApprovalEntry
): ApprovalEntry[] {
  return upsertById(approvals, approval);
}

function updateProgress(
  progress: ProgressEntry[],
  entry: ProgressEntry
): ProgressEntry[] {
  const next = progress.filter((current) => current.stage !== entry.stage);
  return [...next, entry];
}

function prependPanelItem(panel: DataPanel, summary: string): DataPanel {
  return {
    ...panel,
    items: [summary, ...panel.items.filter((item) => item !== summary)].slice(0, 5)
  };
}

function updateDataPanel(
  panels: DataPanel[],
  panelId: string,
  summary: string
): DataPanel[] {
  return panels.map((panel) => {
    if (panel.id !== panelId) {
      return panel;
    }

    return prependPanelItem(panel, summary);
  });
}

function toQueryReasoningEntry(event: QueryExecutedEvent): ReasoningEntry {
  return {
    id: `${event.event}-${event.timestamp}`,
    agentId: event.agentId,
    agentLabel: event.agentLabel,
    title:
      event.event === "rlm_query_executed"
        ? "RLM query executed"
        : "Semantic search executed",
    detail: `${event.query} -> ${event.result}`,
    stepType: event.event,
    timestamp: event.timestamp,
    citations: event.citations
  };
}

function toSharedStateMessage(event: SharedStateUpdatedEvent): MessageEntry {
  return {
    id: `${event.changeType}-${event.timestamp}`,
    fromAgentId: event.fromAgentId ?? event.agentId,
    toAgentId: event.toAgentId ?? "shared-state",
    summary: event.title,
    detail: event.detail,
    timestamp: event.timestamp
  };
}

function applyContextEnrichment(
  snapshot: DashboardSnapshot,
  event: ContextEnrichmentEvent
): DashboardSnapshot {
  return {
    ...snapshot,
    agents: snapshot.agents.map((agent) => {
      if (agent.id !== event.agentId) {
        return agent;
      }

      return {
        ...agent,
        contextVariables: [
          event.variableName,
          ...agent.contextVariables.filter((item) => item !== event.variableName)
        ],
        lastUpdated: event.timestamp
      };
    }),
    dataPanels: updateDataPanel(snapshot.dataPanels, "artifacts", event.summary)
  };
}

export function applyDashboardEvent(
  snapshot: DashboardSnapshot,
  event: DashboardEvent
): DashboardSnapshot {
  switch (event.event) {
    case "agent_started":
    case "agent_completed":
    case "agent_failed":
      return {
        ...snapshot,
        agents: mergeAgent(snapshot.agents, event.agent)
      };
    case "agent_reasoning_step":
      return {
        ...snapshot,
        reasoning: pushReasoning(snapshot.reasoning, {
          id: `${event.agentId}-${event.timestamp}`,
          agentId: event.agentId,
          agentLabel: event.agentLabel,
          title: event.title,
          detail: event.detail,
          stepType: event.stepType,
          timestamp: event.timestamp,
          citations: event.citations
        }),
        citations: appendCitations(snapshot.citations, event.citations)
      };
    case "rlm_query_executed":
    case "semantic_search_executed":
      return {
        ...snapshot,
        reasoning: pushReasoning(snapshot.reasoning, toQueryReasoningEntry(event)),
        citations: appendCitations(snapshot.citations, event.citations)
      };
    case "shared_state_updated":
      return {
        ...snapshot,
        messages:
          event.changeType === "message"
            ? pushMessage(snapshot.messages, toSharedStateMessage(event))
            : snapshot.messages,
        dataPanels:
          event.changeType === "assumption"
            ? updateDataPanel(snapshot.dataPanels, "assumptions", event.detail)
            : snapshot.dataPanels
      };
    case "verification_gate_result":
      return {
        ...snapshot,
        progress: updateProgress(snapshot.progress, {
          stage: event.stage,
          status: event.status,
          detail: event.detail
        })
      };
    case "approval_requested":
      return {
        ...snapshot,
        approvals: upsertApproval(snapshot.approvals, event.approval)
      };
    case "approval_resolved":
      return {
        ...snapshot,
        approvals: snapshot.approvals.map((approval) =>
          approval.id === event.approvalId
            ? {
                ...approval,
                status: event.status,
                detail: event.detail,
                timestamp: event.timestamp
              }
            : approval
        )
      };
    case "context_enrichment":
      return applyContextEnrichment(snapshot, event);
    default:
      return snapshot;
  }
}
