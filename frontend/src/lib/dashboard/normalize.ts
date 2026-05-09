import type {
  Citation,
  DashboardEvent,
  DashboardSnapshot,
  PipelineStage,
  TaskRecord,
  TaskStatus
} from "./contracts";

export interface TopologyNode {
  taskId: string;
  parentTaskId: string | null;
  agentId: string;
  title: string;
  stage: PipelineStage;
  status: TaskStatus;
  summary: string;
  depth: number;
  childTaskIds: string[];
  x: number;
  y: number;
}

export interface TopologyEdge {
  from: string;
  to: string;
}

export interface FeedItem {
  id: string;
  event: DashboardEvent["event"];
  title: string;
  summary: string;
  agentId: string;
  taskId: string;
  parentTaskId: string | null;
  timestamp: string;
  citations: Citation[];
}

export interface TaskDetail {
  task: TaskRecord;
  lineage: TaskRecord[];
  descendants: TaskRecord[];
  relatedFeed: FeedItem[];
  citations: Citation[];
}

export interface DashboardModel {
  snapshot: DashboardSnapshot;
  tasksById: Record<string, TaskRecord>;
  orderedTasks: TaskRecord[];
  topologyNodes: TopologyNode[];
  topologyEdges: TopologyEdge[];
  feed: FeedItem[];
  detailsByTaskId: Record<string, TaskDetail>;
  citationsBySourceId: Record<string, Citation>;
}

const stageOrder: PipelineStage[] = ["plan", "baseline", "improvement", "verification"];

function sortIsoDesc(left: string, right: string) {
  return right.localeCompare(left);
}

function dedupeCitations(citations: Citation[]) {
  const byId = new Map<string, Citation>();

  for (const citation of citations) {
    byId.set(citation.source_id, citation);
  }

  return Array.from(byId.values());
}

function assertNever(value: never): never {
  throw new Error(`Unhandled dashboard event: ${JSON.stringify(value)}`);
}

function buildFeedItem(event: DashboardEvent, tasksById: Record<string, TaskRecord>): FeedItem {
  const task = tasksById[event.task_id];

  switch (event.event) {
    case "agent_started":
    case "agent_completed":
    case "agent_failed":
      return {
        id: `${event.event}:${event.task_id}:${event.timestamp}`,
        event: event.event,
        title: `${task?.title ?? event.agent_id} ${event.event.replace("agent_", "").replace("_", " ")}`,
        summary: event.summary,
        agentId: event.agent_id,
        taskId: event.task_id,
        parentTaskId: event.parent_task_id ?? null,
        timestamp: event.timestamp,
        citations: dedupeCitations(event.citations ?? task?.citations ?? [])
      };
    case "agent_reasoning_step":
    case "rlm_query_executed":
    case "semantic_search_executed":
      return {
        id: `${event.event}:${event.task_id}:${event.timestamp}`,
        event: event.event,
        title: `${event.step_type.replace("_", " ")} in ${task?.title ?? event.agent_id}`,
        summary: `${event.query} -> ${event.result}`,
        agentId: event.agent_id,
        taskId: event.task_id,
        parentTaskId: event.parent_task_id ?? null,
        timestamp: event.timestamp,
        citations: dedupeCitations(event.citations ?? [])
      };
    case "shared_state_updated":
      return {
        id: `${event.event}:${event.task_id}:${event.timestamp}`,
        event: event.event,
        title: `Shared state updated: ${event.record_type}`,
        summary: `${task?.title ?? event.agent_id} published ${event.scope} output with status ${event.status}.`,
        agentId: event.agent_id,
        taskId: event.task_id,
        parentTaskId: event.parent_task_id ?? null,
        timestamp: event.timestamp,
        citations: dedupeCitations(event.citations ?? task?.citations ?? [])
      };
    case "verification_gate_result":
      return {
        id: `${event.event}:${event.task_id}:${event.timestamp}`,
        event: event.event,
        title: `Verification gate: ${event.gate}`,
        summary: `${event.outcome.toUpperCase()}: ${event.details}`,
        agentId: event.agent_id,
        taskId: event.task_id,
        parentTaskId: event.parent_task_id ?? null,
        timestamp: event.timestamp,
        citations: dedupeCitations(event.citations ?? [])
      };
    case "approval_requested":
    case "approval_resolved":
      return {
        id: `${event.event}:${event.task_id}:${event.timestamp}`,
        event: event.event,
        title: `${event.approval_label} ${event.state}`,
        summary: event.details,
        agentId: event.agent_id,
        taskId: event.task_id,
        parentTaskId: event.parent_task_id ?? null,
        timestamp: event.timestamp,
        citations: dedupeCitations(event.citations ?? [])
      };
    case "context_enrichment":
      return {
        id: `${event.event}:${event.task_id}:${event.timestamp}`,
        event: event.event,
        title: `Context enrichment: ${event.variable_name}`,
        summary: event.details,
        agentId: event.agent_id,
        taskId: event.task_id,
        parentTaskId: event.parent_task_id ?? null,
        timestamp: event.timestamp,
        citations: dedupeCitations(event.citations ?? task?.citations ?? [])
      };
  }

  return assertNever(event);
}

function sortTasks(tasks: TaskRecord[]) {
  return [...tasks].sort((left, right) => {
    const depthDelta = left.delegation_depth - right.delegation_depth;

    if (depthDelta !== 0) {
      return depthDelta;
    }

    const stageDelta = stageOrder.indexOf(left.stage) - stageOrder.indexOf(right.stage);

    if (stageDelta !== 0) {
      return stageDelta;
    }

    return left.published_at.localeCompare(right.published_at);
  });
}

export function buildDashboardModel(snapshot: DashboardSnapshot): DashboardModel {
  const orderedTasks = sortTasks(snapshot.tasks);
  const tasksById = Object.fromEntries(orderedTasks.map((task) => [task.task_id, task]));
  const childrenByTaskId = new Map<string, string[]>();
  const citationsBySourceId: Record<string, Citation> = {};

  for (const task of orderedTasks) {
    childrenByTaskId.set(task.task_id, []);

    for (const citation of task.citations) {
      citationsBySourceId[citation.source_id] = citation;
    }
  }

  for (const task of orderedTasks) {
    if (task.parent_task_id && childrenByTaskId.has(task.parent_task_id)) {
      childrenByTaskId.get(task.parent_task_id)?.push(task.task_id);
    }
  }

  const siblingIndexByTaskId = new Map<string, number>();

  for (const [parentTaskId, childTaskIds] of childrenByTaskId.entries()) {
    childTaskIds.forEach((taskId, index) => {
      siblingIndexByTaskId.set(taskId, index);
    });

    if (!tasksById[parentTaskId].parent_task_id) {
      siblingIndexByTaskId.set(parentTaskId, 0);
    }
  }

  const topologyNodes = orderedTasks.map((task) => {
    const siblings = task.parent_task_id
      ? childrenByTaskId.get(task.parent_task_id) ?? [task.task_id]
      : orderedTasks.filter((candidate) => candidate.parent_task_id === null).map((candidate) => candidate.task_id);
    const siblingIndex = siblings.indexOf(task.task_id);
    const spread = siblings.length > 1 ? 1 / (siblings.length - 1) : 0.5;
    const normalizedX = siblings.length > 1 ? siblingIndex * spread : 0.5;

    return {
      taskId: task.task_id,
      parentTaskId: task.parent_task_id,
      agentId: task.agent_id,
      title: task.title,
      stage: task.stage,
      status: task.status,
      summary: task.summary,
      depth: task.delegation_depth,
      childTaskIds: childrenByTaskId.get(task.task_id) ?? [],
      x: 80 + normalizedX * 520 + task.delegation_depth * 24,
      y: 84 + task.delegation_depth * 128 + (siblingIndexByTaskId.get(task.task_id) ?? 0) * 20
    } satisfies TopologyNode;
  });

  const topologyEdges = topologyNodes
    .filter((node) => node.parentTaskId)
    .map((node) => ({
      from: node.parentTaskId ?? "",
      to: node.taskId
    }));

  const feed = snapshot.events
    .map((event) => {
      const citations = event.citations ?? [];

      citations.forEach((citation) => {
        citationsBySourceId[citation.source_id] = citation;
      });

      return buildFeedItem(event, tasksById);
    })
    .sort((left, right) => sortIsoDesc(left.timestamp, right.timestamp));

  const detailsByTaskId = Object.fromEntries(
    orderedTasks.map((task) => {
      const lineage: TaskRecord[] = [];
      let cursor: TaskRecord | undefined = task;

      while (cursor) {
        lineage.unshift(cursor);
        cursor = cursor.parent_task_id ? tasksById[cursor.parent_task_id] : undefined;
      }

      const descendants = orderedTasks.filter((candidate) => {
        let nextParent = candidate.parent_task_id;

        while (nextParent) {
          if (nextParent === task.task_id) {
            return true;
          }

          nextParent = tasksById[nextParent]?.parent_task_id ?? null;
        }

        return false;
      });

      const relatedFeed = feed.filter((item) => item.taskId === task.task_id);
      const citations = dedupeCitations([
        ...task.citations,
        ...relatedFeed.flatMap((item) => item.citations)
      ]);

      return [
        task.task_id,
        {
          task,
          lineage,
          descendants,
          relatedFeed,
          citations
        } satisfies TaskDetail
      ];
    })
  );

  return {
    snapshot,
    tasksById,
    orderedTasks,
    topologyNodes,
    topologyEdges,
    feed,
    detailsByTaskId,
    citationsBySourceId
  };
}
