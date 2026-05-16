import type { LiveDemoRunState } from "@/lib/demo/demo-run-types";
import type { LaidOutNode } from "@/lib/pipeline/layout";

export type Tone = "accent" | "hermes" | "info" | "neutral";
export type NodeState = "done" | "running" | "upcoming";

/**
 * `WorkflowNode` is now an alias for the laid-out pipeline node.
 *
 * Pre-Phase-D the lab held a private NODES array (12 hand-placed
 * objects with `agent`, `x`, `y`, etc.). The canonical topology now
 * lives in `backend/agents/topology.py` and is served at
 * `GET /pipeline/topology`; the frontend lays it out with
 * `lib/pipeline/layout.ts::layoutTopology()`. Any consumer that used
 * to look at `WorkflowNode.agent` should read `node.internal_label`
 * or `node.demo_label` (presentation-mode aware) instead.
 */
export type WorkflowNode = LaidOutNode;

export function deriveStage(run: LiveDemoRunState | null): string | null {
  return run?.payload?.summary.stage ?? null;
}

/**
 * Compute the per-node {done, running, upcoming} state for a run.
 *
 * The `nodes` arg is the laid-out node list from the topology context —
 * pass `layout.nodes`. `improvementPathIds` is the topology's improvement-path
 * node id list — pass `topology.improvement_path_ids`. Stage-to-state mapping
 * is identical to the pre-D.2 implementation; only the source of node ids
 * changed (the old version iterated a hardcoded NODES const and a hardcoded
 * path-id list).
 */
export function stateMapForRun(
  run: LiveDemoRunState | null,
  nodes: LaidOutNode[],
  improvementPathIds: readonly string[]
): Record<string, NodeState> {
  const map = Object.fromEntries(nodes.map((node) => [node.id, "upcoming"])) as Record<
    string,
    NodeState
  >;

  if (!run) {
    return map;
  }

  const stage = deriveStage(run);
  const status = run.status;

  function mark(ids: string[], state: NodeState) {
    for (const id of ids) {
      map[id] = state;
    }
  }

  if (status === "queued" && !stage) {
    mark(["src"], "running");
    return map;
  }

  if (status === "completed") {
    mark(nodes.map((node) => node.id), "done");
    return map;
  }

  mark(["src"], "done");

  if (!stage) {
    mark(["read"], status === "failed" ? "done" : "running");
    return map;
  }

  if (stage === "ingested") {
    mark(["read"], "running");
    return map;
  }

  if (["paper_understood", "artifacts_discovered"].includes(stage)) {
    mark(["read"], "done");
    mark(["env", "plan"], "running");
    return map;
  }

  if (stage === "environment_built") {
    mark(["read"], "done");
    mark(["env"], "done");
    mark(["plan"], "running");
    return map;
  }

  if (["plan_created", "gate_1_passed"].includes(stage)) {
    // `env` finished at `environment_built` (an earlier stage) — it must stay
    // `done` here so the workflow counter never regresses (monotonic progress).
    mark(["read", "env"], "done");
    mark(["plan"], stage === "gate_1_passed" ? "done" : "running");
    return map;
  }

  if (["baseline_implemented", "baseline_run", "gate_2_passed"].includes(stage)) {
    mark(["read", "env", "plan"], "done");
    mark(["impl"], stage === "gate_2_passed" ? "done" : "running");
    return map;
  }

  if (["improvements_selected", "improvements_run", "gate_3_passed"].includes(stage)) {
    mark(["read", "env", "plan", "impl"], "done");
    const perPath = run.payload?.pathStates as Record<string, string> | undefined;
    for (const id of improvementPathIds) {
      const node = perPath?.[id];
      map[id] = node === "done" ? "done"
        : node === "running" || node === "attention" ? "running"
        : stage === "gate_3_passed" ? "done"
        : "upcoming";
    }
    return map;
  }

  if (stage === "research_map_generated") {
    mark(["read", "env", "plan", "impl", ...improvementPathIds], "done");
    mark(["audit"], "running");
    return map;
  }

  if (stage === "complete") {
    mark(nodes.map((node) => node.id), "done");
    return map;
  }

  mark(["read"], "running");
  return map;
}
