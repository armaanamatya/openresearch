import type { PipelineNode, PipelineEdge, PipelineGate, PipelineTopology } from "./topology";

export interface LaidOutNode extends PipelineNode {
  x: number;
  y: number;
}

export interface LaidOutGate extends PipelineGate {
  x: number;
  y: number;
}

export interface Layout {
  nodes: LaidOutNode[];
  gates: LaidOutGate[];
  width: number;
  height: number;
}

export interface LayoutConfig {
  nodeWidth: number;
  nodeHeight: number;
  columnGap: number;
  rowGap: number;
  paddingX: number;
  paddingY: number;
}

const DEFAULT_CONFIG: LayoutConfig = {
  nodeWidth: 200,
  nodeHeight: 80,
  columnGap: 80,
  rowGap: 80,
  paddingX: 20,
  paddingY: 40
};

/**
 * Lay out the topology in stage-bucket columns. Source-typed and
 * agent-typed nodes go in the leftmost columns; improvement_path
 * nodes fan out into a vertical column at the next position; audit
 * and report nodes follow.
 *
 * The layout walks edges topologically: every node's column equals
 * `1 + max(predecessor.column)`. Within a column, improvement_path
 * nodes stack vertically; everything else centres on the column's
 * horizontal axis.
 */
export function layoutTopology(
  topology: PipelineTopology,
  config: Partial<LayoutConfig> = {}
): Layout {
  const cfg = { ...DEFAULT_CONFIG, ...config };

  // Topological columns via Kahn's algorithm.
  const columnByNode: Record<string, number> = {};
  const inDegree: Record<string, number> = {};
  for (const node of topology.nodes) {
    inDegree[node.id] = 0;
    columnByNode[node.id] = 0;
  }
  for (const edge of topology.edges) {
    inDegree[edge.target] = (inDegree[edge.target] ?? 0) + 1;
  }
  const queue: string[] = topology.nodes.filter((n) => inDegree[n.id] === 0).map((n) => n.id);
  while (queue.length) {
    const id = queue.shift()!;
    for (const edge of topology.edges) {
      if (edge.source !== id) continue;
      const nextCol = columnByNode[id] + 1;
      if (nextCol > columnByNode[edge.target]) {
        columnByNode[edge.target] = nextCol;
      }
      inDegree[edge.target] -= 1;
      if (inDegree[edge.target] === 0) queue.push(edge.target);
    }
  }

  // Bucket nodes by column.
  const columns: Record<number, PipelineNode[]> = {};
  for (const node of topology.nodes) {
    const col = columnByNode[node.id];
    (columns[col] ??= []).push(node);
  }

  // Y-position within each column. improvement_path nodes stack
  // vertically (one row each). source/agent/audit/report nodes
  // centre. env vs plan need to stack vertically too — they're
  // both at the same column. The rule: if a column has >1 node,
  // stack them.
  const laidOutNodes: LaidOutNode[] = [];
  let maxRight = 0;
  let maxBottom = 0;
  for (const colKey of Object.keys(columns).map(Number).sort((a, b) => a - b)) {
    const colNodes = columns[colKey];
    const x = cfg.paddingX + colKey * (cfg.nodeWidth + cfg.columnGap);
    // Centre the column vertically around y=320 (rough viewport mid).
    const totalHeight = colNodes.length * cfg.nodeHeight + (colNodes.length - 1) * cfg.rowGap;
    const startY = Math.max(cfg.paddingY, 320 - totalHeight / 2);
    colNodes.forEach((node, i) => {
      const y = startY + i * (cfg.nodeHeight + cfg.rowGap);
      laidOutNodes.push({ ...node, x, y });
      maxRight = Math.max(maxRight, x + cfg.nodeWidth);
      maxBottom = Math.max(maxBottom, y + cfg.nodeHeight);
    });
  }

  // Gate midpoints — between before_node and after_node.
  const byId: Record<string, LaidOutNode> = Object.fromEntries(
    laidOutNodes.map((n) => [n.id, n])
  );
  const laidOutGates: LaidOutGate[] = topology.gates.map((g) => {
    const a = byId[g.before_node];
    const b = byId[g.after_node];
    return {
      ...g,
      x: (a.x + cfg.nodeWidth + b.x) / 2,
      y: (a.y + cfg.nodeHeight / 2 + b.y + cfg.nodeHeight / 2) / 2
    };
  });

  return {
    nodes: laidOutNodes,
    gates: laidOutGates,
    width: maxRight + cfg.paddingX,
    height: maxBottom + cfg.paddingY
  };
}
