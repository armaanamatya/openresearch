"use client";

import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { TreeNode as TreeNodeData, IterationView } from "../../../hooks/use-rlm-run";
import { layoutConstellation, nodeRadius } from "./layout-constellation";
import type { ConstellationNode } from "./layout-constellation";
import styles from "./constellation-canvas.module.css";

export interface ConstellationCanvasProps {
  tree: TreeNodeData[];
  iterations: IterationView[];
  /** Externally controlled selected node id; null means no selection. */
  selectedNodeId?: string | null;
  /** Called when the user clicks a node. */
  onSelectNode?: (id: string) => void;
}

// ─── Constants ──────────────────────────────────────────────────────────────

const CANDIDATE_W = 140;
const CANDIDATE_H = 60;
const WORK_W = 120;
const WORK_H = 44;
/** Minimum bbox dimensions so a sparse 2-node graph doesn't look lost in space. */
const MIN_FIT_W = 600;
const MIN_FIT_H = 400;
const FIT_PADDING = 32;

// Node kinds that are structural (always visible in default view)
const STRUCTURAL_KINDS: TreeNodeData["kind"][] = ["paper", "work", "baseline", "candidate", "declined-group"];
// Node kinds that are detail/activity nodes (hidden until expanded)
const ACTIVITY_KINDS: TreeNodeData["kind"][] = ["primitive", "llm_primitive", "subrlm"];

// ─── BBox helpers ───────────────────────────────────────────────────────────

interface BBox {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

function computeBBox(nodes: ConstellationNode[]): BBox {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const n of nodes) {
    const isRect = STRUCTURAL_KINDS.includes(n.kind) && n.kind !== "paper";
    const hw = isRect ? (n.kind === "candidate" ? CANDIDATE_W / 2 : WORK_W / 2) : n.radius;
    const hh = isRect ? (n.kind === "candidate" ? CANDIDATE_H / 2 : WORK_H / 2) : n.radius;
    if (n.x - hw < minX) minX = n.x - hw;
    if (n.y - hh < minY) minY = n.y - hh;
    if (n.x + hw > maxX) maxX = n.x + hw;
    if (n.y + hh > maxY) maxY = n.y + hh;
  }
  return { minX, minY, maxX, maxY };
}

interface ViewBox { x: number; y: number; w: number; h: number }

function bboxToViewBox(bbox: BBox): ViewBox {
  const rawW = bbox.maxX - bbox.minX + FIT_PADDING * 2;
  const rawH = bbox.maxY - bbox.minY + FIT_PADDING * 2;
  const w = Math.max(rawW, MIN_FIT_W);
  const h = Math.max(rawH, MIN_FIT_H);
  // Center the content within the minimum bounding box
  const x = bbox.minX - FIT_PADDING - (w - rawW) / 2;
  const y = bbox.minY - FIT_PADDING - (h - rawH) / 2;
  return { x, y, w, h };
}

// ─── Activity count per structural node ─────────────────────────────────────

/** Returns a map from structural-node-id → activity node count. */
function computeActivityCounts(nodes: ConstellationNode[]): Map<string, number> {
  const counts = new Map<string, number>();
  // Build a map: structural node id → iterationRange
  const structuralByIter = new Map<string, [number, number]>();
  for (const n of nodes) {
    if (STRUCTURAL_KINDS.includes(n.kind)) {
      structuralByIter.set(n.id, n.iterationRange);
    }
  }
  // For each activity node, find the structural nodes whose iterationRange overlaps
  for (const n of nodes) {
    if (!ACTIVITY_KINDS.includes(n.kind)) continue;
    const [lo, hi] = n.iterationRange;
    for (const [sid, [slo, shi]] of structuralByIter) {
      if (lo <= shi && hi >= slo) {
        counts.set(sid, (counts.get(sid) ?? 0) + 1);
        break; // count each activity node once toward the first overlapping structural node
      }
    }
  }
  return counts;
}

// ─── Visibility logic ───────────────────────────────────────────────────────

/**
 * An activity node is visible iff at least one structural node whose iterationRange
 * overlaps the activity node's iterationRange is in expandedGroups.
 */
function isActivityNodeVisible(
  node: ConstellationNode,
  structuralNodes: ConstellationNode[],
  expandedGroups: Set<string>
): boolean {
  if (!ACTIVITY_KINDS.includes(node.kind)) return true;
  const [lo, hi] = node.iterationRange;
  for (const s of structuralNodes) {
    if (!expandedGroups.has(s.id)) continue;
    const [slo, shi] = s.iterationRange;
    if (lo <= shi && hi >= slo) return true;
  }
  return false;
}

// ─── Candidate/structural rect rendering ───────────────────────────────────

const NodeRect = memo(function NodeRect({
  node,
  selected,
  expanded,
  activityCount,
  onSelect,
  onToggleExpand,
}: {
  node: ConstellationNode;
  selected: boolean;
  expanded: boolean;
  activityCount: number;
  onSelect: (id: string) => void;
  onToggleExpand: (id: string) => void;
}) {
  const { kind, title, outcome } = node;
  const isCandidate = kind === "candidate";
  const w = isCandidate ? CANDIDATE_W : WORK_W;
  const h = isCandidate ? CANDIDATE_H : WORK_H;

  let stroke = "var(--line)";
  let fill = "var(--panel)";
  let textFill = "var(--ink)";
  if (kind === "baseline") stroke = "var(--accent)";
  if (kind === "subrlm") stroke = "var(--hermes)";
  if (kind === "paper") { stroke = "var(--line-2)"; }
  if (outcome === "promoted") stroke = "var(--accent)";
  if (outcome === "marginal") stroke = "var(--warn)";
  if (outcome === "failed") stroke = "var(--err)";
  if (outcome === "running") stroke = "var(--hermes)";
  if (outcome === "skipped" || outcome === "declined") { stroke = "var(--muted-2)"; textFill = "var(--muted)"; fill = "var(--chip)"; }
  if (selected) { stroke = "var(--accent)"; }
  if (kind === "work") fill = "var(--chip)";
  if (kind === "declined-group") { stroke = "var(--muted-2)"; fill = "var(--panel)"; textFill = "var(--muted)"; }

  const displayText =
    isCandidate && node.candidate?.displayTitle
      ? node.candidate.displayTitle
      : title;
  const label =
    displayText.length > 22 ? displayText.slice(0, 21) + "…" : displayText;

  const strokeWidth = selected ? 2.5 : 1.5;

  let dotFill: string | null = null;
  if (outcome === "promoted") dotFill = "var(--accent)";
  else if (outcome === "marginal") dotFill = "var(--warn)";
  else if (outcome === "failed") dotFill = "var(--err)";
  else if (outcome === "running") dotFill = "var(--hermes)";

  // Only show the expand toggle when there are activity nodes
  const showToggle = activityCount > 0;
  // Badge: "+N" when collapsed, "−" when expanded
  const badgeLabel = expanded ? "−" : `+${activityCount}`;

  // Badge position: bottom-left corner of the rect
  const badgeX = node.x - w / 2 + 4;
  const badgeY = node.y + h / 2 - 4;

  return (
    <g role="group" aria-label={outcome ? `${title} — ${outcome}` : title}>
      {/* Main node body — click to SELECT */}
      <g
        role="button"
        aria-label={`Select ${title}`}
        aria-pressed={selected}
        style={{ cursor: "pointer" }}
        onClick={() => onSelect(node.id)}
      >
        <rect
          x={node.x - w / 2}
          y={node.y - h / 2}
          width={w}
          height={h}
          rx={6}
          fill={fill}
          stroke={stroke}
          strokeWidth={strokeWidth}
          strokeDasharray={
            outcome === "skipped" || outcome === "declined" || kind === "subrlm" || kind === "declined-group"
              ? "4 3"
              : undefined
          }
        />
        {/* Selected glow */}
        {selected && (
          <rect
            x={node.x - w / 2 - 2}
            y={node.y - h / 2 - 2}
            width={w + 4}
            height={h + 4}
            rx={8}
            fill="none"
            stroke="var(--accent-soft)"
            strokeWidth={4}
            strokeOpacity={0.5}
            style={{ pointerEvents: "none" }}
          />
        )}
        <text
          x={node.x}
          y={node.y - (isCandidate ? 10 : 2)}
          textAnchor="middle"
          dominantBaseline="middle"
          fontSize={isCandidate ? 13 : 11}
          fontWeight={500}
          fill={textFill}
          fontFamily="var(--font-sans)"
          style={{ pointerEvents: "none" }}
        >
          {label}
        </text>
        {isCandidate && node.candidate?.category && (
          <text
            x={node.x}
            y={node.y + 8}
            textAnchor="middle"
            dominantBaseline="middle"
            fontSize={10}
            fill="var(--muted)"
            fontFamily="var(--font-mono)"
            style={{ pointerEvents: "none" }}
          >
            {node.candidate.category.slice(0, 18)}
          </text>
        )}
        {dotFill && (
          <circle
            cx={node.x + w / 2 - 8}
            cy={node.y - h / 2 + 8}
            r={4}
            fill={dotFill}
            style={{ pointerEvents: "none" }}
          />
        )}
      </g>

      {/* Activity badge / expand toggle — separate click target */}
      {showToggle && (
        <g
          role="button"
          aria-label={expanded ? "Collapse activity" : `Expand ${activityCount} activity nodes`}
          aria-pressed={expanded}
          style={{ cursor: "pointer" }}
          onClick={(e) => { e.stopPropagation(); onToggleExpand(node.id); }}
        >
          <rect
            x={badgeX}
            y={badgeY - 9}
            width={activityCount >= 10 ? 26 : 22}
            height={14}
            rx={4}
            fill={expanded ? "var(--accent)" : "var(--chip)"}
            stroke={expanded ? "var(--accent)" : "var(--line)"}
            strokeWidth={1}
          />
          <text
            x={badgeX + (activityCount >= 10 ? 13 : 11)}
            y={badgeY - 2}
            textAnchor="middle"
            dominantBaseline="middle"
            fontSize={8}
            fontWeight={600}
            fill={expanded ? "var(--canvas-bg)" : "var(--muted)"}
            fontFamily="var(--font-mono)"
            style={{ pointerEvents: "none" }}
          >
            {badgeLabel}
          </text>
        </g>
      )}
    </g>
  );
});

// ─── Circle node (primitive / llm_primitive / subrlm) ──────────────────────

const NodeCircle = memo(function NodeCircle({
  node,
  selected,
  visible,
  onSelect,
}: {
  node: ConstellationNode;
  selected: boolean;
  visible: boolean;
  onSelect: (id: string) => void;
}) {
  const { kind, title } = node;
  const r = nodeRadius(kind);

  let fill = "var(--chip)";
  let stroke = "var(--line)";
  if (kind === "llm_primitive") { fill = "var(--hermes-soft)"; stroke = "var(--hermes)"; }
  if (kind === "subrlm") { fill = "var(--hermes-soft)"; stroke = "var(--hermes)"; }
  if (selected) stroke = "var(--accent)";

  const isPulsing = kind === "llm_primitive" || kind === "subrlm";
  const showLabel = kind === "llm_primitive" || kind === "subrlm";
  const label = title.length > 12 ? title.slice(0, 11) + "…" : title;

  return (
    <g
      role="button"
      aria-label={title}
      aria-pressed={selected}
      style={{ cursor: "pointer" }}
      onClick={() => onSelect(node.id)}
      data-pulsing={isPulsing ? "true" : undefined}
      className={`${isPulsing ? styles.pulsingNode : ""} ${styles.activityNode} ${visible ? styles.activityNodeVisible : ""}`}
    >
      {selected && (
        <circle
          cx={node.x}
          cy={node.y}
          r={r + 5}
          fill="none"
          stroke="var(--accent-soft)"
          strokeWidth={4}
          strokeOpacity={0.5}
          style={{ pointerEvents: "none" }}
        />
      )}
      <circle
        cx={node.x}
        cy={node.y}
        r={r}
        fill={fill}
        stroke={stroke}
        strokeWidth={selected ? 2.5 : 1.5}
      />
      {showLabel && (
        <text
          x={node.x}
          y={node.y}
          textAnchor="middle"
          dominantBaseline="middle"
          fontSize={9}
          fill={kind === "llm_primitive" || kind === "subrlm" ? "var(--hermes-ink)" : "var(--muted)"}
          fontFamily="var(--font-mono)"
          style={{ pointerEvents: "none" }}
        >
          {label}
        </text>
      )}
    </g>
  );
});

// ─── ConstellationCanvas ────────────────────────────────────────────────────

/**
 * ConstellationCanvas — SVG constellation renderer for the RLM exploration tree.
 *
 * Progressive disclosure: default view shows only structural nodes (paper, work,
 * baseline, candidate, declined-group). Activity nodes (primitive, llm_primitive,
 * subrlm) are hidden until the user expands the associated structural node via
 * the "+N" badge. Auto-fits to viewport on load; stops auto-fitting once the user
 * has manually zoomed/panned.
 */
export const ConstellationCanvas = memo(function ConstellationCanvas({
  tree,
  selectedNodeId: externalSelectedNodeId = null,
  onSelectNode,
}: ConstellationCanvasProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // ── Layout ──────────────────────────────────────────────────────────────
  const { positioned, edges } = useMemo(
    () => layoutConstellation(tree),
    [tree]
  );

  // Derived sets
  const structuralNodes = useMemo(
    () => positioned.filter((n) => STRUCTURAL_KINDS.includes(n.kind)),
    [positioned]
  );

  const activityNodes = useMemo(
    () => positioned.filter((n) => ACTIVITY_KINDS.includes(n.kind)),
    [positioned]
  );

  const activityCounts = useMemo(
    () => computeActivityCounts(positioned),
    [positioned]
  );

  // ── Expand/collapse state ────────────────────────────────────────────────
  // Set of structural node ids whose activity nodes are currently visible.
  const [expandedGroups, setExpandedGroups] = useState<Set<string>>(new Set());

  const toggleGroup = useCallback((id: string) => {
    setExpandedGroups((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const allExpandable = structuralNodes.filter((n) => (activityCounts.get(n.id) ?? 0) > 0);
  const allExpanded = allExpandable.length > 0 && allExpandable.every((n) => expandedGroups.has(n.id));

  const toggleAll = useCallback(() => {
    setExpandedGroups(() => {
      if (allExpanded) return new Set();
      return new Set(allExpandable.map((n) => n.id));
    });
  }, [allExpanded, allExpandable]);

  // ── BBox and initial viewBox ─────────────────────────────────────────────
  // Use structural nodes only for the initial fit (activity nodes are hidden by default)
  const fitViewBox = useMemo<ViewBox>(() => {
    const fitNodes = structuralNodes.length > 0 ? structuralNodes : positioned;
    if (fitNodes.length === 0) return { x: 0, y: 0, w: MIN_FIT_W, h: MIN_FIT_H };
    return bboxToViewBox(computeBBox(fitNodes));
  }, [structuralNodes, positioned]);

  // ── ViewBox state for zoom/pan ─────────────────────────────────────────
  const [viewBox, setViewBox] = useState<ViewBox>(fitViewBox);

  // userInteracted: once the user manually zooms/pans, stop auto-re-fitting.
  const userInteractedRef = useRef(false);

  // Key-based auto-refit: when the structural layout changes (node count changes),
  // auto-refit if the user hasn't interacted yet.
  const lastFitKeyRef = useRef("");
  useEffect(() => {
    const key = `${fitViewBox.x.toFixed(1)},${fitViewBox.y.toFixed(1)},${fitViewBox.w.toFixed(1)},${fitViewBox.h.toFixed(1)}`;
    if (key !== lastFitKeyRef.current) {
      lastFitKeyRef.current = key;
      if (!userInteractedRef.current) {
        setViewBox(fitViewBox);
      }
    }
  }, [fitViewBox]);

  // ── Wheel zoom ────────────────────────────────────────────────────────
  const handleWheel = useCallback((e: React.WheelEvent<SVGSVGElement>) => {
    e.preventDefault();
    userInteractedRef.current = true;
    const factor = e.deltaY > 0 ? 1.1 : 0.9;
    setViewBox((vb) => {
      const svg = svgRef.current;
      if (!svg) return vb;
      const rect = svg.getBoundingClientRect();
      const mx = vb.x + ((e.clientX - rect.left) / rect.width) * vb.w;
      const my = vb.y + ((e.clientY - rect.top) / rect.height) * vb.h;
      const newW = vb.w * factor;
      const newH = vb.h * factor;
      return {
        x: mx - (mx - vb.x) * factor,
        y: my - (my - vb.y) * factor,
        w: newW,
        h: newH,
      };
    });
  }, []);

  // ── Pointer drag (pan) ────────────────────────────────────────────────
  const dragState = useRef<{ startX: number; startY: number; vbStart: ViewBox; moved: boolean } | null>(null);

  const handlePointerDown = useCallback((e: React.PointerEvent<SVGSVGElement>) => {
    if (e.button !== 0) return;
    dragState.current = { startX: e.clientX, startY: e.clientY, vbStart: viewBox, moved: false };
    // Capture is DEFERRED to the first real pan movement (see handlePointerMove):
    // capturing on pointerdown retargets the subsequent click event to the svg,
    // so the node <g onClick> handlers never fired and node selection was
    // silently broken for all users (audit 2026-06-10, found via Playwright).
  }, [viewBox]);

  const handlePointerMove = useCallback((e: React.PointerEvent<SVGSVGElement>) => {
    if (!dragState.current) return;
    if (e.buttons === 0) {
      // pointerup was missed (released outside the svg before capture was
      // taken — possible since capture is deferred to the first real pan
      // movement): clear the stale drag so a button-less hover can't
      // ghost-pan the canvas (audit 2026-06-11).
      dragState.current = null;
      return;
    }
    const dx = e.clientX - dragState.current.startX;
    const dy = e.clientY - dragState.current.startY;
    if (Math.abs(dx) > 3 || Math.abs(dy) > 3) {
      if (!dragState.current.moved) {
        // A real pan started — NOW capture so the drag keeps tracking outside
        // the svg. A plain click never reaches this branch, so its click event
        // still targets the node.
        (e.currentTarget as SVGSVGElement).setPointerCapture(e.pointerId);
      }
      dragState.current.moved = true;
      userInteractedRef.current = true;
    }
    if (!dragState.current.moved) return;
    const svg = svgRef.current;
    if (!svg) return;
    const rect = svg.getBoundingClientRect();
    const scaleX = dragState.current.vbStart.w / rect.width;
    const scaleY = dragState.current.vbStart.h / rect.height;
    setViewBox({
      ...dragState.current.vbStart,
      x: dragState.current.vbStart.x - dx * scaleX,
      y: dragState.current.vbStart.y - dy * scaleY,
    });
  }, []);

  const handlePointerUp = useCallback(() => {
    dragState.current = null;
  }, []);

  // ── Node click (suppressed on drag) ──────────────────────────────────
  const handleNodeSelect = useCallback((id: string) => {
    if (dragState.current?.moved) return;
    onSelectNode?.(id);
  }, [onSelectNode]);

  // ── Reset view — restore bbox-fit viewBox ─────────────────────────────
  const handleResetView = useCallback(() => {
    userInteractedRef.current = false;
    setViewBox(fitViewBox);
  }, [fitViewBox]);

  // ── Build node lookup for edges ───────────────────────────────────────
  const posById = useMemo(() => {
    const m = new Map<string, ConstellationNode>();
    for (const n of positioned) m.set(n.id, n);
    return m;
  }, [positioned]);

  // ── Visibility map for all nodes ─────────────────────────────────────
  const visibilityMap = useMemo(() => {
    const m = new Map<string, boolean>();
    for (const n of positioned) {
      if (STRUCTURAL_KINDS.includes(n.kind)) {
        m.set(n.id, true);
      } else {
        m.set(n.id, isActivityNodeVisible(n, structuralNodes, expandedGroups));
      }
    }
    return m;
  }, [positioned, structuralNodes, expandedGroups]);

  const svgViewBox = `${viewBox.x} ${viewBox.y} ${viewBox.w} ${viewBox.h}`;
  const hasActivity = allExpandable.length > 0;

  return (
    <div className={styles.wrapper} ref={containerRef} data-testid="constellation-canvas">
      <svg
        ref={svgRef}
        className={styles.svg}
        viewBox={svgViewBox}
        onWheel={handleWheel}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerCancel={handlePointerUp}
        onLostPointerCapture={handlePointerUp}
        style={{ cursor: dragState.current?.moved ? "grabbing" : "grab" }}
      >
        {/* Dot grid background */}
        <defs>
          <pattern id="constellation-grid" x="0" y="0" width="22" height="22" patternUnits="userSpaceOnUse">
            <circle cx="1" cy="1" r="1" fill="var(--canvas-grid)" />
          </pattern>
          <marker id="arrowhead" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
            <path d="M0,0 L6,3 L0,6 Z" fill="var(--muted-2)" />
          </marker>
        </defs>
        <rect
          x={viewBox.x}
          y={viewBox.y}
          width={viewBox.w}
          height={viewBox.h}
          fill="url(#constellation-grid)"
        />

        {/* Edges layer — only render edges where both endpoints are visible */}
        <g className={styles.edgesLayer}>
          {edges.map((edge) => {
            const src = posById.get(edge.from);
            const tgt = posById.get(edge.to);
            if (!src || !tgt) return null;
            // Hide edges when either endpoint is hidden (activity node collapsed)
            const srcVisible = visibilityMap.get(edge.from) ?? true;
            const tgtVisible = visibilityMap.get(edge.to) ?? true;
            if (!srcVisible || !tgtVisible) return null;

            const mx = (src.x + tgt.x) / 2;
            const my = (src.y + tgt.y) / 2;
            const len = Math.hypot(tgt.x - src.x, tgt.y - src.y);
            const curve = Math.min(len * 0.15, 30);
            const nx = -(tgt.y - src.y) / (len || 1);
            const ny = (tgt.x - src.x) / (len || 1);
            const cx = mx + nx * curve;
            const cy = my + ny * curve;

            let stroke = "var(--line)";
            let strokeOpacity = 0.5;
            if (edge.outcome === "promoted") { stroke = "var(--accent)"; strokeOpacity = 0.6; }
            if (edge.outcome === "failed") { stroke = "var(--err)"; strokeOpacity = 0.4; }

            const tgtNode = posById.get(edge.to);
            const isPrimEdge = tgtNode && (tgtNode.kind === "primitive" || tgtNode.kind === "llm_primitive");

            return (
              <path
                key={`${edge.from}-${edge.to}`}
                d={`M${src.x},${src.y} Q${cx},${cy} ${tgt.x},${tgt.y}`}
                fill="none"
                stroke={stroke}
                strokeOpacity={strokeOpacity}
                strokeWidth={isPrimEdge ? 0.8 : 1.5}
                strokeDasharray={
                  edge.outcome === "declined" || edge.outcome === "skipped" ? "4 3" : undefined
                }
              />
            );
          })}
        </g>

        {/* Nodes layer — activity circles first, then structural rects (rects on top) */}
        <g className={styles.nodesLayer}>
          {/* Activity nodes (primitive / llm_primitive / subrlm) */}
          {activityNodes.map((node) => {
            const visible = visibilityMap.get(node.id) ?? false;
            return (
              <NodeCircle
                key={node.id}
                node={node}
                selected={node.id === externalSelectedNodeId}
                visible={visible}
                onSelect={handleNodeSelect}
              />
            );
          })}
          {/* Structural rects */}
          {structuralNodes.map((node) => (
            <NodeRect
              key={node.id}
              node={node}
              selected={node.id === externalSelectedNodeId}
              expanded={expandedGroups.has(node.id)}
              activityCount={activityCounts.get(node.id) ?? 0}
              onSelect={handleNodeSelect}
              onToggleExpand={toggleGroup}
            />
          ))}
        </g>
      </svg>

      {/* Legend — fixed top-left with backdrop-blur */}
      <div className={styles.legend}>
        <span className={styles.legendItem}>
          <svg width="12" height="12" aria-hidden="true"><rect x="1" y="3" width="10" height="6" rx="1" fill="var(--panel)" stroke="var(--accent)" strokeWidth="1.5" /></svg>
          Candidate
        </span>
        <span className={styles.legendItem}>
          <svg width="12" height="12" aria-hidden="true"><rect x="1" y="3" width="10" height="6" rx="1" fill="var(--chip)" stroke="var(--line)" strokeWidth="1.5" /></svg>
          Work
        </span>
        <span className={styles.legendItem}>
          <svg width="12" height="12" aria-hidden="true"><circle cx="6" cy="6" r="5" fill="var(--hermes-soft)" stroke="var(--hermes)" strokeWidth="1.5" /></svg>
          LLM call
        </span>
        <span className={styles.legendItem}>
          <svg width="12" height="12" aria-hidden="true"><circle cx="6" cy="6" r="4" fill="var(--chip)" stroke="var(--line)" strokeWidth="1.5" /></svg>
          Tool call
        </span>
        {hasActivity && (
          <button
            type="button"
            className={styles.toggleAllBtn}
            onClick={toggleAll}
            aria-label={allExpanded ? "Hide all activity" : "Show all activity"}
          >
            {allExpanded ? "Hide all activity" : "Show all activity"}
          </button>
        )}
      </div>

      {/* Reset view button */}
      <button
        type="button"
        className={styles.resetBtn}
        onClick={handleResetView}
        aria-label="Reset view"
        title="Reset zoom/pan to fit"
      >
        ⊡
      </button>
    </div>
  );
});
