"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { TreeNode as TreeNodeData, IterationView } from "../../../hooks/use-rlm-run";
import { usePan } from "../../../hooks/use-pan";
import { layoutTree, COLUMN_WIDTH, ROW_HEIGHT } from "./layout-tree";
import { TreeNode } from "./tree-node";
import { TreeEdges } from "./tree-edges";
import { NodeDetailPopup } from "./node-detail-popup";
import styles from "./exploration-canvas.module.css";

export interface ExplorationCanvasProps {
  tree: TreeNodeData[];
  iterations: IterationView[];
}

/** Sibling-candidate soft cap — fans wider than this collapse to a "+N more" node. */
const SOFT_CAP = 8;
/** Synthetic soft-cap node id prefix — never collides with a real node id. */
const SOFTCAP_PREFIX = "__softcap__:";
/** Padding around the laid-out scene so pan has somewhere to travel (spec §8). */
const SCENE_PADDING = 160;

/**
 * ExplorationCanvas — the live exploration-tree canvas (spec §8).
 *
 * Composes the already-built pieces:
 *   layoutTree → positioned nodes + edges
 *   TreeEdges  → the SVG edge layer (behind the nodes)
 *   TreeNode   → one node card per positioned node, absolutely placed
 *   NodeDetailPopup → the detail card for the selected node
 *
 * Behaviors:
 *   - Pan via the shared `usePan` hook (scroll-based drag-to-pan); a drag that
 *     moved the pointer suppresses the click so panning never selects a node.
 *   - Soft cap: a parent with more than 8 candidate children renders the first
 *     8 plus a "+N more" button; clicking it expands the rest.
 *   - Frontier auto-select: the deepest in-progress node (a running candidate,
 *     else the last visible node) is selected until the user clicks something.
 *   - Focus dance (§9): opening the popup moves focus into it; closing it
 *     restores focus to the TreeNode button that opened it.
 *   - New nodes fade in (200ms); the running pulse + fade respect
 *     prefers-reduced-motion (handled in CSS).
 */
export function ExplorationCanvas({ tree, iterations }: ExplorationCanvasProps) {
  // Parents whose collapsed candidate tail the user has chosen to expand.
  const [expandedFans, setExpandedFans] = useState<ReadonlySet<string>>(
    () => new Set()
  );

  // Selected node id. `null` means "follow the frontier" — once the user
  // clicks a node we hold that selection (userPicked flips true).
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [userPicked, setUserPicked] = useState(false);

  // Pan: shared hook. `wrapRef` is the scroll container, `dragRef.moved` tells
  // us whether the pointer moved enough to count as a pan (→ suppress click).
  const { wrapRef, dragRef, onPointerDown } = usePan();

  // Node-id → its TreeNode <button> element, for the focus-restore dance.
  const nodeElsRef = useRef<Map<string, HTMLButtonElement>>(new Map());
  // The node id whose button opened the currently-open popup.
  const triggerNodeIdRef = useRef<string | null>(null);

  // ── Soft-cap pre-processing ───────────────────────────────────────────────
  // Group candidate children by parent; for any fan wider than the cap that the
  // user has NOT expanded, keep the first 8 and append a synthetic "+N more"
  // node in place of the tail. declined-group nodes already arrive collapsed.
  const visibleTree = useMemo<TreeNodeData[]>(() => {
    // candidateChildren[parentId] = ordered ids of that parent's candidate kids.
    const candidateChildren = new Map<string, string[]>();
    for (const n of tree) {
      if (n.kind === "candidate" && n.parentId != null) {
        const list = candidateChildren.get(n.parentId) ?? [];
        list.push(n.id);
        candidateChildren.set(n.parentId, list);
      }
    }

    // hiddenIds = candidate ids past the cap in an un-expanded fan.
    const hiddenIds = new Set<string>();
    // softCapAfter[parentId] = how many candidates are hidden behind its "+N".
    const softCapCount = new Map<string, number>();
    for (const [parentId, kids] of candidateChildren) {
      if (kids.length > SOFT_CAP && !expandedFans.has(parentId)) {
        for (const id of kids.slice(SOFT_CAP)) hiddenIds.add(id);
        softCapCount.set(parentId, kids.length - SOFT_CAP);
      }
    }

    if (hiddenIds.size === 0) return tree;

    const out: TreeNodeData[] = [];
    const emittedSoftCap = new Set<string>();
    for (const n of tree) {
      if (hiddenIds.has(n.id)) {
        // First time we drop a hidden candidate for this parent, emit the
        // synthetic "+N more" node in its place (preserving rough position).
        const parentId = n.parentId!;
        if (!emittedSoftCap.has(parentId)) {
          emittedSoftCap.add(parentId);
          out.push({
            id: `${SOFTCAP_PREFIX}${parentId}`,
            kind: "declined-group",
            parentId,
            title: `+${softCapCount.get(parentId)} more`,
            iterationRange: [0, 0],
          });
        }
        continue; // drop the hidden candidate itself
      }
      out.push(n);
    }
    return out;
  }, [tree, expandedFans]);

  // ── Layout ────────────────────────────────────────────────────────────────
  const { positioned, edges } = useMemo(
    () => layoutTree(visibleTree),
    [visibleTree]
  );

  // Scene size: enough to contain every node + padding so pan has travel room.
  const sceneSize = useMemo(() => {
    let maxX = 0;
    let maxY = 0;
    for (const n of positioned) {
      if (n.x > maxX) maxX = n.x;
      if (n.y > maxY) maxY = n.y;
    }
    return {
      width: maxX + COLUMN_WIDTH + SCENE_PADDING * 2,
      height: maxY + ROW_HEIGHT + SCENE_PADDING * 2,
    };
  }, [positioned]);

  // ── Frontier node (auto-selection target) ─────────────────────────────────
  // The deepest in-progress node among the *visible*, *real* nodes: the last
  // running candidate, else the last visible real node. Scoping to visible
  // nodes guarantees the auto-selected node has a rendered button (so
  // focus-restore works and the popup anchors to something real); skipping the
  // synthetic "+N more" node keeps the popup off a non-move placeholder.
  const frontierNodeId = useMemo<string | null>(() => {
    if (positioned.length === 0) return null;
    for (let i = positioned.length - 1; i >= 0; i--) {
      const n = positioned[i];
      if (n.kind === "candidate" && n.outcome === "running") return n.id;
    }
    for (let i = positioned.length - 1; i >= 0; i--) {
      const n = positioned[i];
      if (!n.id.startsWith(SOFTCAP_PREFIX)) return n.id;
    }
    return null;
  }, [positioned]);

  // Effective selection: the user's pick once they've clicked, else the
  // frontier. Guard against a stale id that no longer resolves to a node.
  const effectiveSelectedId = useMemo<string | null>(() => {
    const candidate = userPicked ? selectedNodeId : frontierNodeId;
    if (candidate == null) return null;
    return positioned.some((n) => n.id === candidate) ? candidate : null;
  }, [userPicked, selectedNodeId, frontierNodeId, positioned]);

  const selectedNode = useMemo(
    () => positioned.find((n) => n.id === effectiveSelectedId) ?? null,
    [positioned, effectiveSelectedId]
  );

  // ── Iteration resolution for the popup ────────────────────────────────────
  // Match by iteration number inside the node's iterationRange; if several
  // qualify, take the most-recent (highest iteration). `null` when none match.
  const selectedIteration = useMemo<IterationView | null>(() => {
    if (!selectedNode) return null;
    const [lo, hi] = selectedNode.iterationRange;
    let best: IterationView | null = null;
    for (const it of iterations) {
      if (it.iteration >= lo && it.iteration <= hi) {
        if (best === null || it.iteration > best.iteration) best = it;
      }
    }
    return best;
  }, [selectedNode, iterations]);

  // ── Selection handlers ────────────────────────────────────────────────────
  const handleSelect = useCallback(
    (id: string) => {
      // A drag that moved the pointer is a pan, not a click — ignore it.
      if (dragRef.current.moved) return;
      triggerNodeIdRef.current = id;
      setSelectedNodeId(id);
      setUserPicked(true);
    },
    [dragRef]
  );

  const handleClosePopup = useCallback(() => {
    // Focus-restore half of the §9 focus dance: return focus to the TreeNode
    // button that opened the popup before clearing the selection.
    const trigger = triggerNodeIdRef.current;
    if (trigger) {
      nodeElsRef.current.get(trigger)?.focus();
    }
    triggerNodeIdRef.current = null;
    setSelectedNodeId(null);
    setUserPicked(true); // a deliberate close shouldn't snap back to the frontier
  }, []);

  // Expand a collapsed fan when its "+N more" node is clicked.
  const handleExpandFan = useCallback((parentId: string) => {
    setExpandedFans((prev) => {
      const next = new Set(prev);
      next.add(parentId);
      return next;
    });
  }, []);

  // Prune node-element refs that no longer correspond to a rendered node
  // (e.g. candidates hidden by a re-collapsed fan) so the map can't leak.
  useEffect(() => {
    const live = new Set(positioned.map((n) => n.id));
    for (const id of nodeElsRef.current.keys()) {
      if (!live.has(id)) nodeElsRef.current.delete(id);
    }
  }, [positioned]);

  // ── Initial framing ───────────────────────────────────────────────────────
  // Frame the tree on first render that has data, centering on the frontier
  // node (or the first positioned node as fallback). The guard ensures the
  // user's subsequent panning is never overridden as the tree grows.
  //
  // usePan's own mount-centering effect registers first (it fires before this
  // one); this effect runs after it and overrides the stale old-canvas offset.
  //
  // In jsdom (unit tests), clientWidth / clientHeight are 0, so the math
  // reduces to scroll = SCENE_PADDING + node position — harmless; nothing
  // visible to assert in a headless layout engine.
  const framedRef = useRef(false);
  useEffect(() => {
    if (framedRef.current) return;
    const wrap = wrapRef.current;
    if (!wrap || positioned.length === 0) return;

    const target =
      positioned.find((n) => n.id === frontierNodeId) ?? positioned[0];

    wrap.scrollLeft = Math.max(
      0,
      SCENE_PADDING + target.x + COLUMN_WIDTH / 2 - wrap.clientWidth / 2
    );
    wrap.scrollTop = Math.max(
      0,
      SCENE_PADDING + target.y + ROW_HEIGHT / 2 - wrap.clientHeight / 2
    );

    framedRef.current = true;
  }, [positioned, frontierNodeId, wrapRef]);

  // Register / unregister a TreeNode button element by node id.
  const registerNodeEl = useCallback(
    (id: string, el: HTMLButtonElement | null) => {
      if (el) nodeElsRef.current.set(id, el);
      else nodeElsRef.current.delete(id);
    },
    []
  );

  return (
    <div
      className={styles.canvas}
      ref={wrapRef}
      onPointerDown={onPointerDown}
      data-testid="exploration-canvas"
    >
      <div
        className={styles.scene}
        style={{ width: sceneSize.width, height: sceneSize.height }}
      >
        {/* Inner offset so the (0,0)-anchored tree sits inside the padding. */}
        <div
          className={styles.tree}
          style={{ transform: `translate(${SCENE_PADDING}px, ${SCENE_PADDING}px)` }}
        >
          {/* Edge layer — behind the nodes. */}
          <TreeEdges positioned={positioned} edges={edges} />

          {/* One node per positioned node, absolutely placed at {x,y}. */}
          {positioned.map((node) => {
            const isSoftCap = node.id.startsWith(SOFTCAP_PREFIX);
            const parentId = isSoftCap
              ? node.id.slice(SOFTCAP_PREFIX.length)
              : null;
            return (
              <div
                key={node.id}
                className={styles.nodeSlot}
                data-node
                style={{ left: node.x, top: node.y }}
              >
                {isSoftCap ? (
                  // Synthetic "+N more" node — a plain button that expands the fan.
                  <button
                    type="button"
                    className={styles.softCapNode}
                    onClick={() => {
                      if (dragRef.current.moved) return;
                      if (parentId) handleExpandFan(parentId);
                    }}
                  >
                    {node.title}
                  </button>
                ) : (
                  <TreeNodeRefWrapper
                    node={node}
                    selected={node.id === effectiveSelectedId}
                    onSelect={handleSelect}
                    register={registerNodeEl}
                  />
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Detail popup for the selected node. Keyed by node id so switching
          selection remounts it — re-running its focus-in effect (§9), so
          focus moves into the popup for every selection, not just the first. */}
      {selectedNode && (
        <NodeDetailPopup
          key={selectedNode.id}
          node={selectedNode}
          iteration={selectedIteration}
          onClose={handleClosePopup}
        />
      )}
    </div>
  );
}

/**
 * Thin wrapper that captures the rendered TreeNode <button> element so the
 * canvas can restore focus to it when the popup closes. TreeNode renders a
 * <button> as its root, so the slot's firstElementChild is that button.
 */
function TreeNodeRefWrapper({
  node,
  selected,
  onSelect,
  register,
}: {
  node: TreeNodeData;
  selected: boolean;
  onSelect: (id: string) => void;
  register: (id: string, el: HTMLButtonElement | null) => void;
}) {
  const slotRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const btn =
      (slotRef.current?.querySelector("button") as HTMLButtonElement | null) ??
      null;
    register(node.id, btn);
    return () => register(node.id, null);
  }, [node.id, register]);

  return (
    <div ref={slotRef} className={styles.fadeIn}>
      <TreeNode node={node} selected={selected} onSelect={onSelect} />
    </div>
  );
}
