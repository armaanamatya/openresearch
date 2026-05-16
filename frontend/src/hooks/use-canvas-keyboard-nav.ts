"use client";

import { useEffect } from "react";

/**
 * j/k (or arrow keys) step through the selection in the order supplied
 * by the caller; Esc clears it. Modifier presses and typing inside an
 * input/textarea are ignored so the shortcut never interferes with the
 * command palette or any form.
 *
 * `order` must be the desired traversal sequence of node ids — usually
 * computed from the laid-out topology (sort by x, then y) by the
 * caller so adding a node in topology.py reflows the keyboard order
 * automatically.
 */
export function useCanvasKeyboardNav({
  selectedId,
  onSelect,
  enabled,
  order
}: {
  selectedId: string | null;
  onSelect: (id: string | null) => void;
  enabled: boolean;
  order: string[];
}) {
  useEffect(() => {
    if (!enabled) return;
    function onKey(e: KeyboardEvent) {
      const target = e.target as HTMLElement | null;
      const tag = target?.tagName ?? "";
      if (tag === "INPUT" || tag === "TEXTAREA" || target?.isContentEditable) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (order.length === 0) return;
      const idx = selectedId ? order.indexOf(selectedId) : -1;
      if (e.key === "j" || e.key === "ArrowDown" || e.key === "ArrowRight") {
        e.preventDefault();
        onSelect(order[Math.min(order.length - 1, idx + 1)] ?? order[0]);
      } else if (e.key === "k" || e.key === "ArrowUp" || e.key === "ArrowLeft") {
        e.preventDefault();
        onSelect(order[Math.max(0, idx - 1)] ?? order[order.length - 1]);
      } else if (e.key === "Escape") {
        onSelect(null);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selectedId, onSelect, enabled, order]);
}
