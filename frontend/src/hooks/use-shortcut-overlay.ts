"use client";

import { useEffect, useState } from "react";

/**
 * Toggles the shortcut-help overlay on `?` press. Esc always closes.
 * Mirrors the input-suppression rules from useCanvasKeyboardNav so
 * typing `?` inside an input or contenteditable does not open the
 * overlay.
 */
export function useShortcutOverlay() {
  const [open, setOpen] = useState(false);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") {
        setOpen(false);
        return;
      }
      if (e.key !== "?") return;
      const target = e.target as HTMLElement | null;
      const tag = target?.tagName ?? "";
      if (tag === "INPUT" || tag === "TEXTAREA" || target?.isContentEditable) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      e.preventDefault();
      setOpen((v) => !v);
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return { open, setOpen };
}
