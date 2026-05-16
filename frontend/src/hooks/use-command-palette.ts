"use client";

import { useEffect, useState } from "react";

/**
 * Global Cmd/Ctrl-K toggle for the command palette. Esc always closes.
 * Listens at the window level so the shortcut works regardless of which
 * view (upload or workflow) the lab is currently rendering.
 */
export function useCommandPalette() {
  const [open, setOpen] = useState(false);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        setOpen((v) => !v);
      }
      if (e.key === "Escape") {
        setOpen(false);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return { open, setOpen };
}
