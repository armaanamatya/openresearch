"use client";

import { useEffect } from "react";

/**
 * Adds `scrolledClass` to the element with the given id once
 * window.scrollY > 8px. Mounted as a side-effect; returns nothing.
 *
 * We target by id rather than ref so the Nav can remain a server
 * component — a client-side useEffect can find the DOM node after
 * hydration without needing the ref wired through the server boundary.
 */
export function useNavScroll(targetId: string, scrolledClass: string): void {
  useEffect(() => {
    const el = document.getElementById(targetId);
    if (!el) return;
    const onScroll = () => {
      if (window.scrollY > 8) el.classList.add(scrolledClass);
      else el.classList.remove(scrolledClass);
    };
    document.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
    return () => document.removeEventListener("scroll", onScroll);
  }, [targetId, scrolledClass]);
}
