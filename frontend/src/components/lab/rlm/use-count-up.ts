"use client";

import { useEffect, useRef, useState } from "react";

/**
 * useCountUp — tween a numeric value to a target with cubic-out easing.
 *
 * Returns the currently-displayed value. The tween restarts whenever
 * `target` changes; cancelled if the component unmounts.
 *
 * Spec: docs/superpowers/specs/2026-05-23-rubric-climb-leaderboard.md §4.3.
 */
export function useCountUp(target: number, durationMs = 400): number {
  // The displayed value. Initialised at `target` so the very first render
  // matches the caller's intent without any visible animation.
  const [value, setValue] = useState(target);
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    // Snapshot via setValue's previous-state callback so we read the latest
    // displayed value without depending on `value` in the effect deps.
    let from: number | null = null;
    const startMs = performance.now();

    const step = (now: number) => {
      const elapsed = now - startMs;
      const t = Math.min(1, elapsed / durationMs);
      const eased = 1 - Math.pow(1 - t, 3); // cubic-out

      setValue((prev) => {
        if (from === null) from = prev;
        return from + (target - from) * eased;
      });

      if (t < 1) {
        rafRef.current = requestAnimationFrame(step);
      } else {
        rafRef.current = null;
      }
    };

    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
    }
    rafRef.current = requestAnimationFrame(step);

    return () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
  }, [target, durationMs]);

  return value;
}
